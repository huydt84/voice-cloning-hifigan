# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Dropout

from vocoder.hifigan import Generator
from spk_enc.encoder.params_model import model_embedding_size as spkr_embedding_dim

class SpeakerEncoder(torch.nn.Module):
    def __init__(self, mel_n_channels=128, model_num_layers=3, model_hidden_size=256, model_embedding_size=256):
        super(SpeakerEncoder, self).__init__()
        self.lstm = nn.LSTM(mel_n_channels, model_hidden_size, model_num_layers, batch_first=True)
        self.linear = nn.Linear(model_hidden_size, model_embedding_size)
        self.relu = nn.ReLU()

    def forward(self, mels):
        self.lstm.flatten_parameters()
        _, (hidden, _) = self.lstm(mels)
        embeds_raw = self.relu(self.linear(hidden[-1]))
        return embeds_raw / torch.norm(embeds_raw, dim=1, keepdim=True)
        
    def compute_partial_slices(self, total_frames, partial_frames, partial_hop):
        mel_slices = []
        for i in range(0, total_frames-partial_frames, partial_hop):
            mel_range = torch.arange(i, i+partial_frames)
            mel_slices.append(mel_range)
            
        return mel_slices
    
    def embed_utterance(self, mel, partial_frames=128, partial_hop=64):
        mel_len = mel.size(1)
        last_mel = mel[:,-partial_frames:]
        
        if mel_len > partial_frames:
            mel_slices = self.compute_partial_slices(mel_len, partial_frames, partial_hop)
            mels = list(mel[:,s] for s in mel_slices)
            mels.append(last_mel)
            mels = torch.stack(tuple(mels), 0).squeeze(1)
        
            with torch.no_grad():
                partial_embeds = self(mels)
            embed = torch.mean(partial_embeds, axis=0).unsqueeze(0)
            #embed = embed / torch.linalg.norm(embed, 2)
        else:
            with torch.no_grad():
                embed = self(last_mel)
        
        return embed



class VariancePredictor(nn.Module):
    def __init__(
        self,
        encoder_embed_dim: int,
        var_pred_hidden_dim: int,
        var_pred_kernel_size: int,
        var_pred_dropout: float,
    ):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(
                encoder_embed_dim,
                var_pred_hidden_dim,
                kernel_size=var_pred_kernel_size,
                padding=(var_pred_kernel_size - 1) // 2,
            ),
            nn.ReLU(),
        )
        self.ln1 = nn.LayerNorm(var_pred_hidden_dim)
        self.dropout_module = Dropout(p=var_pred_dropout)
        self.conv2 = nn.Sequential(
            nn.Conv1d(
                var_pred_hidden_dim,
                var_pred_hidden_dim,
                kernel_size=var_pred_kernel_size,
                padding=1,
            ),
            nn.ReLU(),
        )
        self.ln2 = nn.LayerNorm(var_pred_hidden_dim)
        self.proj = nn.Linear(var_pred_hidden_dim, 1)

    def forward(self, x: Tensor) -> Any:
        # Input: B x T x C; Output: B x T
        x = self.conv1(x.transpose(1, 2)).transpose(1, 2)
        x = self.dropout_module(self.ln1(x))
        x = self.conv2(x.transpose(1, 2)).transpose(1, 2)
        x = self.dropout_module(self.ln2(x))
        return self.proj(x).squeeze(dim=2)


class CodeGenerator(Generator):
    def __init__(
        self,
        upsample_rates: List[int],
        upsample_kernel_sizes: List[int],
        upsample_initial_channel: int,
        resblock_kernel_sizes: List[int],
        resblock_dilation_sizes: List[List[int]],
        model_in_dim: Optional[int],
        num_embeddings: int,
        embedding_dim: int,
        dur_predictor_params: Dict[str, Any],
        lang_embedding_dim: int,
        num_langs: int,
        spkr_embedding_dim: int,
        num_spkrs: int,
    ):
        super().__init__(
            upsample_rates,
            upsample_kernel_sizes,
            upsample_initial_channel,
            resblock_kernel_sizes,
            resblock_dilation_sizes,
            model_in_dim,
        )
        self.dict = nn.Embedding(num_embeddings, embedding_dim)
        self.spkr = nn.Embedding(num_spkrs, spkr_embedding_dim)
        self.lang = nn.Embedding(num_langs, lang_embedding_dim)

        self.dur_predictor = None
        if dur_predictor_params:
            self.dur_predictor = VariancePredictor(**dur_predictor_params)

        self.num_spkrs = num_spkrs
        self.num_langs = num_langs

    @staticmethod
    def _upsample(signal: Tensor, max_frames: int) -> Tensor:
        if signal.dim() == 3:
            bsz, channels, cond_length = signal.size()
        elif signal.dim() == 2:
            signal = signal.unsqueeze(2)
            bsz, channels, cond_length = signal.size()
        else:
            signal = signal.view(-1, 1, 1)
            bsz, channels, cond_length = signal.size()

        signal = signal.unsqueeze(3).repeat(1, 1, 1, max_frames // cond_length)

        # pad zeros as needed (if signal's shape does not divide completely with max_frames)
        reminder = (max_frames - signal.shape[2] * signal.shape[3]) // signal.shape[3]
        if reminder > 0:
            raise NotImplementedError(
                "Padding condition signal - misalignment between condition features."
            )

        signal = signal.view(bsz, channels, max_frames)
        return signal

    def forward(self, sample: Dict[str, Any], dur_prediction: bool) -> Tensor:  # type: ignore
        x = sample["code"].clone().to(device=self.dict.weight.device)
        x = self.dict(x).transpose(1, 2)

        if self.dur_predictor and dur_prediction:
            assert x.size(0) == 1, "only support single sample"
            log_dur_pred = self.dur_predictor(x.transpose(1, 2))
            dur_out = torch.clamp(
                torch.round((torch.exp(log_dur_pred) - 1)).long(), min=1
            )
            # B x C x T
            x = torch.repeat_interleave(x, dur_out.view(-1), dim=2)

        spkr = self.spkr(sample["spkr"].to(self.spkr.weight.device)).transpose(1, 2)
        spkr = self._upsample(spkr, x.shape[-1])
        x = torch.cat([x, spkr], dim=1)

        lang = self.lang(sample["lang"].to(self.lang.weight.device)).transpose(1, 2)
        lang = self._upsample(lang, x.shape[-1])
        x = torch.cat([lang, x], dim=1)

        return super().forward(x)
    

class CustomCodeGenerator(Generator):
    def __init__(
        self,
        upsample_rates: List[int],
        upsample_kernel_sizes: List[int],
        upsample_initial_channel: int,
        resblock_kernel_sizes: List[int],
        resblock_dilation_sizes: List[List[int]],
        model_in_dim: Optional[int],
        num_embeddings: int,
        embedding_dim: int,
        dur_predictor_params: Dict[str, Any],
        lang_embedding_dim: int,
        num_langs: int,
        spkr_embedding_dim: int,
        num_spkrs: int,
    ):
        super().__init__(
            upsample_rates,
            upsample_kernel_sizes,
            upsample_initial_channel,
            resblock_kernel_sizes,
            resblock_dilation_sizes,
            model_in_dim,
        )
        self.dict = nn.Embedding(num_embeddings, embedding_dim)
        self.spkr = nn.Embedding(num_spkrs, spkr_embedding_dim)
        self.lang = nn.Embedding(num_langs, lang_embedding_dim)

        self.dur_predictor = None
        if dur_predictor_params:
            self.dur_predictor = VariancePredictor(**dur_predictor_params)

        self.num_spkrs = num_spkrs
        self.num_langs = num_langs

    @staticmethod
    def _upsample(signal: Tensor, max_frames: int) -> Tensor:
        if signal.dim() == 3:
            bsz, channels, cond_length = signal.size()
        elif signal.dim() == 2:
            signal = signal.unsqueeze(2)
            bsz, channels, cond_length = signal.size()
        else:
            signal = signal.view(-1, 1, 1)
            bsz, channels, cond_length = signal.size()

        signal = signal.unsqueeze(3).repeat(1, 1, 1, max_frames // cond_length)

        # pad zeros as needed (if signal's shape does not divide completely with max_frames)
        reminder = (max_frames - signal.shape[2] * signal.shape[3]) // signal.shape[3]
        if reminder > 0:
            raise NotImplementedError(
                "Padding condition signal - misalignment between condition features."
            )

        signal = signal.view(bsz, channels, max_frames)
        return signal

    def forward(self, sample: Dict[str, Any], dur_prediction: bool) -> Tensor:  # type: ignore
        x = sample["code"].clone().to(device=self.dict.weight.device)
        x = self.dict(x).transpose(1, 2)

        if self.dur_predictor and dur_prediction:
            assert x.size(0) == 1, "only support single sample"
            log_dur_pred = self.dur_predictor(x.transpose(1, 2))
            dur_out = torch.clamp(
                torch.round((torch.exp(log_dur_pred) - 1)).long(), min=1
            )
            # B x C x T
            x = torch.repeat_interleave(x, dur_out.view(-1), dim=2)

        # spkr = self.spkr(sample["spkr"].to(self.spkr.weight.device)).transpose(1, 2)
        spkr = sample["spkr"].transpose(1, 2)
        spkr = self._upsample(spkr, x.shape[-1])
        x = torch.cat([x, spkr], dim=1)  

        lang = self.lang(sample["lang"].to(self.lang.weight.device)).transpose(1, 2)
        lang = self._upsample(lang, x.shape[-1])
        x = torch.cat([lang, x], dim=1)

        return super().forward(x)