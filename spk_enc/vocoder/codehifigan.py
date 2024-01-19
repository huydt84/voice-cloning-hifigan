# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from typing import Any, Dict, List, Optional
import numpy as np

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Dropout

from vocoder.hifigan import Generator
from encoder.params_model import model_embedding_size as spkr_embedding_dim
from encoder.model_ecapa_tdnn import SpeakerEncoder

from encoder.audio import preprocess_wav
from matplotlib import cm
from encoder import audio
from encoder.params_data import *


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

    def forward(self, sample: Dict[str, Any], dur_prediction: bool = False) -> Tensor:  # type: ignore
        
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
        self.spkr = nn.Linear(spkr_embedding_dim, spkr_embedding_dim)
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

    def forward(self, sample: Dict[str, Any], dur_prediction: bool = False) -> Tensor:  # type: ignore
        units = sample["code"]
        x = self.dict(units).transpose(1, 2)

        if self.dur_predictor and dur_prediction:
            log_dur_pred = self.dur_predictor(x.transpose(1, 2))
            dur_out = torch.clamp(
                torch.round((torch.exp(log_dur_pred) - 1)).long(), min=1
            )
            # B x C x T
            repeat_interleaved_x = []
            for i in range(x.size(0)):
                repeat_interleaved_x.append(torch.repeat_interleave(x[i].unsqueeze(0), dur_out[i].view(-1), dim=2))
            x = torch.cat(repeat_interleaved_x)

        upsampled_spkr = []
        upsampled_lang = []
        
        spkr = self.spkr(sample["spkr"]).unsqueeze(-1)
        lang = self.lang(sample["lang"]).transpose(1, 2)
        for i in range(x.size(0)):
            upsampled_spkr.append(self._upsample(spkr[i], x.shape[-1]))
            upsampled_lang.append(self._upsample(lang[i], x.shape[-1]))
        spkr = torch.cat(upsampled_spkr, dim=1).transpose(0, 1)
        lang = torch.cat(upsampled_lang, dim=1).transpose(0, 1)
        x = torch.cat([x, spkr], dim=1)  
        x = torch.cat([lang, x], dim=1)

        return super().forward(x)
    
    
class CustomExpressiveCodeGenerator(Generator):
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
        first_init: bool = True,
        speaker_encoder_path: str = "/content/drive/MyDrive/voice-cloning-hifigan/saved_models/2/encoder.pt"
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

        self.device = torch.device("cuda")
        self.speaker_encoder = SpeakerEncoder(self.device, self.device)

        if first_init and speaker_encoder_path:
            print("\nLoad pretrained speaker encoder...\n")
            checkpoint = torch.load(speaker_encoder_path, self.device)
            self.speaker_encoder.load_state_dict(checkpoint["model_state"])
            
    def embed_frames_batch(self, frames_batch):
        """
        Computes embeddings for a batch of mel spectrogram.

        :param frames_batch: a batch mel of spectrogram as a numpy array of float32 of shape
        (batch_size, n_frames, n_channels)
        :return: the embeddings as a numpy array of float32 of shape (batch_size, model_embedding_size)
        """

        frames = torch.from_numpy(frames_batch).to(self.device)
        embed = self.speaker_encoder.forward(frames).detach().cpu().numpy()
        return embed


    def compute_partial_slices(self, n_samples, partial_utterance_n_frames=partials_n_frames,
                            min_pad_coverage=0.75, overlap=0.5):
        """
        Computes where to split an utterance waveform and its corresponding mel spectrogram to obtain
        partial utterances of <partial_utterance_n_frames> each. Both the waveform and the mel
        spectrogram slices are returned, so as to make each partial utterance waveform correspond to
        its spectrogram. This function assumes that the mel spectrogram parameters used are those
        defined in params_data.py.

        The returned ranges may be indexing further than the length of the waveform. It is
        recommended that you pad the waveform with zeros up to wave_slices[-1].stop.

        :param n_samples: the number of samples in the waveform
        :param partial_utterance_n_frames: the number of mel spectrogram frames in each partial
        utterance
        :param min_pad_coverage: when reaching the last partial utterance, it may or may not have
        enough frames. If at least <min_pad_coverage> of <partial_utterance_n_frames> are present,
        then the last partial utterance will be considered, as if we padded the audio. Otherwise,
        it will be discarded, as if we trimmed the audio. If there aren't enough frames for 1 partial
        utterance, this parameter is ignored so that the function always returns at least 1 slice.
        :param overlap: by how much the partial utterance should overlap. If set to 0, the partial
        utterances are entirely disjoint.
        :return: the waveform slices and mel spectrogram slices as lists of array slices. Index
        respectively the waveform and the mel spectrogram with these slices to obtain the partial
        utterances.
        """
        assert 0 <= overlap < 1
        assert 0 < min_pad_coverage <= 1

        samples_per_frame = int((sampling_rate * mel_window_step / 1000))
        n_frames = int(np.ceil((n_samples + 1) / samples_per_frame))
        frame_step = max(int(np.round(partial_utterance_n_frames * (1 - overlap))), 1)

        # Compute the slices
        wav_slices, mel_slices = [], []
        steps = max(1, n_frames - partial_utterance_n_frames + frame_step + 1)
        for i in range(0, steps, frame_step):
            mel_range = np.array([i, i + partial_utterance_n_frames])
            wav_range = mel_range * samples_per_frame
            mel_slices.append(slice(*mel_range))
            wav_slices.append(slice(*wav_range))

        # Evaluate whether extra padding is warranted or not
        last_wav_range = wav_slices[-1]
        coverage = (n_samples - last_wav_range.start) / (last_wav_range.stop - last_wav_range.start)
        if coverage < min_pad_coverage and len(mel_slices) > 1:
            mel_slices = mel_slices[:-1]
            wav_slices = wav_slices[:-1]

        return wav_slices, mel_slices


    def embed_utterance(self, wav, using_partials=True, return_partials=False, **kwargs):
        """
        Computes an embedding for a single utterance.

        # TODO: handle multiple wavs to benefit from batching on GPU
        :param wav: a preprocessed (see audio.py) utterance waveform as a numpy array of float32
        :param using_partials: if True, then the utterance is split in partial utterances of
        <partial_utterance_n_frames> frames and the utterance embedding is computed from their
        normalized average. If False, the utterance is instead computed from feeding the entire
        spectogram to the network.
        :param return_partials: if True, the partial embeddings will also be returned along with the
        wav slices that correspond to the partial embeddings.
        :param kwargs: additional arguments to compute_partial_splits()
        :return: the embedding as a numpy array of float32 of shape (model_embedding_size,). If
        <return_partials> is True, the partial utterances as a numpy array of float32 of shape
        (n_partials, model_embedding_size) and the wav partials as a list of slices will also be
        returned. If <using_partials> is simultaneously set to False, both these values will be None
        instead.
        """
        # Process the entire utterance if not using partials
        if not using_partials:
            frames = audio.wav_to_mel_spectrogram(wav)
            embed = self.embed_frames_batch(frames[None, ...])[0]
            if return_partials:
                return embed, None, None
            return embed

        # Compute where to split the utterance into partials and pad if necessary
        wave_slices, mel_slices = self.compute_partial_slices(len(wav), **kwargs)
        max_wave_length = wave_slices[-1].stop
        if max_wave_length >= len(wav):
            wav = np.pad(wav, (0, max_wave_length - len(wav)), "constant")

        # Split the utterance into partials
        frames = audio.wav_to_mel_spectrogram(wav)
        frames_batch = np.array([frames[s] for s in mel_slices])
        partial_embeds = self.embed_frames_batch(frames_batch)

        # Compute the utterance embedding from the partial embeddings
        raw_embed = np.mean(partial_embeds, axis=0)
        embed = raw_embed / np.linalg.norm(raw_embed, 2)

        if return_partials:
            return embed, partial_embeds, wave_slices
        return embed


    def embed_speaker(self, wavs, **kwargs):
        raise NotImplemented()


    def plot_embedding_as_heatmap(self, embed, ax=None, title="", shape=None, color_range=(0, 0.30)):
        import matplotlib.pyplot as plt
        if ax is None:
            ax = plt.gca()

        if shape is None:
            height = int(np.sqrt(len(embed)))
            shape = (height, -1)
        embed = embed.reshape(shape)

        cmap = cm.get_cmap()
        mappable = ax.imshow(embed, cmap=cmap)
        cbar = plt.colorbar(mappable, ax=ax, fraction=0.046, pad=0.04)
        sm = cm.ScalarMappable(cmap=cmap)
        sm.set_clim(*color_range)

        ax.set_xticks([]), ax.set_yticks([])
        ax.set_title(title)


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

    def forward(self, sample: Dict[str, Any], dur_prediction: bool = False) -> Tensor:  # type: ignore
        units = sample["code"]
        x = self.dict(units).transpose(1, 2)

        if self.dur_predictor and dur_prediction:
            log_dur_pred = self.dur_predictor(x.transpose(1, 2))
            dur_out = torch.clamp(
                torch.round((torch.exp(log_dur_pred) - 1)).long(), min=1
            )
            # B x C x T
            repeat_interleaved_x = []
            for i in range(x.size(0)):
                repeat_interleaved_x.append(torch.repeat_interleave(x[i].unsqueeze(0), dur_out[i].view(-1), dim=2))
            x = torch.cat(repeat_interleaved_x)

        upsampled_spkr = []
        upsampled_lang = []
        
        lang = self.lang(sample["lang"]).transpose(1, 2)
        embeded = self.embed_utterance(sample["spkr"])
        
        for i in range(x.size(0)):
            upsampled_spkr.append(self._upsample(embeded[i], x.shape[-1]))
            upsampled_lang.append(self._upsample(lang[i], x.shape[-1]))
        spkr = torch.cat(upsampled_spkr, dim=1).transpose(0, 1)
        lang = torch.cat(upsampled_lang, dim=1).transpose(0, 1)
        x = torch.cat([x, spkr], dim=1)  
        x = torch.cat([lang, x], dim=1)

        return super().forward(x)