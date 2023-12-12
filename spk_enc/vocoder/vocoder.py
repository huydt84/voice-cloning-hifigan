# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional, Union
import json

import torch
import torch.nn as nn
from torch import Tensor

from vocoder.codehifigan import CodeGenerator, CustomCodeGenerator


class Vocoder(nn.Module):
    def __init__(self, code_generator: Union[CustomCodeGenerator, CodeGenerator], lang_spkr_idx_map: dict):
        super(Vocoder, self).__init__()
        self.code_generator = code_generator
        self.lang_spkr_idx_map = lang_spkr_idx_map

    def forward(
        self,
        code: List[int],
        lang: str,
        spkr,
        dur_prediction: bool = True,
    ) -> Tensor:
        x = {
            "code": torch.LongTensor(code).view(1, -1),
        }
        lang_idx = self.lang_spkr_idx_map["multilingual"][lang]
        
        if isinstance(self.code_generator, CodeGenerator):
            spkr_list = self.lang_spkr_idx_map["multispkr"][lang]
            if not spkr:
                spkr = -1
            spkr = spkr_list[0] if spkr == -1 else spkr
            x["spkr"] = torch.tensor([[spkr]])
        else:
            x["spkr"] = spkr
        x["lang"] = torch.tensor([[lang_idx]])
        return self.code_generator(x, dur_prediction)

LANGUAGE_CODE = {
    "arb": 0,
    "ben": 1,
    "cat": 2,
    "ces": 3,
    "cmn": 4,
    "cym": 5,
    "dan": 6,
    "deu": 7,
    "eng": 8,
    "est": 9,
    "fin": 10,
    "fra": 11,
    "hin": 12,
    "ind": 13,
    "ita": 14,
    "jpn": 15,
    "kor": 16,
    "mlt": 17,
    "nld": 18,
    "pes": 19,
    "pol": 20,
    "por": 21,
    "ron": 22,
    "rus": 23,
    "slk": 24,
    "spa": 25,
    "swe": 26,
    "swh": 27,
    "tel": 28,
    "tgl": 29,
    "tha": 30,
    "tur": 31,
    "ukr": 32,
    "urd": 33,
    "uzn": 34,
    "vie": 35
}

def init_vocoder(model_config_path, lang_spkr_idx_map=LANGUAGE_CODE):
    with open(model_config_path) as f:
        data = f.read()

    json_config = json.loads(data)
    code_generator = CustomCodeGenerator(**json_config)
    
    return Vocoder(code_generator, lang_spkr_idx_map)