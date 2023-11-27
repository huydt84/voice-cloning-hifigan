# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional, Union

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
        spkr: Union[Optional[int], torch.tensor],
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
