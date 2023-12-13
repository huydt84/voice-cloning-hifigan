import torch
from vocoder import init_vocoder

vocoder = init_vocoder("spk_enc/vocoder/model_config.json")

input = { 
    "code": torch.rand([16, 256, 1]),
    "spkr": torch.rand([16, 256, 1]),
    "lang": torch.rand([16, 1])
}
output = vocoder.code_generator(input)

print(output.shape)