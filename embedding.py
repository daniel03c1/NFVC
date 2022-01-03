import torch
import torch.nn as nn


class Embedding(nn.Module):
    def forward(self, inputs):
        raise NotImplemented()

    def get_output_size(self):
        raise NotImplemented()


class RFF(Embedding):
    def __init__(self, in_features, emb_size, emb_sigma):
        super().__init__()
        self.B = nn.Parameter(
            torch.normal(0, emb_sigma, (in_features, emb_size)),
            requires_grad=False)
        self.output_size = emb_size * 2

    def forward(self, inputs):
        outs = inputs @ self.B.to(inputs.device)
        return torch.cat([torch.sin(outs), torch.cos(outs)], -1)

    def get_output_size(self):
        return self.output_size 


class PosEncoding(Embedding):
    def __init__(self, in_features, n_freqs, include_inputs=True):
        super().__init__()
        self.in_features = in_features
        self.n_freqs = n_freqs
        self.freqs = [2**i for i in range(n_freqs)]
        self.include_inputs = include_inputs
        self.output_size = in_features * (2*n_freqs + include_inputs)

    def forward(self, inputs):
        outs = []
        if self.include_inputs:
            outs.append(inputs)
        for freq in self.freqs:
            outs.append(torch.sin(inputs * freq))
            outs.append(torch.cos(inputs * freq))
        return torch.cat(outs, -1)

    def get_output_size(self):
        return self.output_size

