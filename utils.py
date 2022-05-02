import torch
import torch.nn as nn


def get_activation_fn(act_type):
    act_type = act_type.lower()
    if act_type == 'relu':
        act_layer = nn.ReLU(True)
    elif act_type == 'leaky':
        act_layer = nn.LeakyReLU(inplace=True)
    elif act_type == 'leaky01':
        act_layer = nn.LeakyReLU(negative_slope=0.1, inplace=True)
    elif act_type == 'relu6':
        act_layer = nn.ReLU6(inplace=True)
    elif act_type == 'gelu':
        act_layer = nn.GELU()
    elif act_type == 'sin':
        act_layer = torch.sin
    elif act_type == 'swish':
        act_layer = nn.SiLU(inplace=True)
    elif act_type == 'softplus':
        act_layer = nn.Softplus()
    elif act_type == 'hardswish':
        act_layer = nn.Hardswish(inplace=True)
    else:
        raise KeyError(f"Unknown activation function {act_type}.")

    return act_layer


def make_grid(*input_size, minvalue=-1, maxvalue=1):
    # generate grid for a given input size
    # 2D coordinates: use make_input_grid(H, W) -> (H, W, 2) shaped tensor
    # 3D coordinates: use make_input_grid(T, H, W) -> (T, H, W, 3) shaped tensor
    # return torch.stack(
    #     torch.meshgrid(*[(torch.arange(int(S))+0.5)/int(S)*(maxvalue-minvalue)
    #                      + minvalue
    #                      for S in input_size]), -1)
    return torch.stack(
        torch.meshgrid(*[torch.linspace(minvalue, maxvalue, int(S))
                         for S in input_size]), -1)


class PosEncoding(nn.Module):
    def __init__(self, in_features, n_freqs, include_inputs=False,
                 trainable=False):
        super().__init__()
        self.in_features = in_features

        if isinstance(n_freqs, (int, float)):
            n_freqs = [n_freqs for _ in range(in_features)]
        self.n_freqs = torch.tensor(n_freqs).int()

        eye = torch.eye(in_features)
        self.freq_mat = nn.Parameter(
            torch.cat([torch.stack([eye[i] * (2**j)
                                    for j in range(self.n_freqs[i])], -1)
                       for i in range(in_features)], -1),
            requires_grad=trainable)

        self.include_inputs = include_inputs
        self.output_size = in_features * include_inputs + 2 * sum(self.n_freqs)

    def forward(self, inputs):
        outs = []
        if self.include_inputs:
            outs.append(inputs)
        mapped = inputs @ self.freq_mat.to(inputs.device)
        outs.append(torch.cos(mapped))
        outs.append(torch.sin(mapped))

        return torch.cat(outs, -1)

    def get_output_size(self):
        return self.output_size

