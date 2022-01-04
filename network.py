import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from embedding import Embedding


def activation_mapper(activation):
    if activation in [None, '']:
        return nn.Identity
    elif isinstance(activation, str):
        return getattr(torch.nn, activation)
    elif callable(activation):
        return activation
    else:
        raise ValueError()


# TODO: update "use_qat" option
class NeuralFieldsNetwork(nn.Module):
    def __init__(self,
                 in_features: int, out_features: int,
                 hidden_features: int, n_hidden_layers: int,
                 input_embedding=None,
                 activation='ReLU',
                 output_activation=None,
                 use_qat=False):
        super(NeuralFieldsNetwork, self).__init__()
 
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_features = hidden_features
        self.n_hidden_layers = n_hidden_layers

        activation = activation_mapper(activation)
        output_activation = activation_mapper(output_activation)

        self.nets = []

        if input_embedding is not None:
            assert isinstance(input_embedding, Embedding)
            self.nets.extend([
                input_embedding,
                nn.Linear(input_embedding.get_output_size(),
                          hidden_features),
                activation()])
        else: # input_embedding is None
            self.nets.extend([
                nn.Linear(in_features, hidden_features),
                activation()])

        for i in range(self.n_hidden_layers):
            self.nets.extend([
                nn.Linear(hidden_features, hidden_features),
                activation()])

        self.nets.extend([
            nn.Linear(hidden_features, out_features),
            output_activation()])

        self.nets = nn.Sequential(*self.nets)

    def forward(self, inputs):
        return self.nets(inputs)

