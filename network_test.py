import torch
import torch.nn as nn
import unittest
from network import *


class NetworkTest(unittest.TestCase):
    def test_neural_fields_network(self):
        in_features, out_features = 3, 5
        hidden_features, n_hidden_layers = 64, 3

        network = NeuralFieldsNetwork(in_features, out_features,
                                      hidden_features, n_hidden_layers)
        self.assertEqual(network(torch.ones([32, in_features])).size(),
                         (32, out_features))

    def test_swish(self):
        in_features, out_features = 3, 5
        network = nn.Sequential(
            nn.Linear(in_features, out_features),
            Swish())

        inputs = torch.rand(32, in_features)
        self.assertEqual(network(inputs).size(), (32, out_features))


if __name__ == '__main__':
    unittest.main()

