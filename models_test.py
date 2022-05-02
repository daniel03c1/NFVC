import torch
import torch.nn as nn
import unittest
from models import *


class ModelsTest(unittest.TestCase):
    def test_nerv_block(self):
        in_resolution = [32, 24]
        in_chan = 4
        out_chan = 3
        inputs = torch.ones((128, in_chan, *in_resolution))

        for upscale_factor in [1, 2]:
            outputs = NeRVBlock(in_chan, out_chan, 5, upscale_factor)(inputs)
            self.assertEqual(
                outputs.size(),
                (128, out_chan, *[d*upscale_factor for d in in_resolution]))

    def test_basic_block(self):
        in_resolution = [32, 24]
        in_chan = 4
        out_chan = 4 # 16
        inputs = torch.rand(128, in_chan, *in_resolution)

        block = BasicBlock(in_chan, out_chan, 5, activation='relu')
        block.param_init()
        outputs = block(inputs)
        self.assertEqual(outputs.size(), (128, out_chan, *inputs.shape[-2:]))

        import pdb; pdb.set_trace()
        print()

    def test_mlp(self):
        dim_list = [2, 7, 4]
        inputs = torch.ones((128, dim_list[0]))

        outputs = MLP(dim_list)(inputs)
        self.assertEqual(outputs.size(), (128, dim_list[-1]))

    def test_my_nerv(self):
        n_frames = 128
        resolution = [128, 64]

        channels = 16
        inputs = torch.ones((1, *resolution, 3))

        outputs = MyNeRV([n_frames, 3, *resolution], channels)(inputs)
        self.assertEqual(outputs.size(), tuple([1, 3, *resolution]))

    def test_nerv(self):
        input_dim = [1]
        output_resolution = [3, 128, 64]
        channels = 7
        inputs = torch.ones((128, *input_dim))

        outputs = NeRV(input_dim, output_resolution, channels, pos_enc_freqs=10,
                       upscale_factors=[4, 2], n_blocks=2)(inputs)
        self.assertEqual(outputs.size(), tuple([128, *output_resolution]))


if __name__ == '__main__':
    unittest.main()

