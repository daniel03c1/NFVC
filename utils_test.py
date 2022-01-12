import torch
import unittest
from utils import *


class UtilsTest(unittest.TestCase):
    def test_bilinear_interpolation(self):
        # test 1: single value
        images = torch.ones((5, 5))
        images[3, 3] = 4
        images[3, 4] = 3
        images = images.resize(1, 1, 5, 5)

        coords = torch.tensor([[0, 3.2, 3.4]]) # [1, 3]

        outputs = bilinear_interpolation(images, coords)
        self.assertAlmostEqual(float(outputs[0, 0]), 2.68, 5)

        # test 2: various forms of coords
        n_images, size = 5, 32
        images = torch.rand(n_images, 3, size, size)

        for shape in [(7,), (7, 4), (2, 4, 3)]:
            coords = torch.cat([torch.randint(n_images, size=shape+(1,)),
                                torch.rand(*shape, 2) * (size-1)],
                               -1)
            outputs = bilinear_interpolation(images, coords)
            self.assertEqual(outputs.size(), shape + (3,))

        # test 3: check gradient flows
        coords = torch.nn.Parameter(torch.tensor([[0, 3.2, 3.4]]))
        self.assertEqual(coords.grad, None)
        bilinear_interpolation(images, coords).sum().backward()
        self.assertNotEqual(coords.grad, None)


if __name__ == '__main__':
    unittest.main()

