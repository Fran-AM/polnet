import unittest
# from polnet.membrane import MbEllipsoid

from src.sample_generation.membranes.mb import Mb
from src.sample_generation.membranes.mb_ellipsoid import MbEllipsoid
from src.sample_generation.membranes.mb_sphere import MbSphere
from src.sample_generation.membranes.mb_torus import MbTorus

#from ..src.sample_generation.membranes.mb_sphere import MbSphere
#from ..src.sample_generation.membranes.mb_torus import MbTorus
#from ..src.sample_generation.membranes.set_membranes import SetMembranes

from src.utils.lio import write_mrc

import numpy as np
import os

class TestMembranes(unittest.TestCase):

    def setUp(self):
        self.tomo_shape = (200, 200, 200)
        self.v_size = 1.0
        self.center = (100, 100, 100)
        self.rot_q = (1, 0, 0, 0)
        self.thick = 1
        self.layer_s = 1
        self.output_path = os.path.join(os.path.dirname(__file__), 'outputs') + '/'
        os.makedirs(self.output_path, exist_ok=True)

    def test_mb_ellipsoid(self):
        ellipsoid = MbEllipsoid(self.tomo_shape, self.v_size, self.center, self.rot_q, self.thick, self.layer_s, a=40, b=30, c=20)
        self.assertEqual(ellipsoid.get_thick(), self.thick)
        self.assertEqual(ellipsoid.get_layer_s(), self.layer_s)
        write_mrc(ellipsoid.get_mask().astype(np.int8), self.output_path + 'test_mb_ellipsoid_mask.mrc', v_size=self.v_size, dtype=np.int8)

    def test_mb_sphere(self):
        sphere = MbSphere(self.tomo_shape, self.v_size, self.center, self.rot_q, self.thick, self.layer_s, rad=10)
        self.assertEqual(sphere.get_thick(), self.thick)
        self.assertEqual(sphere.get_layer_s(), self.layer_s)
        write_mrc(sphere.get_mask().astype(np.int8), self.output_path + 'test_mb_sphere_mask.mrc', v_size=self.v_size, dtype=np.int8)

    def test_mb_torus(self):
        torus = MbTorus(self.tomo_shape, self.v_size, self.center, self.rot_q, self.thick, self.layer_s, rad_a=10, rad_b=3)
        self.assertEqual(torus.get_thick(), self.thick)
        self.assertEqual(torus.get_layer_s(), self.layer_s)
        write_mrc(torus.get_mask().astype(np.int8), self.output_path + 'test_mb_torus_mask.mrc', v_size=self.v_size, dtype=np.int8)

    # def test_set_membranes(self):
    #     voi = np.ones(self.tomo_shape, dtype=bool)
    #     set_membranes = SetMembranes(voi, self.v_size, None, (1, 10), (1, 5), (1, 5), occ=50)
    #     self.assertEqual(set_membranes.get_vol(), self.v_size ** 3 * np.sum(voi))

if __name__ == '__main__':
    unittest.main()