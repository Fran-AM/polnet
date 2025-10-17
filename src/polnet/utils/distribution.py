import random
import math

import numpy as np

PI_2 = math.pi * 2


def gen_rand_unit_quaternion():
    """
    Generates a random sample for a unit quaternion
    KUFFNER, James J. Effective sampling and distance metrics for 3D rigid body path planning.
    In IEEE International Conference on Robotics and Automation, 2004. Proceedings. ICRA'04. 2004. IEEE, 2004. p. 3993-3998.

    :return: a unit quaternion in the shape Q=(w,x,y,z)=(cos(θ/2),vx*sin(θ/2),vy*sin(θ/2),vz*sin(θ/2)),
             where (vx,vy,vz) is the axis and \theta is the angle
    """
    s = random.random()
    sigma_1, sigma_2 = math.sqrt(1 - s), math.sqrt(s)
    theta_1, theta_2 = PI_2 * random.random(), PI_2 * random.random()
    w = math.cos(theta_2) * sigma_2
    x = math.sin(theta_1) * sigma_1
    y = math.cos(theta_1) * sigma_1
    z = math.sin(theta_2) * sigma_2
    return np.asarray((w, x, y, z))
