import numpy as np
import torch
import torch.nn.functional as F
import math
from pyrep import PyRep
from pyrep.robots.arms.panda import Panda
from pyrep.robots.end_effectors.panda_gripper import PandaGripper
from pyrep.objects.dummy import Dummy
from pyrep.objects.shape import Shape
from pyrep.objects.joint import Joint
from pyrep.const import PrimitiveShape

# x = torch.Tensor([[1, 0],[2,0]])

# print(torch.eq(x, 1))
# print(np.isinf(math.inf))
# a = np.array([1,2,3])
# a = a.reshape(1, 3)
# print(a)
# a = np.array(
#     [[1,2,3],
#     [4,5,6],
#     [7,8,9]]
# )

# print(a[True, True, False])