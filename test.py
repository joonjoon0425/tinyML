import cupy as cp
print(cp.is_available())
print(cp.asarray([1, 2, 3]).device)

import core
tensor = core.as_tensor(cp.asarray([1, 2, 3]))
print(core._tensor.get_array_module(tensor.data))
print(core._tensor._gpu_is_avaiable)
print(tensor.shape)