import numpy as np
import warnings

# check if cupy is available
try:
    import cupy as cp
    _gpu_is_avaiable = True
except:
    _gpu_is_avaiable = False

class Tensor:
    '''
    A tensor class.

    **warning**: Do not use 1-dimensional array.
    '''
    def __init__(self, data, requires_grad=False):
        '''
        This constructor only receives NDArray or list type.
        Never allow 1-dimensional array! Change it to 2-dimensional array, internally. --> column shaped
        This will occur some inconvinience.
        Plus, use float64 in all arrays.
        '''
        # ensure that the self.data is xp.ndarray
        self.data = Tensor._ensure_asarray(data)
        if self.data.ndim == 1:
            self.data = self.data.reshape(-1, 1)
        self.requires_grad = requires_grad
        self.creator = None
        self.generation = None

    @staticmethod
    def _ensure_asarray(data):
        if isinstance(data, (list, tuple)):
            return np.asarray(data, dtype=np.float64)
        elif isinstance(data, np.ndarray):
            if data.dtype != np.float64:
                warnings.warn(f'Data type is {data.dtype}, the data will be copied instead of being shared.', Warning)
            return np.asarray(data, dtype=np.float64)
        elif _gpu_is_avaiable and isinstance(data, cp.ndarray):
            if data.dtype != cp.float64:
                warnings.warn(f'Data type is {data.dtype}, the data will be copied instead of being shared.', Warning)
            return cp.asarray(data, dtype=cp.float64)
        else:
            raise ValueError(f'Expected list, tuple, np.ndarray or cp.ndarray, got {type(data)}.')
        
    @property
    def shape(self):
        return self.data.shape
    
    @property
    def dtype(self):
        return self.data.dtype
    
    @property
    def ndim(self):
        return self.data.ndim

# tensor generators
# 1. From existing Tensors, with same data
def tensor(tensor_data):
    '''
    Create a new copy of tensor, copying the data, but not detaching from the calculation graph.
    '''
    xp = get_array_module(tensor_data.data)
    data = xp.array(tensor_data.data)
    instance = Tensor(data=data, requires_grad=tensor_data.requires_grad)
    # I don't know if this is right, check it later.
    instance.creator = tensor_data.creator
    instance.generation = tensor_data.generation
    return instance

def as_tensor(data):
    '''
    Create a new tensor, using the tensor_data or given list as view, but detaching from the caculation graph.
    '''
    if isinstance(data, list):
        data = Tensor._ensure_asarray(data)
    # now data is NDArray type
    xp = get_array_module(data)
    data = xp.asarray(data)
    instance = Tensor(data=data, requires_grad=False)
    return instance

# 2. From existing Tensors, with same shape
def zeros_like(tensor_data):
    pass
def ones_like(tensor_data):
    pass
def full_like(tensor_data, fill_value):
    pass
def rand_like(tensor_data):
    pass
# 3. From shape
def zeros(shape):
    pass
def ones(shape):
    pass
def full(shape, fill_value):
    pass
def rand(shape):
    pass

def get_array_module(data):
    '''
    Return a module of given NDArray data
    '''
    if not _gpu_is_avaiable:
        if isinstance(data, np.ndarray):
            return np
        else:
            raise ValueError(f'Expected np.ndarray or cp.ndarray, got {type(data)}.')
    return cp.get_array_module(data)