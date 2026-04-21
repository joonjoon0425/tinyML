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
    
    # implement backward (the most important)
    def backward(self, grad):
        pass

    # implement the operators
    # this time, think carefully
    # 1. broadcast
    # 2. requires_grad
    def __add__(self, other):
        pass
    def __radd__(self, other):
        pass
    def __matmul__(self, other):
        pass
    def __rmatmul__(self, other):
        pass
    def __mul__(self, other):
        pass
    def __rmul__(self, other):
        pass
    def __pow__(self, exponent):
        pass
    def __neg__(self):
        pass
    def __sub__(self):
        pass
    @property
    def T(self):
        pass



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
    
    def __repr__(self):
        return f'{self.data}'

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
    Create a new tensor, using the tensor_data, NDArray or given list, but detaching from the caculation graph.
    copies if a list is given, but uses view if Tensor or NDArray is given.
    '''
    if isinstance(data, (int, float)):
        data = np.array(data, dtype=np.float64)
    elif isinstance(data, list):
        data = Tensor._ensure_asarray(data)
    elif isinstance(data, Tensor):
        data = data.data
    # now data is NDArray type
    xp = get_array_module(data)
    data = xp.asarray(data)
    instance = Tensor(data=data, requires_grad=False)
    return instance

# 2. From existing Tensors, with same shape
def zeros_like(tensor_data):
    xp = get_array_module(tensor_data.data)
    data = xp.zeros_like(tensor_data.data, dtype=xp.float64)
    return Tensor(data)
def ones_like(tensor_data):
    xp = get_array_module(tensor_data.data)
    data = xp.ones_like(tensor_data.data, dtype=xp.float64)
    return Tensor(data)
def full_like(tensor_data, fill_value):
    xp = get_array_module(tensor_data.data)
    data = xp.full_like(tensor_data.data, fill_value=fill_value, dtype=xp.float64)
    return Tensor(data)
def empty_like(tensor_data):
    xp = get_array_module(tensor_data.data)
    data = xp.empty_like(tensor_data.data, dtype=xp.float64)
    return Tensor(data)
# 3. From shape
def zeros(shape):
    data = np.zeros(shape=shape, dtype=np.float64)
    return Tensor(data)
def ones(shape):
    data = np.ones(shape=shape, dtype=np.float64)
    return Tensor(data)
def full(shape, fill_value):
    data = np.full(shape=shape, fill_value=fill_value, dtype=np.float64)
    return Tensor(data)
def empty(shape):
    return np.empty(shape=shape, dtype=np.float64)

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