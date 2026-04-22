from ._tensor import get_array_module
from ._utils import sum_to
import weakref

class Function:
    '''
    Base Function class
    '''
    def __init__(self):
        self.generation = None

    def __call__(self, *args):
        # 1. Get the NDArray of input arguments, and maximum generation
        data = [arg.data for arg in args]
        max_gen = max([arg.generation for arg in args])
        self.generation = max_gen + 1
        requires_grad_list = [arg.requires_grad for arg in args]
        requires_grad = False
        for req in requires_grad_list: requires_grad = requires_grad or req
        # 2. Calculate outputs
        output_data = self.forward(*data)
        if not isinstance(output_data, tuple):
            output_data = (output_data,)
        # 3. Translate to Tensor
        from ._tensor import Tensor
        outputs = [Tensor(output) for output in output_data]
        # 4. Set attributes of output Tensor
        for output in outputs:
            output.creator = self
            output.generation = max_gen + 1
            output.requires_grad = requires_grad
        # 5. Save inputs and outputs, where output takes weak reference
        self.inputs = args
        self.outputs = [weakref.ref(output) for output in outputs]
        # 6. Return strong reference
        return outputs[0] if len(outputs) == 1 else outputs
        

    # Assume forward takes n parameters as input and has m number of outputs.
    # Then, the backward method takes m parameters as input and has n number of outputs.
    # Important! Manipulate only NDArrays. Do not manipulate Tensor objects directly.
    def forward(self, *xs):
        raise NotImplementedError
    
    def backward(self, *gys):
        raise NotImplementedError
    
class Add(Function):
    def forward(self, x1, x2):
        # warning: Take NDArray type
        return x1 + x2
    
    def backward(self, gy):
        assert gy is not None, 'gradient is None.'
        assert gy.ndim > 1, f'gy.ndim is smaller than 2, gy.ndim: {gy.ndim}'
        xp = get_array_module(gy)
        x1, x2 = self.inputs[0].data, self.inputs[1].data
        assert isinstance(x1, xp.ndarray) and isinstance(x2, xp.ndarray), f'x1: {type(x1)}, x2: {type(x2)}'
        gx1, gx2 = xp.array(gy), xp.array(gy)
        assert gx1.dtype == xp.float64 and gx2.dtype == xp.float64, f'gx1.dtype: {gx1.dtype}, gx2.dtype: {gx2.dtype}'
        assert gx1.ndim != 1 or gx2.ndim != 1, f'gx1.shape: {gx1.shape}, gx2.shape: {gx2.shape}'

        gx1 = sum_to(gx1, x1.shape)
        gx2 = sum_to(gx2, x2.shape)
        assert gx1.shape == x1.shape and gx2.shape == x2.shape, f'gx1.shape and x1.shape: {gx1.shape} and {x1.shape}, gx2.shape and x2.shape: {gx2.shape} and {x2.shape}'
        return gx1, gx2
    
class MatMul(Function):
    def forward(self, x1, x2):
        assert x1.ndim > 1 and x2.ndim > 1, f'x1.ndim: {x1.ndim}, x2.ndim: {x2.ndim}, should be bigger than 1.'
        xp = get_array_module(x1)
        return xp.matmul(x1, x2)
    
    def backward(self, gy):
        assert gy is not None, 'gradient is None.'
        assert gy.ndim > 1, f'1 dimensional tensor or pure scalar not allowed, got gy: {gy.ndim}'
        x1 = self.inputs[0].data
        x2 = self.inputs[1].data
        xp = get_array_module(x1)
        
        x1_T = xp.swapaxes(x1, -1, -2)
        x2_T = xp.swapaxes(x2, -1, -2)
        
        gx1 = xp.matmul(gy, x2_T)
        gx2 = xp.matmul(x1_T, gy)

        gx1 = sum_to(gx1, x1.shape)
        gx2 = sum_to(gx2, x2.shape)

        return gx1, gx2

class Transpose(Function):
    def forward(self, x):
        xp = get_array_module(x)
        return xp.swapaxes(x, -1, -2)
    
    def backward(self, gy):
        assert gy is not None, 'gradient is None'
        xp = get_array_module(gy)
        return xp.swapaxes(gy, -1, -2)
    
class Neg(Function):
    def forward(self, x):
        return -x
    
    def backward(self, gy):
        assert gy is not None, 'gradient is None'
        return -gy
    
class Mul(Function):
    def forward(self, x1, x2):
        assert x1.ndim > 1 and x2.ndim > 1, f'x1.ndim: {x1.ndim}, x2.ndim: {x2.ndim}'
        return x1 * x2
    
    def backward(self, gy):
        assert gy is not None, f'gradient is None'
        x1 = self.inputs[0].data
        x2 = self.inputs[1].data
        gx1 = gy * x2
        gx2 = gy * x1
        gx1 = sum_to(gx1, x1.shape)
        gx2 = sum_to(gx2, x2.shape)
        return gx1, gx2
    
class Pow(Function):
    def forward(self, x1, x2):
        return x1 ** x2
    
    def backward(self, gy):
        xp = get_array_module(gy)
        x1 = self.inputs[0].data
        x2 = self.inputs[1].data
        gx1 = x2 * gy * x1 ** (x2 - 1)
        gx2 = gy * (x1 ** x2) * xp.log(x2)

        gx1 = sum_to(gx1, x1.shape)
        gx2 = sum_to(gx2, x2.shape)

        return gx1, gx2

class Div(Function):
    def forward(self, x1, x2):
        return x1 / x2
    
    def backward(self, gy):
        x1 = self.inputs[0].data
        x2 = self.inputs[1].data
        inv = 1 / x2
        gx1 = gy * inv
        gx2 = - gy * x1 * (inv ** 2)

        gx1 = sum_to(gx1, x1.shape)
        gx2 = sum_to(gx2, x2.shape)

        return gx1, gx2
