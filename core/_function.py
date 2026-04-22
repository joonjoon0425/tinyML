from ._tensor import Tensor
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
    def forward(self, *xs):
        '''
        Take NDArray input
        '''
        raise NotImplementedError
    
    def backward(self, *gys):
        raise NotImplementedError
    
class Add(Function):
    def forward(self, x1, x2):
        # warning: Take NDArray type
        return x1 + x2