from . import _tensor
from ._utils import PriorityQueue

def _backward(tensor, xp, gy=None):
    if gy is None:
        gy = xp.ones(shape=tensor.shape, dtype=xp.float64)
    assert tensor.requires_grad is True, 'You are calling backward from a Tensor that does not requires gradient'
    tensor.grad = gy
    start_func = tensor.creator
    assert start_func is not None, 'This tensor has no creator'
    
    pq_func = PriorityQueue(comp_func=lambda f1, f2: f1.generation > f2.generation)
    pq_func.enqueue(start_func)
    seen_func = set([start_func])

    while not pq_func.empty():
        # 1. pop the function
        func = pq_func.dequeue()
        # 2. the descendant Tensors are done computing the gradients. Receive it.
        gys = [y().grad for y in func.outputs]
        # 3. calculate the gradients of ascendant Tensors
        gxs = func.backward(*gys)
        if not isinstance(gxs, tuple):
            gxs = (gxs,)
        # 4. set the gradients of ascendant Tensors
        for x, gx in zip(func.inputs, gxs):
            if x.requires_grad:
                if x.grad is None:
                    x.grad = gx
                else:
                    x.grad += gx
        # 5. push the functions
        for x in func.inputs:
            if x.requires_grad and x.creator is not None:
                # this will ensure that backward won't receive NoneType gradients
                if x.creator not in seen_func:
                    pq_func.enqueue(x.creator)
                    seen_func.add(x.creator)