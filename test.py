import unittest
import core
import numpy as np

class AddTest(unittest.TestCase):
    def setUp(self):
        self.x0 = core.as_tensor([1, 2, 3])
        self.x1 = core.as_tensor([1, 2, -2])

    def test_forward(self):
        f = core.Add()
        y = f(self.x0, self.x1)

        expected = core.as_tensor([2, 4, 1])
        np.testing.assert_allclose(y.data, expected.data)
    
    def test_backward_linkage(self):
        f = core.Add()
        y = f(self.x0, self.x1)
        gen = self.x0.generation + 1
        self.assertEqual(y.generation, gen)
        self.assertEqual(f.generation, gen)

    def test_requires_grad_propagation(self):
        f = core.Add()
        self.x0.requires_grad = True
        y = f(self.x0, self.x1)
        self.assertTrue(y.requires_grad)

    def test_broadcast(self):
        x0 = core.as_tensor([[1, 2]])
        x1 = core.as_tensor([[1, 2], [3, 9]])
        x2 = core.as_tensor([1, 2])
        f = core.Add()

        y = 1 + x0
        self.assertEqual(y.creator.inputs[1].shape, (1, 2))

        y1 = f(x0, x1)
        y2 = f(x1, x2)

        expected1 = core.as_tensor([[2, 4], [4, 11]])
        expected2 = core.as_tensor([[2, 3], [5, 11]])

        np.testing.assert_allclose(y1.data, expected1.data)
        np.testing.assert_allclose(y2.data, expected2.data)

    def test_broadcast_backward(self):
        x0 = core.as_tensor([[1, 2]], True)
        x1 = core.as_tensor([[-1, 9], [3, 4], [4, -1]], False)

        f = core.Add()
        y = f(x0, x1)
        gx1, gx2 = f.backward(np.asarray([[1., 1.], [-3., 3.], [4., 5.]]))

        expected1 = np.asarray([[2., 9.]])

        np.testing.assert_allclose(gx1, expected1)

class MatMulTest(unittest.TestCase):
    def test_forward(self):
        l0 = [1, 2, 3]
        l1 = [3, 4, 1]
        l2 = [[3, 4], [4, 5], [-1, -2]]
        x0 = core.as_tensor(l0)
        x1 = core.as_tensor(l1)
        x2 = core.as_tensor(l2)

        f = core.MatMul()
        y0 = f(x0, x1.T)
        expected0 = core.as_tensor([[3, 4, 1], [6, 8, 2], [9, 12, 3]])

        y1 = f(x2.T, x1)
        expected1 = core.as_tensor([24, 30])

        np.testing.assert_allclose(y0.data, expected0.data)
        np.testing.assert_allclose(y1.data, expected1.data)

    def test_backward(self):
        gy = np.asarray([[3.]])
        x0 = core.as_tensor([1, 3, 4], requires_grad=True)
        
        f = core.MatMul()
        y = f(x0.T, x0)
        gx1, gx2 = f.backward(gy)

        self.assertEqual(x0.shape, gx2.shape)
        self.assertEqual(x0.T.shape, gx1.shape)

class CompositeTest(unittest.TestCase):
    def test_MatMul_and_Add_with_scalar(self):
        x0 = core.as_tensor([[1, 2, 3], [1, 2, -1]], requires_grad=False)
        x1 = core.as_tensor([3, 4, -1], requires_grad=True)
        x2 = core.as_tensor(2, True)
        y = x0 @ (2 * x1) + x2
        y.backward()

class GradCorrectTest(unittest.TestCase):
    def setUp(self):
        self.eps = 1e-4
        self.tol = 1e-6
    
    def test_composite(self):
        # val = np.asarray([[1, 2, 3], [1, 2, 3], [1, 2, 3]], dtype=np.float64)
        val = 3.
        x = core.as_tensor(val, requires_grad=True)
        f = f2
        y = f(x)
        y.backward()
        dy_dx = numerical_grad(f, val)
        
        print(x.grad)
        print(dy_dx)
        np.testing.assert_allclose(x.grad, dy_dx)
        
        
def f1(x):
    W = np.asarray([[1, 3, -1], [2, 3, -1]], dtype=np.float64)
    if isinstance(x, np.ndarray):
        result = W @ x + 2
    else:
        W = core.as_tensor(W)
        result = W @ x + 2
    return result

def f2(x):
    return 1 / x

def numerical_grad(f, x_, eps=1e-4):
    x = np.array(x_)
    x = np.atleast_2d(x)
    grad = np.zeros_like(x)

    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]

        # f(x + eps)
        x[idx] = tmp_val + eps
        y1 = f(x)
        # f(x - eps)
        x[idx] = tmp_val - eps
        y2 = f(x)

        grad[idx] = (y1.sum() - y2.sum()) / (2 * eps)

        x[idx] = tmp_val
        it.iternext()
    
    return np.atleast_2d(grad)