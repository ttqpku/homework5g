import numpy as np
from mosek.fusion import *

class Out:
    def __init__(self):
        self.fval = 0.
        self.Runtime = 0.
        self.itr = 0

np.random.seed(97006855)
m = 256
n = 512
A = np.random.randn(m, n)
k = round(n * 0.1)
A = np.random.randn(m, n)
l = 2
p = np.random.permutation(n)
p = p[0:k]
u = np.zeros((n, l))
u[p, :] = np.random.randn(k, l)
b = np.matmul(A, u)
mu = 1e-2
x_0 = np.random.randn(n, l)



def gl_mosek(x0: np.ndarray, A: np.ndarray, b: np.ndarray, mu: float, opts):
    out = Out()
    # 给定A,b的维数
    m, n = A.shape
    _, l = b.shape
    with Model() as M:
        x = M.variable("x",[n, l], Domain.unbounded())
        t = M.variable(n+m, Domain.unbounded())
        t0 = M.parameter("t0", m)
        t0.setValue(np.ones(m))
        M.constraint(Expr.hstack(t0, t.slice(0, m), Expr.sub(Expr.mul(A, x), b)), Domain.inRotatedQCone())
        M.constraint(Expr.hstack([t.slice(m, m + n), x]), Domain.inQCone())

        c = []
        for i in np.arange(m):
            c.append(1.0)
        for i in np.arange(n):
            c.append(mu)
        M.objective("obj", ObjectiveSense.Minimize, Expr.dot(c, t))
        M.solve()

        # r = np.dot(A, solu_x) - b
        # f_ = 0.5 * np.linalg.norm(r, ord='fro') ** 2 + mu * np.sum(np.linalg.norm(u, ord=2, axis=1))
        # print("optval:", f_)

        out.fval = M.primalObjValue()
        out.Runtime = M.getSolverDoubleInfo("optimizerTime")

        solu_x = M.getVariable('x').level().reshape(n ,l)
        out.itr = M.getSolverIntInfo("intpntIter")
        # print("primalObjValue:", M.primalObjValue())
        # print("solution x:", solu_x)

        # tm = M.getSolverDoubleInfo("optimizerTime")
        # it = M.getSolverIntInfo("intpntIter")
        # print('Time:{0}\nIterations:{1}'.format(tm, it))
    return solu_x, out.itr, out


