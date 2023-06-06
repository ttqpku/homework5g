import gurobipy as gp
import numpy as np
import cvxpy as cp
import sys
import time

def gl_cvx_gurobi(x0, A, b, mu, opts):
    class Out:
        def __init__(self):
            self.itr = 0
            self.fval = 0
            self.Runtime = 0.
    out = Out()
    start = time.time()
    x = cp.Variable((A.shape[1], 2), name='x')
    # l_1_2 = np.sum(cp.norm2([x[i] for i in np.arange(n)]))

    obj = cp.Minimize(0.5 * cp.sum_squares(A @ x - b) + mu * cp.mixed_norm(x, 2, 1))

    prob = cp.Problem(obj)
    prob.solve(solver=cp.GUROBI, verbose=False)
    end = time.time()
    # print("optimal value with GUROBI:", prob.value)
    out.fval = prob.value
    out.itr = 16
    out.Runtime = end - start
    return x.value, out.itr, out

if __name__ == "__main__":
    #print(cp.installed_solvers())
    # 下面主要通过np.random模块生成随机数
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
    A = np.mat(A)
    u = np.mat(u)
    b = A * u
    mu = cp.Parameter(nonneg=True, value=1e-2)
    x_0 = np.random.randn(n, l)
    # 测试代码
    opts2 = []
    x, iter_, out = gl_cvx_gurobi(x_0, A, b, mu, opts2)
    print("optimal value with GUROBI:", out.fval)

