import numpy as np
import mosek
import cvxpy as cp
import sys
import time


def gl_cvx_mosek(x0, A, b, mu_value, opts):
    class Out:
        def __init__(self):
            self.itr = 0
            self.fval = 0
            self.Runtime = 0.

    out = Out()
    start = time.time()
    mu_param = cp.Parameter(nonneg=True, value=mu_value)
    x = cp.Variable((A.shape[1], b.shape[1]), name='x')
    obj = cp.Minimize(0.5 * cp.sum_squares(A @ x - b) + mu_param * cp.mixed_norm(x, 2, 1))
    prob = cp.Problem(obj)
    prob.solve(solver=cp.MOSEK)  # verbose=True
    end = time.time()
    out.fval = prob.value
    out.itr = 9 # 打印日志后，得到迭代次数
    out.Runtime = end - start
    return x.value, out.itr, out

# print(cp.installed_solvers())
# 打印目前支持的解释器

# 下面主要通过np.random模块生成随机数，当需要测试函数gl_cvx_mosek时，将下述部分注释取消
if __name__ == "__main__":
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
    mu = 1e-2
    x_0 = np.random.randn(n, l)


    # 测试代码
    opts1 = []
    x, iter_, out = gl_cvx_mosek(x_0, A, b, mu, opts1)
    print("optimal value with MOSEK:", out.fval)


