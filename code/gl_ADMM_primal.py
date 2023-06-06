import numpy as np
import time

class OptsADMM:
    def __init__(self):
        self.sigma = 10
        self.maxit = 10000
        self.thre = 1e-6





def prox(x, mu):
    nrmx = np.linalg.norm(x, ord=2, axis=1).reshape(-1, 1)
    flag = nrmx > mu
    prox_x = (nrmx - mu) * x  / (nrmx + 1e-10)
    prox_x = prox_x * flag

    return prox_x


def gl_ADMM_primal(x0: np.ndarray, A: np.ndarray, b: np.ndarray, mu: float, opts:OptsADMM):
    class Out:
        def __init__(self):
            self.itr = 0
            self.prim_hist = []
            self.dual_hist = []
            self.fval = 0
            self.Runtime = 0.
    out = Out()
    start = time.time()
    m, n = A.shape
    _, l = b.shape


    x = x0
    y = x0
    z = np.zeros((n, l))

    sigma = opts.sigma
    inv = np.linalg.inv(sigma * np.eye(n) + A.T @ A)
    ATb = A.T @ b

    for it in range(opts.maxit):
        x = inv @ (sigma * y + ATb - z)
        y0 = y
        y = prox(x + z / sigma, mu / sigma)
        z = z + sigma * (x - y)

        primal_sat = np.linalg.norm(x - y)
        dual_sat = np.linalg.norm(y0 - y)
        f = 0.5 * np.linalg.norm(A @ x - b, ord='fro') ** 2 + mu * np.sum(np.linalg.norm(x, ord=2, axis=1))
        out.prim_hist.append(f)
        out.itr += 1

        if primal_sat < opts.thre and dual_sat < opts.thre:
            break
    out.fval = f
    out.Runtime = time.time() - start
    return x, out.itr, out


if __name__ == "__main__":
    seed = 97006855
    np.random.seed(seed)
    n = 512
    m = 256
    A = np.random.randn(m, n)
    k = round(n * 0.1)
    l = 2
    A = np.random.randn(m, n)
    p = np.random.permutation(n)
    p = p[:k]
    u = np.zeros((n, l))
    u[p, :] = np.random.randn(k, l)
    b = np.matmul(A, u)
    mu0 = 1e-2
    x0 = np.random.randn(n, l)
    # x0 = np.zeros((n, l))

    r = np.dot(A, u) - b
    f_ = 0.5 * np.linalg.norm(r, ord='fro') ** 2 + mu0 * np.sum(np.linalg.norm(u, ord=2, axis=1))
    print(f_)
    print(u)

    opts = OptsADMM()
    x, iter_, out = gl_ADMM_primal(x0, A, b, mu0, opts)
    print(x)
    print(iter_)
    print(out.fval)
    print(out.OptTime)
