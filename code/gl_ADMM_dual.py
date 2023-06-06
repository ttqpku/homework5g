import numpy as np
import time


class OptsADMM:
    def __init__(self):
        self.sigma = 10
        self.maxit = 1000
        self.thre = 1e-6





def update_z(ref, mu):
    norm = np.linalg.norm(ref, axis=1, keepdims=True)
    norm[norm < mu] = mu
    return ref * (mu / norm)


def gl_ADMM_dual(x0: np.ndarray, A: np.ndarray, b: np.ndarray, mu: float, opts: OptsADMM):
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

    sigma = opts.sigma  # 二次罚函数系数
    inv = np.linalg.inv(np.eye(m) + sigma * A @ A.T)
    z = np.zeros((n, l))

    for it in range(opts.maxit):
        y = inv @ (A @ x - sigma * A @ z - b)
        zp = z
        z = update_z(x / sigma - A.T @ y, mu)
        inner_gap = np.linalg.norm(z - zp, 'fro')
        out.itr += 1

        x = x - sigma * (A.T @ y + z)

        f = 0.5 * np.linalg.norm(A @ x - b, ord='fro') ** 2 + mu * np.sum(np.linalg.norm(x, ord=2, axis=1))
        f_dual = 0.5 * np.linalg.norm(y, ord='fro') ** 2 + np.sum(y * b)
        out.prim_hist.append(f)
        out.dual_hist.append(f_dual)

        if np.linalg.norm(A.T @ y + z) < opts.thre:
            break

    out.fval = out.prim_hist[-1]
    out.Runtime = time.time() - start

    return x,  out.itr, out


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
    x, iter_, out = gl_ADMM_dual(x0, A, b, mu0, opts)
    print(x)
    print(iter_)
    print(out.fval)
    print(out.OptTime)
