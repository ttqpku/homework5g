import numpy as np
import time


class OptsALM:
    def __init__(self):
        self.sigma = 10
        self.maxit = 100
        self.maxit_inn = 300
        self.thre = 1e-5
        self.thre_inn = 1e-3




def update_s(ref, mu):
    norm = np.linalg.norm(ref, axis=1, keepdims=True)
    norm[norm < mu] = mu
    return ref * (mu / norm)


def gl_ALM_dual(x0: np.ndarray, A: np.ndarray, b: np.ndarray, mu: float, opts: OptsALM):
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

    lambda_k = -x0

    sigma = opts.sigma  # 二次罚函数系数
    inv = np.linalg.inv(np.eye(m) + sigma * A @ A.T)
    s = np.zeros((n, l))

    for it in range(opts.maxit):
        for it2 in range(opts.maxit_inn):
            y = inv @ (sigma * A @ s - A @ lambda_k - b)
            sp = s
            s = update_s(lambda_k / sigma + A.T @ y, mu)
            inner_gap = np.linalg.norm(s - sp, 'fro')
            out.itr += 1
            if inner_gap < opts.thre_inn:
                break

        lambda_k = lambda_k + sigma * (A.T @ y - s)

        f = 0.5 * np.linalg.norm(A @ (-lambda_k) - b, ord='fro') ** 2 + mu * np.sum(np.linalg.norm(lambda_k, ord=2, axis=1))
        f_dual = 0.5 * np.linalg.norm(y, ord='fro') ** 2 + np.sum(y * b)
        out.prim_hist.append(f)
        out.dual_hist.append(f_dual)

        if np.linalg.norm(A.T @ y - s) < opts.thre:
            break

    out.fval = out.prim_hist[-1]
    out.Runtime = time.time() - start

    return -lambda_k, out.itr, out


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

    opts = OptsALM()
    x, iter_, out = gl_ALM_dual(x0, A, b, mu0, opts)
    print(x)
    print(iter_)
    print(out.fval)
    print(out.OptTime)
