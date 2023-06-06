import numpy as np
import sys
from math import sqrt
from tqdm import tqdm
from LASSO_con import Opts, Out, lasso_con

def prox(x, mu):
    nrmx = np.linalg.norm(x, ord=2, axis=1)
    flag = nrmx > mu
    prox_x = x - mu * x / (nrmx.reshape(-1, 1) + 1e-10)
    prox_x = prox_x * flag.reshape(-1, 1)
    return prox_x


def BBupdate(y, yp, g, gp, k, alpha):
    dy = y - yp
    dg = g - gp
    dyg = np.abs(np.sum(dy * dg))
    if dyg > 0:
        if np.mod(k, 2) == 1:
            alpha = (np.sum(dy * dy) / dyg)
        else:
            alpha = (dyg / np.sum(dg * dg))
    alpha = max(min(alpha, 1e12), 1e-12)

    return alpha


def gl_FProxSGD_primal(x0: np.ndarray, A: np.ndarray, b: np.ndarray, mu: float, mu0: float, opts: Opts):
    x = x0
    y = x
    xp = x0
    r = np.matmul(A, y) - b
    g = np.matmul(A.T, r)
    tmp = 0.5 * np.linalg.norm(r, ord='fro') ** 2
    tmpf = tmp + mu * np.sum(np.linalg.norm(x, ord=2, axis=1))
    f = tmp + mu0 * np.sum(np.linalg.norm(x, ord=2, axis=1))
    nrmG = np.linalg.norm(x - prox(x - g, mu), ord="fro")
    out = Out()
    f_best = 10000000.
    Cval = tmpf
    alpha = opts.alpha0
    for k in np.arange(opts.maxit):
        yp = y
        gp = g
        theta = (k - 1) / (k + 2)  # k starts with 0
        y = x + theta * (x - xp)
        xp = x
        r = np.matmul(A, y) - b
        g = np.matmul(A.T, r)

        out.g_hist.append(nrmG)
        out.f_hist.append(f)
        f_best = np.min([f_best, f])

        out.f_hist_best.append(f_best)

        if k > 2 and np.abs(out.f_hist[k] - out.f_hist[k - 1]) < opts.ftol and out.g_hist[k] < opts.gtol:
            break

        # BB
        alpha = BBupdate(y, yp, g, gp, k, alpha)
        x = prox(y - alpha * g, alpha * mu)

        nls = 1
        while 1:
            tmp = 0.5 * np.linalg.norm(np.matmul(A, x) - b, ord='fro') ** 2
            tmpf = tmp + mu * np.sum(np.linalg.norm(x, ord=2, axis=1))

            # if (tmpf <= Cval - 0.5 * alpha * opts.rhols * np.sum(np.linalg.norm(x - yp, ord=2, axis=1)) ** 2) or (nls >= 10):
            if (tmpf <= Cval - 0.5 * alpha * opts.rhols * nrmG ** 2) or (nls >= 10):
                break
            alpha = opts.eta * alpha
            nls = nls + 1
            x = prox(y - alpha * g, alpha * mu)
        f = tmp + mu0 * np.sum(np.linalg.norm(x, ord=2, axis=1))

        nrmG = np.linalg.norm(x - y, ord='fro') / alpha
        Qp = opts.Q
        opts.Q = opts.gamma * Qp + 1
        Cval = (opts.gamma * Qp * Cval + tmpf) / opts.Q

    out.itr = k + 1

    return x, out


if __name__ == "__main__":
    seed = 97006855
    seed = 7897333
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
    mu = 1e-2
    x0 = np.random.randn(n, l)

    r = np.dot(A, u) - b
    f_ = 0.5 * np.linalg.norm(r, ord='fro') ** 2 + mu0 * np.sum(np.linalg.norm(u, ord=2, axis=1))
    print(f_)
    print(u)

    opts = Opts()
    opts.method = gl_FProxSGD_primal
    x, iter_, out = lasso_con(x0, A, b, mu0, opts)
    print("solution:", x)
    print("exact solution:", u)
    print("iter_num_outer:", out.itr)
    print("iter_num_inner:", out.itr_inn)
    print("f_val:", out.fval)

