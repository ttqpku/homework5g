import numpy as np
import sys
from math import sqrt
from tqdm import tqdm
from LASSO_con import Opts, Out, lasso_con

def prox(x, mu):
    nrmx = np.linalg.norm(x, ord=2, axis=1).reshape(-1, 1)
    flag = nrmx > mu
    prox_x = (nrmx - mu) * x / (nrmx + 1e-10)
    prox_x = prox_x * flag
    return prox_x


def BBupdate(x, xp, g, gp, k, alpha):
    dx = x - xp
    dg = g - gp
    dxg = np.abs(np.sum(dx * dg))
    if dxg > 0:
        if np.mod(k, 2) == 1:
            alpha = (np.sum(dx * dx) / dxg)
        else:
            alpha = (dxg / np.sum(dg * dg))
    alpha = max(min(alpha, 1e12), 1e-12)

    return alpha


def gl_ProxGD_primal_inner(x0: np.ndarray, A: np.ndarray, b: np.ndarray, mu: float, mu0: float, opts: Opts):
    x = x0
    r = np.matmul(A, x) - b
    g = np.matmul(A.T, r)
    tmp = 0.5 * np.linalg.norm(r, ord='fro') ** 2
    tmpf = tmp + mu * np.sum(np.linalg.norm(x, ord=2, axis=1))
    f = tmp + mu0 * np.sum(np.linalg.norm(x, ord=2, axis=1))
    nrmG = np.linalg.norm(x - prox(x - g, mu), ord="fro")
    out = Out()
    f_best = 10000000
    Cval = tmpf
    alpha = opts.alpha0
    for k in np.arange(opts.maxit):
        fp = f
        gp = g
        xp = x

        out.g_hist.append(nrmG)
        out.f_hist.append(f)
        f_best = np.min([f_best, f])

        out.f_hist_best.append(f_best)

        if k > 2 and np.abs(out.f_hist[k] - out.f_hist[k - 1]) < opts.ftol and out.g_hist[k] < opts.gtol:
            out.flag = 1
            break

        nls = 1
        while 1:
            x = prox(xp - alpha * g, alpha * mu)
            tmp = 0.5 * np.linalg.norm(np.matmul(A, x) - b, ord='fro') ** 2
            tmpf = tmp + mu * np.sum(np.linalg.norm(x, ord=2, axis=1))

            if (tmpf <= Cval - 0.5 * alpha * opts.rhols * nrmG ** 2) or (nls >= 10):
                break
            alpha = opts.eta * alpha
            nls = nls + 1

        r = np.matmul(A, x) - b
        g = np.matmul(A.T, r)
        f = tmp + mu0 * np.sum(np.linalg.norm(x, ord=2, axis=1))

        nrmG = np.linalg.norm(x - xp, ord='fro') / alpha
        Qp = opts.Q
        opts.Q = opts.gamma * Qp + 1
        Cval = (opts.gamma * Qp * Cval + tmpf) / opts.Q

        # BB
        alpha = BBupdate(x, xp, g, gp, k, alpha)

    out.itr = k + 1

    return x,  out

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
    b = np.matmul(A, u)
    mu = 1e-2
    x_0 = np.random.randn(n, l)
    r = np.matmul(A, u) - b
    g = np.matmul(A.T, r)
    tmp = 0.5 * np.linalg.norm(r, ord='fro') ** 2
    tmpf = tmp + mu * np.sum(np.linalg.norm(u, ord=2, axis=1))
    opts = Opts()
    opts.method = gl_ProxGD_primal_inner
    x, iter_, out = lasso_con(x_0, A, b, mu, opts)
    print("solution:", x)
    print("exact solution:", u)
    print("iter_num_outer:", out.itr)
    print("iter_num_inner:", out.itr_inn)
    print("fval:", out.fval)
