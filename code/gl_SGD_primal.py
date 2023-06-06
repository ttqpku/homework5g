import numpy as np
import sys
from math import sqrt
from tqdm import tqdm
from LASSO_con import Opts, Out, lasso_con




def set_step(k, opts):
    type = opts.step_type
    if type == "fixed":
        return opts.alpha0
    elif type == "diminishing":
        return opts.alpha0 / sqrt(k+1)
    elif type == "diminishing2":
        return opts.alpha0 / (k+1)**2
    else:
        print("unsupported type.")


def min_(a, b):
    if a <= b:
        return a
    else:
        return b


def gl_SGD_primal(x0,A,b,mu,mu0,opts):
    x = x0
    r = np.matmul(A, x) - b
    g = np.matmul(A.T, r)
    norm_x = np.linalg.norm(x, axis=1).reshape((-1, 1))
    sub_g = x / ((norm_x <= 1e-6) + norm_x)
    sub_g = sub_g * mu + g
    nrmG = np.linalg.norm(g, ord='fro')
    f = 0.5 * np.linalg.norm(r, ord='fro') ** 2 + mu0 * np.sum(np.linalg.norm(x, ord=2, axis=1))
    tmpf = 0.5 * np.linalg.norm(r, ord='fro') ** 2 + mu * np.sum(np.linalg.norm(x, ord=2, axis=1))
    out = Out()
    f_best = 1000000000
    Cval = tmpf
    alpha = opts.alpha0
    for k in np.arange(opts.maxit):
        fp = f
        gp = g
        xp = x

        out.g_hist.append(nrmG)
        # 记录可微分部分的F范数
        out.f_hist.append(f)
        f_best = min_(f_best, f)
        out.f_hist_best.append(f_best)

        if k > 2 and np.abs(out.f_hist[k]-out.f_hist[k-1]) < opts.ftol:
            out.flag = 1
            break

        nls = 1
        while 1:
            x = xp - alpha * sub_g
            tmpf = 0.5 * np.linalg.norm(r, ord='fro') ** 2 + mu * np.sum(np.linalg.norm(x, ord=2, axis=1))

            if (tmpf <= Cval - 0.5 * alpha * opts.rhols * nrmG ** 2) or (nls >= 10):
                break
            alpha = opts.eta * alpha
            nls = nls + 1

        r = np.matmul(A, x) - b
        g = np.matmul(A.T, r)
        norm_x = np.linalg.norm(x, axis=1).reshape((-1, 1))
        sub_g = x / ((norm_x <= 1e-6) + norm_x)
        sub_g = sub_g * mu + g

        # BB
        dx = x - xp
        dg = g - gp
        dxg = np.abs(np.sum(dx * dg))
        if dxg > 0:
            if np.mod(k, 2) == 1:
                alpha = (np.sum(dx * dx) / dxg)
            else:
                alpha = (dxg / np.sum(dg * dg))
        alpha = max(min(alpha, 1e12), 1e-12)

        nrmG = np.linalg.norm(g, ord='fro')
        f = 0.5 * np.linalg.norm(np.matmul(A, x) - b, ord='fro') ** 2 + mu0 * np.sum(np.linalg.norm(x, ord=2, axis=1))
        Qp = opts.Q
        Q = opts.gamma * Qp + 1
        opts.Q = Q
        Cval = (opts.gamma * Qp * Cval + tmpf) / Q

    out.itr = k + 1
    return x, out

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

    # r = np.dot(A, u) - b
    # f_ = 0.5 * np.linalg.norm(r, ord='fro') ** 2 + mu * np.sum(np.linalg.norm(u, ord=2, axis=1))
    # print(f_)

    opts = Opts()
    opts.method = gl_SGD_primal
    x,iter_,  out = lasso_con(x_0, A, b, mu, opts)
    print("solution:", x)
    print("exact solution:", u)
    print("iter_num:", out.itr)
    print(out.fval)
    # print("f_hist_best:", out.f_hist_best)
    # print("f_hist:", out.f_hist)
    # print("g_hist:", out.g_hist)

