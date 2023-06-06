import numpy as np
import sys
from math import sqrt
from tqdm import tqdm
import time

class Opts:
    def __init__(self):
        self.maxit = 100
        self.maxit_inn = 300
        self.alpha0 = 1.
        self.mu1 = 100
        self.thres = 1e-5
        self.ftol = 1e-10
        self.ftol_init_ratio = 1e5
        self.gtol = 1e-6
        self.gtol_init_ratio = 1 / self.gtol
        self.factor = 0.1
        self.etaf = 0.1
        self.etag = 0.1
        self.eta = 0.2
        self.rhols = 1e-6
        self.gamma = 0.85
        self.Q = 1
        self.opts1 = None
        self.method = None
class Out:
    def __init__(self):
        self.itr = 0
        self.f_hist = []
        self.f_hist_best = []
        self.g_hist = []
        self.itr_inn = 0
        self.flag = 0
        self.fval = 0
        self.Runtime = 0.


def prox(x, mu):
    nrmx = np.linalg.norm(x, ord=2, axis=1)
    flag = nrmx > mu
    prox_x = x - mu * x / (nrmx.reshape(-1, 1) + 1e-10)
    prox_x = prox_x * flag.reshape(-1, 1)
    return prox_x


def lasso_con(x0, A, b, mu0, opts):
    start = time.time()
    eigs = np.linalg.eig(np.matmul(A.T, A))[0]
    eigs = np.real(eigs[np.isreal(eigs)])
    opts.alpha0 = 1 / np.max(eigs)
    opts.opts1 = Opts()
    opts1 = opts.opts1
    out = Out()


    opts1.ftol = opts.ftol * opts.ftol_init_ratio
    opts1.gtol = opts.gtol * opts.gtol_init_ratio
    out.itr_inn = 0

    x = x0
    r = np.matmul(A, x) - b
    mu_t = opts.mu1
    f = 0.5 * np.linalg.norm(r, ord='fro') ** 2 + mu_t * np.sum(np.linalg.norm(x, ord=2, axis=1))

    for k in range(opts.maxit):
        opts1.maxit = opts.maxit_inn
        opts1.gtol = max(opts1.gtol * opts.etag, opts.gtol)
        opts1.ftol = max(opts1.ftol * opts.etaf, opts.ftol)
        opts1.alpha0 = opts.alpha0
        fp = f
        algf = opts.method
        x, out1 = algf(x, A, b, mu_t, mu0, opts1)
        f = out1.f_hist[-1]
        out.f_hist.extend(out1.f_hist)

        r = np.matmul(A, x) - b
        nrmG = np.linalg.norm(x - prox(x - np.matmul(A.T, r), mu0), ord="fro")

        if not out1.flag:
            mu_t = max(mu_t * opts.factor, mu0)

        if mu_t == mu0 and (nrmG < opts.gtol or abs(f - fp) < opts.ftol):
            break

        out.itr_inn = out.itr_inn + out1.itr

    out.fval = f
    out.itr = k + 1
    out.Runtime = time.time()-start

    return x, out.itr, out


