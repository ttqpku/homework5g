import numpy as np
import mosek
import gurobipy as gp
import cvxpy as cp
import sys
import time
from gl_cvx_mosek import gl_cvx_mosek
from gl_cvx_gurobi import gl_cvx_gurobi
from gl_mosek import gl_mosek
from gl_gurobi import gl_gurobi
from gl_SGD_primal import gl_SGD_primal
from LASSO_con import lasso_con, Opts, Out
from gl_ProxGD_primal import gl_ProxGD_primal_inner
from gl_FProxGD_primal import gl_FProxSGD_primal
from gl_ALM_dual import gl_ALM_dual, OptsALM
from gl_ADMM_primal import gl_ADMM_primal, OptsADMM
from gl_ADMM_dual import gl_ADMM_dual, OptsADMM
import matplotlib.pyplot as plt



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
b = np.matmul(A,u)
mu = 1e-2
x_0 = np.random.randn(n, l)


def errfun(x_1,x_2):
    return np.linalg.norm(x_1-x_2, ord='fro')/(1+np.linalg.norm(x_1, ord='fro'))


def errfun_exact(x):
    return np.linalg.norm(x-u, ord='fro')/(1+np.linalg.norm(u, ord='fro'))


def sparisity(x):
    return np.sum((np.abs(x.ravel()) > 1e-6 + 0) * np.max(np.abs(x.ravel())))/(n*l)

def plot_x(x,title):
    # hspace调节子图之间的竖直距离
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.4)
    plt.subplot(2, 1, 1)
    plt.scatter(np.arange(u.shape[0]) + 1, u[:, 0], color='w', marker='o', edgecolors='#E4A987')
    plt.scatter(np.arange(u.shape[0]) + 1, u[:, 1], color='#3C70AD', marker=(5, 2))
    plt.title(r"(1) exact solution $u$")
    plt.subplot(2, 1, 2)
    plt.scatter(np.arange(x.shape[0]) + 1, x[:, 0], color='w', marker='o', edgecolors='#E4A987')
    plt.scatter(np.arange(x.shape[0]) + 1, x[:, 1], color='#3C70AD', marker=(5, 2))
    plt.title("(2)"+title)
    plt.savefig(title + ".jpg")





if __name__ == "__main__":

    # cvx_mosek
    opts1 = []
    x_1, iter_1, out_1 = gl_cvx_mosek(x_0, A, b, mu, opts1)

    # cvx_gurobi
    opts2 = []
    x_2, iter_2, out_2 = gl_cvx_gurobi(x_0, A, b, mu, opts2)

    # mosek
    opts3 = []
    x_3, iter_3, out_3 = gl_mosek(x_0, A, b, mu, opts3)

    # gurobi
    opts4 = []
    x_4, iter_4, out_4 = gl_gurobi(x_0, A, b, mu, opts4)

    # SGD
    opts5 = Opts()
    opts5.method = gl_SGD_primal
    x_5, iter_5, out_5 = lasso_con(x_0, A, b, mu, opts=opts5)

    # ProxGD
    opts6 = Opts()
    opts6.method = gl_ProxGD_primal_inner
    x_6, iter_6, out_6 = lasso_con(x_0, A, b, mu, opts=opts6)

    # FProxGD
    opts7 = Opts()
    opts7.method = gl_FProxSGD_primal
    x_7, iter_7, out_7 = lasso_con(x_0, A, b, mu, opts=opts7)

    # ALM_dual
    opts8 = OptsALM()
    x_8, iter_8, out_8 = gl_ALM_dual(x_0, A, b, mu, opts=opts8)

    # ADMM_dual
    opts9 = OptsADMM()
    x_9, iter_9, out_9 = gl_ADMM_dual(x_0, A, b, mu, opts=opts9)

    # ADMM_primal
    opts10 = OptsADMM()
    opts10.maxit = 10000
    x_10, iter_10, out_10 = gl_ADMM_primal(x_0, A, b, mu, opts=opts10)


    print(f"CVX-Mosek: cpu: {out_1.Runtime:5.2f},iter: {out_1.itr:5d}, optval: {out_1.fval:6.5E}, sparisity: "
          f"{sparisity(x_1):4.3f}, err-to-exact: {errfun_exact(x_1):3.2E}, err-to-cvx-mosek:"
          f" {errfun(x_1, x_1):3.2E}, err-to-cvx-gurobi: {errfun(x_1, x_2):3.2E}")
    print(f"CVX-Gurobi: cpu: {out_2.Runtime:5.2f},iter: {out_2.itr:5d}, optval: {out_2.fval:6.5E}, sparisity: "
          f"{sparisity(x_2):4.3f}, err-to-exact: {errfun_exact(x_2):3.2E}, err-to-cvx-mosek:"
          f" {errfun(x_2, x_1):3.2E}, err-to-cvx-gurobi: {errfun(x_2, x_2):3.2E}")
    print(f"Mosek: cpu: {out_3.Runtime:5.2f},iter: {out_3.itr:5d}, optval: {out_3.fval:6.5E}, sparisity: "
          f"{sparisity(x_3):4.3f}, err-to-exact: {errfun_exact(x_3):3.2E}, err-to-cvx-mosek:"
          f" {errfun(x_3, x_1):3.2E}, err-to-cvx-gurobi: {errfun(x_3, x_2):3.2E}")
    print(f"Gurobi: cpu: {out_4.Runtime:5.2f},iter: {out_4.itr:5d}, optval: {out_4.fval:6.5E}, sparisity: "
          f"{sparisity(x_4):4.3f}, err-to-exact: {errfun_exact(x_4):3.2E}, err-to-cvx-mosek:"
          f" {errfun(x_4, x_1):3.2E}, err-to-cvx-gurobi: {errfun(x_4, x_2):3.2E}")
    print(f"SGD Primal: cpu: {out_5.Runtime:5.2f},iter: {out_5.itr:5d}, optval: {out_5.fval:6.5E}, sparisity: "
          f"{sparisity(x_5):4.3f}, err-to-exact: {errfun_exact(x_5):3.2E}, err-to-cvx-mosek:"
          f" {errfun(x_5, x_1):3.2E}, err-to-cvx-gurobi: {errfun(x_5, x_2):3.2E}")
    print(f"ProxGD Primal: cpu: {out_6.Runtime:5.2f},iter: {out_6.itr:5d}, optval: {out_6.fval:6.5E}, sparisity: "
          f"{sparisity(x_6):4.3f}, err-to-exact: {errfun_exact(x_6):3.2E}, err-to-cvx-mosek:"
          f" {errfun(x_6, x_1):3.2E}, err-to-cvx-gurobi: {errfun(x_6, x_2):3.2E}")
    print(f"FProxGD Primal: cpu: {out_7.Runtime:5.2f},iter: {out_7.itr:5d}, optval: {out_7.fval:6.5E}, sparisity: "
          f"{sparisity(x_7):4.3f}, err-to-exact: {errfun_exact(x_7):3.2E}, err-to-cvx-mosek:"
          f" {errfun(x_7, x_1):3.2E}, err-to-cvx-gurobi: {errfun(x_7, x_2):3.2E}")
    print(f"ALM Dual: cpu: {out_8.Runtime:5.2f},iter: {out_8.itr:5d}, optval: {out_8.fval:6.5E}, sparisity: "
          f"{sparisity(x_8):4.3f}, err-to-exact: {errfun_exact(x_8):3.2E}, err-to-cvx-mosek:"
          f" {errfun(x_8, x_1):3.2E}, err-to-cvx-gurobi: {errfun(x_8, x_2):3.2E}")
    print(f"ADMM Dual: cpu: {out_9.Runtime:5.2f},iter: {out_9.itr:5d}, optval: {out_9.fval:6.5E}, sparisity: "
          f"{sparisity(x_9):4.3f}, err-to-exact: {errfun_exact(x_9):3.2E}, err-to-cvx-mosek:"
          f" {errfun(x_9, x_1):3.2E}, err-to-cvx-gurobi: {errfun(x_9, x_2):3.2E}")
    print(f"ADMM Primal: cpu: {out_10.Runtime:5.2f},iter: {out_10.itr:5d}, optval: {out_10.fval:6.5E}, sparisity: "
          f"{sparisity(x_10):4.3f}, err-to-exact: {errfun_exact(x_10):3.2E}, err-to-cvx-mosek:"
          f" {errfun(x_10, x_1):3.2E}, err-to-cvx-gurobi: {errfun(x_10, x_2):3.2E}")

    plot_x(x_1, "CVX-Mosek")
    plot_x(x_2, "CVX-Gurobi")
    plot_x(x_3, "Mosek")
    plot_x(x_4, "Gurobi")
    plot_x(x_5, "SGD Primal")
    plot_x(x_6, "ProxGD Primal")
    plot_x(x_7, "FProxGD Primal")
    plot_x(x_8, "ALM Dual")
    plot_x(x_9, "ADMM Dual")
    plot_x(x_10, "ADMM Primal")





