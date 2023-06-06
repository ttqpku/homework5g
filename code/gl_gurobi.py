import numpy as np
import gurobipy as gp
from gurobipy import *
import time

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



def gl_gurobi(x0: np.ndarray, A: np.ndarray, b: np.ndarray, mu: float, opts):
    class Out:
        def __init__(self):
            self.fval = 0.
            self.Runtime = 0.
            self.itr = 0
    # 创建模型
    out = Out()
    start = time.time()
    # 给定A,b的维数
    m, n = A.shape
    _, l = b.shape
    Model = gp.Model()
    x = Model.addVars(n, l, lb=-GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, name="x")
    t = Model.addVars(n, vtype=gp.GRB.CONTINUOUS, name="t")

    Model.update()

    Model.setObjective(0.5 * gp.quicksum((gp.quicksum(A[i, j] * x[j, k] for j in np.arange(n)) - b[i, k]) ** 2
                                         for k in np.arange(l) for i in np.arange(m)) + mu * gp.quicksum(t[i] for i in np.arange(n)), gp.GRB.MINIMIZE)
    for i in np.arange(n):
        Model.addQConstr(gp.quicksum(x[i, k] ** 2 for k in np.arange(l)), GRB.LESS_EQUAL, t[i] * t[i])
    # Model.addConstrs(np.linalg.norm(X[i, :], 2) <= t[i] for i in np.arange(n))

    Model.setParam("OutputFlag", 0)
    Model.setParam("BarCorrectors", 1000)
    Model.optimize()
    # OutputFlag=0
    out.fval = Model.objVal
    out.Runtime = Model.Runtime
    out.itr = Model.BarIterCount
    # print(Model.getAttr(GRB.Attr.X, Model.getVars()))
    # for v in Model.getVars():
    #     print(v.X)
    # # print(Vars.value())
    # print("Obj:", Model.objVal)
    # print("Runtime:", Model.Runtime)
    # print("IterCount:", Model.IterCount)
    solu_x = Model.getAttr(GRB.Attr.X, Model.getVars())
    solu_x_ = np.array(solu_x)[0:n*l].reshape(n, l)
    return solu_x_, out.itr, out


# x_2, _, _ = gl_gurobi(x_0, A, b, mu, [])
# print(x_2)