import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import functools
from pyDOE import lhs

from functorch import make_functional, vmap, grad, jacrev, hessian

from collections import namedtuple, OrderedDict
import datetime
import time
from IPython.display import clear_output

import warnings
warnings.filterwarnings('ignore')

'''  Solve the following PDE

'''
'''-------------------------Empty cache and check devices-------------------------'''
torch.cuda.empty_cache()
torch.set_default_dtype(torch.float64)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('device = ', device)
'''-------------------------Pre-setup-------------------------'''
# iteration counts and check
tr_iter_max    = 10000                      # max. iteration
ts_input_new   = 1000                       # renew testing points
ls_check       = 1000
ls_check0      = ls_check - 1
# number of training points and testing points
N_tsd_final = 250000 #100*N_trd
N_tsg_final = 1000   #10*N_trg
# tolerence for LM
tol_main    = 10**(-12)
tol_machine = 10**(-15)
mu_max      = 10**8
mu_ini      = 10**8
'''-------------------------Data generator-------------------------'''
# Ω:[-1,1]×[-1,1]上随机取点
def get_omega_points(num):
    x1 = 2 * torch.rand(num, 1) - 1
    x2 = 2 * torch.rand(num, 1) - 1
    x = torch.cat((x1, x2), dim=1)
    return x

# 边界上取点
def get_boundary_points(num):
    index1 = torch.rand(num,1) * 2 - 1
    index2 = torch.rand(num, 1) * 2 - 1
    index3 = torch.rand(num, 1) * 2 - 1
    index4 = torch.rand(num, 1) * 2 - 1
    xb1 = torch.cat((index1, torch.ones_like(index1)), dim=1)
    xb2 = torch.cat((index2, torch.full_like(index2, -1)), dim=1)
    xb3 = torch.cat((torch.ones_like(index3), index3), dim=1)
    xb4 = torch.cat((torch.full_like(index4, -1), index4), dim=1)
    xb = torch.cat((xb1, xb2, xb3, xb4), dim=0)
    return xb

# Γ上取点
def get_interface_points(num, r = 0.5):
    theta = torch.rand(num, 1) * torch.pi * 2
    x1 = r*torch.cos(theta)
    x2 = r*torch.sin(theta)
    x = torch.cat((x1, x2), dim=1)
    return x

# 构造三维数据[x1,x2,1]和[x1,x2,-1]
def add_dimension_positive(x):
    xx = torch.unsqueeze(torch.ones_like(x[:,0]), 1)
    x_3 = torch.cat((x,xx), dim=1)
    return x_3

def add_dimension_negetive(x):
    xx = torch.unsqueeze(-1*torch.ones_like(x[:, 0]), 1)
    x_3 = torch.cat((x, xx), dim=1)
    return x_3

# 构造三维数据[x1,x2,1]和[x1,x2,-1]
def add_dimension(x):
    r = torch.sum(torch.pow(x, 2), 1)
    r = torch.unsqueeze(r, 1)
    index_p = r > 0.25
    index_n = r <= 0.25
    index_p = index_p.float()
    index_n = index_n.float()
    index = index_p - index_n
    x_3 = torch.cat((x, index), dim=1)
    return x_3
'''-------------------------Functions-------------------------'''
# 获取界面处的法向导数
def get_normal_interface(x):
    r = torch.sqrt(torch.sum(torch.pow(x, 2), 1))
    n1 = x[:, 0] / r
    n2 = x[:, 1] / r
    n1 = torch.unsqueeze(n1, 1)
    n2 = torch.unsqueeze(n2, 1)
    n = torch.cat((n1, n2), dim=1)
    return n

def u_x(x):
    x_3 = add_dimension(x)
    index = x_3[:,2]
    r2 = torch.sum(torch.pow(x, 2), 1)
    u1 = -1 * torch.log(r2) + torch.sum(torch.sin(x), 1)
    u2 = -1 * torch.log(0.25*torch.ones_like(r2)) + torch.sum(torch.sin(x), 1)
    u_x = torch.where(index >= 0, u1, u2)
    u_x = torch.unsqueeze(u_x, 1)
    return u_x

def g_x(x):
    r2 = torch.sum(torch.pow(x, 2), 1)
    g = -1 * torch.log(r2) + torch.sum(torch.sin(x), 1)
    g = torch.unsqueeze(g, 1)
    return g

def f_x(x):
    x_3 = add_dimension(x)
    index = x_3[:,2]
    r2 = torch.sum(torch.pow(x, 2), 1)
    f1 = torch.log(r2) - 2 * torch.sum(torch.sin(x), 1)
    f2 = torch.log(0.25*torch.ones_like(r2)) - 2 * torch.sum(torch.sin(x), 1)
    f = torch.where(index >= 0, f1, f2)
    f = torch.unsqueeze(f, 1)
    return f
'''-------------------------Define networks-------------------------'''
class NeuralNet_Shallow(torch.nn.Module):

    ### in_dim: dimension of input; h_dim: number of neurons; out_dim: dimension of output

    def __init__(self, in_dim, h_dim, out_dim):
        super(NeuralNet_Shallow, self).__init__()
        self.ln1 = nn.Linear(in_dim, h_dim)
        self.act1 = nn.Sigmoid()
        # self.act1 = nn.Tanh()
        # self.act1 = nn.ReLU()

        self.ln2 = nn.Linear(h_dim, out_dim, bias=False)

    def forward(self, x):
        out = self.ln1(x)
        out = self.act1(out)
        out = self.ln2(out)
        return out


class NeuralNet_Deep(torch.nn.Module):

    ### in_dim: dimension of input; h_dim: number of neurons; out_dim: dimension of output
    ### depth: depth of the network
    def __init__(self, in_dim, h_dim, out_dim, depth):
        super(NeuralNet_Deep, self).__init__()
        self.depth = depth - 1
        self.list = nn.ModuleList()
        self.ln1 = nn.Linear(in_dim, h_dim)
        self.act1 = nn.Sigmoid()
        #self.act1 = nn.Tanh()
        # self.act1 = nn.ReLU()

        for i in range(self.depth):
            self.list.append(nn.Linear(h_dim, h_dim))

        self.lnd = nn.Linear(h_dim, out_dim, bias=False)

    def forward(self, x):
        out = self.ln1(x)
        out = self.act1(out)
        for i in range(self.depth):
            out = self.list[i](out)
            out = self.act1(out)
        out = self.lnd(out)
        return out

'''-------------------------Loss functions-------------------------'''
def func_loss_op(func_params, x_o, f_o):
    def f(x, func_params):
        fx = func_model(func_params, x)
        return fx.squeeze(0).squeeze(0)
        # derivatives of u wrt inputs

    d2u = jacrev(jacrev(f))(x_o, func_params)

    f_o = f_o[0]
    u = f(x_o, func_params)
    u_xx = d2u[0][0]
    u_yy = d2u[1][1]
    loss_op = u_xx + u_yy - u - f_o
    return loss_op

def func_loss_b(func_params, x_b, f_b):
    def f(x, func_params):
        fx = func_model(func_params, x)
        return fx.squeeze(0).squeeze(0)

    f_b = f_b[0]
    # function value at the boundary (Dirichlet)
    u = f(x_b, func_params)
    loss_b = u - f_b
    return 100*loss_b

def func_loss_if1(func_params, x_ifp, x_ifn):
    def f(x, func_params):
        fx = func_model(func_params, x)
        return fx.squeeze(0).squeeze(0)

    u_p = f(x_ifp, func_params)
    u_n = f(x_ifn, func_params)

    loss_if1 = u_p - u_n
    return 10*loss_if1

def func_loss_if2(func_params, x_ifp, x_ifn, nor):
    def f(x, func_params):
        fx = func_model(func_params, x)
        return fx.squeeze(0).squeeze(0)

    nor1 = nor[0]
    nor2 = nor[1]
    d1u_p = jacrev(f)(x_ifp, func_params)
    d1u_n = jacrev(f)(x_ifn, func_params)
    du_px = d1u_p[0]
    du_py = d1u_p[1]
    du_nx = d1u_n[0]
    du_ny = d1u_n[1]
    loss_if2 = (du_px - du_nx) * nor1 + (du_py - du_ny) * nor2 + 4
    return 10*loss_if2

'''-------------------------Levenberg-Marquardt (LM) optimizer-------------------------'''
# parameters counter
def count_parameters(func_params):
    return sum(p.numel() for p in func_params if p.requires_grad)

# get the model's parameter
def get_p_vec(func_params):
    p_vec = []
    cnt = 0
    for p in func_params:
        p_vec = p.contiguous().view(-1) if cnt == 0 else torch.cat([p_vec, p.contiguous().view(-1)])
        cnt = 1
    return p_vec

# Initialization of LM method
def generate_initial_LM(func_params, Xo_len, Xb_len, Xg_len1, Xg_len2):
    # data_length
    data_length = Xo_len + Xb_len + Xg_len1 + Xg_len2 # 输入数据长度和

    # p_vector p向量自然为model参数
    with torch.no_grad():
        p_vec_old = get_p_vec(func_params).double().to(device)

    # dp 初始所有参量搜索方向设置为0，其size应当和model参数一致
    dp_old = torch.zeros([count_parameters(func_params), 1]).double().to(device)

    # Loss 损失函数值同样设置为0
    L_old = torch.zeros([data_length, 1]).double().to(device)

    # Jacobian J矩阵同样
    J_old = torch.zeros([data_length, count_parameters(func_params)]).double().to(device)

    return p_vec_old, dp_old, L_old, J_old


def train_PINNs_LM(func_params, LM_setup, tr_input, lossval, lossval_dbg):
    # assign tuple elements of LM_set_up
    p_vec_o, dp_o, L_o, J_o, mu, criterion = LM_setup # old参数导入
    I_pvec = torch.eye(len(p_vec_o)).to(device) # 单位阵

    # assign tuple elements of data_input
    [X_o, F_o, X_b, F_b, X_ifp, X_ifn, nor, NL, NL_sqrt] = tr_input #训练参数

    # iteration counts and check
    Comput_old = True
    step = 0

    # try-except statement to avoid jam in the code
    try:
        while (lossval[-1] > tol_main) and (step <= tr_iter_max):

            torch.cuda.empty_cache()

            ############################################################
            # LM_optimizer
            if (Comput_old == True):  # need to compute loss_old and J_old

                ### computation of loss 计算各部分损失函数
                Lo = vmap((func_loss_op), (None, 0, 0))(func_params, X_o, F_o).flatten().detach()
                Lb = vmap((func_loss_b), (None, 0, 0))(func_params, X_b, F_b).flatten().detach()
                Lif1 = vmap((func_loss_if1), (None, 0, 0))(func_params, X_ifp, X_ifn).flatten().detach()
                Lif2 = vmap((func_loss_if2), (None, 0, 0, 0))(func_params, X_ifp, X_ifn, nor).flatten().detach()
                L = torch.cat((Lo / NL_sqrt[1], Lb / NL_sqrt[2], Lif1 / NL_sqrt[3], Lif2 / NL_sqrt[4]))
                L = L.reshape(NL[0], 1).detach()
                lsdp_sum = torch.sum(Lo * Lo) / NL[1]
                lsdn_sum = torch.sum(Lb * Lb) / NL[2]
                lsb_sum = torch.sum(Lif1 * Lif1) / NL[3]
                lsif_sum = torch.sum(Lif2 * Lif2) / NL[4]
                loss_dbg_old = [lsdp_sum.item(), lsdn_sum.item(), lsb_sum.item(), lsif_sum.item()]

            loss_old = lossval[-1]
            loss_dbg_old = lossval_dbg[-1]

            ### compute the gradinet of loss function for each point
            with torch.no_grad():
                p_vec = get_p_vec(func_params).detach()  # get p_vec for p_vec_old if neccessary

            if criterion:
                per_sample_grads = vmap(jacrev(func_loss_op), (None, 0, 0))(func_params, X_o, F_o)
                cnt = 0
                for g in per_sample_grads:
                    g = g.detach()
                    J_o = g.reshape(len(g), -1) if cnt == 0 else torch.hstack([J_o, g.reshape(len(g), -1)])
                    cnt = 1

                per_sample_grads = vmap(jacrev(func_loss_b), (None, 0, 0))(func_params, X_b, F_b)
                cnt = 0
                for g in per_sample_grads:
                    g = g.detach()
                    J_b = g.reshape(len(g), -1) if cnt == 0 else torch.hstack([J_b, g.reshape(len(g), -1)])
                    cnt = 1

                per_sample_grads = vmap(jacrev(func_loss_if1), (None, 0, 0))(func_params, X_ifp, X_ifn)
                cnt = 0
                for g in per_sample_grads:
                    g = g.detach()
                    J_if1 = g.reshape(len(g), -1) if cnt == 0 else torch.hstack([J_if1, g.reshape(len(g), -1)])
                    cnt = 1

                per_sample_grads = vmap(jacrev(func_loss_if2), (None, 0, 0, 0))(func_params, X_ifp, X_ifn, nor)
                cnt = 0
                for g in per_sample_grads:
                    g = g.detach()
                    J_if2 = g.reshape(len(g), -1) if cnt == 0 else torch.hstack([J_if2, g.reshape(len(g), -1)])
                    cnt = 1

                J = torch.cat((J_o / NL_sqrt[1], J_b / NL_sqrt[2], J_if1 / NL_sqrt[3], J_if2 / NL_sqrt[4])).detach()
                # 组装好了J矩阵
                ### info. normal equation of J
                J_product = J.t() @ J
                rhs = - J.t() @ L

            with torch.no_grad():
                ### solve the linear system
                dp = torch.linalg.solve(J_product + mu * I_pvec, rhs)
                cnt = 0
                for p in func_params:
                    mm = torch.Tensor([p.shape]).tolist()[0]
                    num = int(functools.reduce(lambda x, y: x * y, mm, 1))
                    p += dp[cnt:cnt + num].reshape(p.shape)
                    cnt += num

            ### Compute loss_new
            Lo = vmap((func_loss_op), (None, 0, 0))(func_params, X_o, F_o).flatten().detach()
            Lb = vmap((func_loss_b), (None, 0, 0))(func_params, X_b, F_b).flatten().detach()
            Lif1 = vmap((func_loss_if1), (None, 0, 0))(func_params, X_ifp, X_ifn).flatten().detach()
            Lif2 = vmap((func_loss_if2), (None, 0, 0, 0))(func_params, X_ifp, X_ifn, nor).flatten().detach()
            L = torch.cat((Lo / NL_sqrt[1], Lb / NL_sqrt[2], Lif1 / NL_sqrt[3], Lif2 / NL_sqrt[4]))
            L = L.reshape(NL[0], 1).detach()
            loss_new = torch.sum(L * L).item()
            lsdp_sum = torch.sum(Lo * Lo) / NL[1]
            lsdn_sum = torch.sum(Lb * Lb) / NL[2]
            lsb_sum = torch.sum(Lif1 * Lif1) / NL[3]
            lsif_sum = torch.sum(Lif2 * Lif2) / NL[4]
            loss_dbg_new = [lsdp_sum.item(), lsdn_sum.item(), lsb_sum.item(), lsif_sum.item()]

            # strategy to update mu
            if (step > 0):

                with torch.no_grad():

                    # accept update
                    if loss_new < loss_old:
                        p_vec_old = p_vec.detach()
                        dp_old = dp
                        L_old = L
                        J_old = J
                        mu = max(mu / mu_div, tol_machine)
                        criterion = True  # False
                        Comput_old = False
                        lossval.append(loss_new)
                        lossval_dbg.append(loss_dbg_new)

                    else:
                        cosine = nn.functional.cosine_similarity(dp, dp_old, dim=0, eps=1e-15)
                        cosine_check = (1. - cosine) * loss_new > min(lossval)  # loss_old
                        if cosine_check:  # give up the direction
                            cnt = 0
                            for p in func_params:
                                mm = torch.Tensor([p.shape]).tolist()[0]
                                num = int(functools.reduce(lambda x, y: x * y, mm, 1))
                                p -= dp[cnt:cnt + num].reshape(p.shape)
                                cnt += num
                            mu = min(mu_mul * mu, mu_max)
                            criterion = False
                            Comput_old = False
                        else:  # accept
                            p_vec_old = p_vec.detach()
                            dp_old = dp
                            L_old = L
                            J_old = J
                            mu = max(mu / mu_div, tol_machine)
                            criterion = True
                            Comput_old = False
                        lossval.append(loss_old)
                        lossval_dbg.append(loss_dbg_old)

            else:  # for old info.

                with torch.no_grad():

                    p_vec_old = p_vec.detach()
                    dp_old = dp
                    L_old = L
                    J_old = J
                    mu = max(mu / mu_div, tol_machine)
                    criterion = True
                    Comput_old = False
                    lossval.append(loss_new)
                    lossval_dbg.append(loss_dbg_new)

            if step % ls_check == ls_check0:
                print("Step %s: " % (step))
                print(f" training loss: {lossval[-1]:.4e}")

            step += 1

        print("Step %s: " % (step - 1))
        print(f" training loss: {lossval[-1]:.4e}")
        print('finished')
        lossval = lossval[1:]
        lossval_dbg = lossval_dbg[1:]
        relerr_loss = lossval[-1]
        return lossval, lossval_dbg, relerr_loss

    except KeyboardInterrupt:
        print('Interrupt')
        print('steps = ', step)
        lossval = lossval[1:]
        lossval_dbg = lossval_dbg[1:]
        relerr_loss = lossval[-1]
        return lossval, lossval_dbg, relerr_loss

'''-------------------------Train-------------------------'''
def generate_data(num_omega, num_b, num_if):
    xb = get_boundary_points(num_b)
    xo = get_omega_points(num_omega)
    xif = get_interface_points(num_if)

    xo_tr = add_dimension(xo)
    xb_tr = add_dimension(xb)
    xifp_tr = add_dimension_positive(xif)
    xifn_tr = add_dimension_negetive(xif)
    fo_tr = f_x(xo)
    fb_tr = g_x(xb)
    nor = get_normal_interface(xif)

    len_sum = num_omega + 4 * num_b + 2*num_if
    NL = [len_sum, num_omega, 4 * num_b, num_if, num_if]
    NL_sqrt = np.sqrt(NL)

    xo_tr = torch.tensor(xo_tr, requires_grad=True).double().to(device)
    xb_tr = torch.tensor(xb_tr, requires_grad=True).double().to(device)
    xifp_tr = torch.tensor(xifp_tr, requires_grad=True).double().to(device)
    xifn_tr = torch.tensor(xifn_tr, requires_grad=True).double().to(device)
    fo_tr = torch.tensor(fo_tr).double().to(device)
    fb_tr = torch.tensor(fb_tr).double().to(device)
    nor = torch.tensor(nor).double().to(device)

    return xo_tr, fo_tr, xb_tr, fb_tr, xifp_tr, xifn_tr, nor, NL, NL_sqrt

# Essential namedtuples in the model
DataInput = namedtuple( "DataInput" , [ "X_o" , "F_o", "X_b" , "F_b" , "X_ifp", "X_ifn", "nor", "NL" , "NL_sqrt"] )
LM_Setup  = namedtuple( "LM_Setup" , [ 'p_vec_o' , 'dp_o' , 'L_o' , 'J_o' , 'mu0' , 'criterion' ] )

# create names for storages
fname = 'test'
char_id = 'a'

# Network size
n_input = 3
n_hidden = 30
n_output = 1
n_depth = 2  # only used in deep NN
mu_div = 3.
mu_mul = 2.

# number of training and test data points
c_addpt = 1.
num_omega = 400
num_b = 20 #注意此处设置的为每个边取点数量
num_if = 80

# storages for errors, time instants, and IRK stages
relerr_loss = []
for char in char_id:
    # file name
    fname_char = fname + char

    torch.cuda.empty_cache()  # 清理变量

    # NN structure
    if n_depth == 1:  # Shallow NN
        model = NeuralNet_Shallow(n_input, n_hidden, n_output).double().to(device)
    else:  # Deep NN
        model = NeuralNet_Deep(n_input, n_hidden, n_output, n_depth).double().to(device)

    # use Pytorch and functorch
    func_model, func_params = make_functional(model)  # 获取model及其参数

    xo_tr, fo_tr, xb_tr, fb_tr, xifp_tr, xifn_tr, nor, NL_tr, NL_sqrt_tr = generate_data(num_omega, num_b, num_if)
    tr_input = DataInput(X_o=xo_tr, F_o=fo_tr, X_b=xb_tr, F_b=fb_tr, X_ifp=xifp_tr, X_ifn=xifn_tr, nor=nor, NL=NL_tr, NL_sqrt=NL_sqrt_tr)

    # initialization of LM
    p_vec_old, dp_old, L_old, J_old = generate_initial_LM(func_params, NL_tr[1], NL_tr[2], NL_tr[3], NL_tr[4])  # 初始化LM算法
    print(f"No. of parameters = {len(p_vec_old)}")

    # LM_setup
    mu = 10 ** (8)  # 初始mu值设置大一点，有利于快速下降
    criterion = True
    LM_setup = LM_Setup(p_vec_o=p_vec_old, dp_o=dp_old, L_o=L_old, J_o=J_old, mu0=mu, criterion=criterion)  # 初始化参数导入

    # allocate loss
    lossval = []  # 总损失函数平均值
    lossval_dbg = []  # 各部分损失函数平均值
    lossval.append(1.)
    lossval_dbg.append([1., 1., 1.])

    # start the timer
    cnt_start = time.time()

    # train the model by LM optimizer
    lossval, lossval_dbg, relerr_loss_char = train_PINNs_LM(func_params, LM_setup, tr_input, lossval, lossval_dbg)
    relerr_loss.append(relerr_loss_char)

    end_start = time.time()
    total_T = str(datetime.timedelta(seconds=end_start - cnt_start))
    print(f"total time : {total_T}")

    print('ok')


# plot evolution of loss
N_loss = len(lossval)
lossval        = np.array(lossval).reshape(N_loss,1)
epochcol = np.linspace(1, N_loss, N_loss).reshape(N_loss,1)

plt.figure(figsize = (5,5))

plt.semilogy(epochcol, lossval)
plt.title('loss')
plt.xlabel('epoch')
plt.show()

'''-------------------------Error-------------------------'''
point_r = get_omega_points(10000)
point_p = add_dimension(point_r)
pred_u = func_model(func_params, point_p).cpu().detach().numpy().flatten()
pred_u = torch.tensor(pred_u).double()
pred_u = torch.unsqueeze(pred_u, 1)
loss_l2 = torch.sqrt(torch.mean(torch.pow(pred_u - u_x(point_r),2)))
loss_rel_l2 = loss_l2 / torch.sqrt(torch.mean(torch.pow(u_x(point_r), 2)))
loss_inf = torch.max(torch.abs(pred_u - u_x(point_r)))
loss_rel_inf = loss_inf / torch.max(u_x(point_r))
print('l2相对误差:', loss_rel_l2.item())
print('无穷范数相对误差:', loss_rel_inf.item())

'''-------------------------Plot-------------------------'''
with torch.no_grad():
    x1 = torch.linspace(-1, 1, 2001)
    x2 = torch.linspace(-1, 1, 2001)
    X, Y = torch.meshgrid(x1, x2)
    Z = torch.cat((Y.flatten()[:, None], Y.T.flatten()[:, None]), dim=1)
    ur = u_x(Z)
    dataz = add_dimension(Z)
    pred = func_model(func_params, dataz).cpu().detach().numpy().flatten()
    pred = torch.tensor(pred).double()
    pred = torch.unsqueeze(pred, 1)

plt.figure(1)
pred = pred.reshape(2001, 2001)
ur = ur.reshape(2001, 2001)
h = plt.imshow( torch.abs(pred - ur) , interpolation='nearest', cmap='coolwarm',
                extent=[-1, 1, -1, 1],
                origin='lower', aspect='auto')
plt.title('Error distribution')
plt.xlabel('x')
plt.ylabel('y')
plt.colorbar(h)
plt.savefig('error.jpg', bbox_inches='tight', dpi=600)

plt.figure(2)
h = plt.imshow( pred , interpolation='nearest', cmap='coolwarm',
                extent=[-1, 1, -1, 1],
                origin='lower', aspect='auto')
plt.title('Approximate solution')
plt.xlabel('x')
plt.ylabel('y')
plt.colorbar(h)
plt.savefig('pred_u.jpg', bbox_inches='tight', dpi=600)


