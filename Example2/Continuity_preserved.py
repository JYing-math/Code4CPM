import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import functools


from functorch import make_functional, vmap, grad, jacrev, hessian

from collections import namedtuple, OrderedDict
import datetime
import time
from IPython.display import clear_output

from Generate_data import generate_data
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
tr_iter_max    = 2000                     # max. iteration
ts_input_new   = 1000                       # renew testing points
ls_check       = 10
ls_check0      = ls_check - 1
# number of training points and testing points
N_tsd_final = 250000 #100*N_trd
N_tsg_final = 1000   #10*N_trg
# tolerence for LM
tol_main    = 10**(-14)
tol_machine = 10**(-15)
mu_max      = 10**8
mu_ini      = 10**8
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
        #self.act1 = nn.Sigmoid()
        self.act1 = nn.Tanh()
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
def func_loss_op(func_params, x_op, f_op, phi_x, phi_y, phi_xx, phi_yy):
    def f(x, func_params):
        fx = func_model(func_params, x)
        return fx.squeeze(0).squeeze(0)
        # derivatives of u wrt inputs

    d1u = jacrev(f)(x_op, func_params)
    d2u = jacrev(jacrev(f))(x_op, func_params)

    f_op = f_op[0]
    phi_x = phi_x[0]
    phi_y = phi_y[0]
    phi_xx = phi_xx[0]
    phi_yy = phi_yy[0]
    u_z = d1u[2]
    u_xz = d2u[0][2]
    u_xx = d2u[0][0]
    u_yz = d2u[1][2]
    u_yy = d2u[1][1]
    u_zz = d2u[2][2]
    U_xx = u_xx + u_xz * phi_x + u_xz * phi_x + u_zz * phi_x * phi_x + u_z * phi_xx
    U_yy = u_yy + u_yz * phi_y + u_yz * phi_y + u_zz * phi_y * phi_y + u_z * phi_yy
    loss_op = U_xx + U_yy - f_op
    return loss_op

def func_loss_on(func_params, x_on, f_on, phi_x, phi_y, phi_xx, phi_yy):
    def f(x, func_params):
        fx = func_model(func_params, x)
        return fx.squeeze(0).squeeze(0)
        # derivatives of u wrt inputs

    d1u = jacrev(f)(x_on, func_params)
    d2u = jacrev(jacrev(f))(x_on, func_params)

    f_on = f_on[0]
    phi_x = phi_x[0]
    phi_y = phi_y[0]
    phi_xx = phi_xx[0]
    phi_yy = phi_yy[0]
    u_z = d1u[2]
    u_xz = d2u[0][2]
    u_xx = d2u[0][0]
    u_yz = d2u[1][2]
    u_yy = d2u[1][1]
    u_zz = d2u[2][2]
    U_xx = u_xx - u_xz * phi_x - u_xz * phi_x + u_zz * phi_x * phi_x - u_z * phi_xx
    U_yy = u_yy - u_yz * phi_y - u_yz * phi_y + u_zz * phi_y * phi_y - u_z * phi_yy
    loss_on = u_xx + u_yy - f_on
    return loss_on

def func_loss_b(func_params, x_b, f_b):
    def f(x, func_params):
        fx = func_model(func_params, x)
        return fx.squeeze(0).squeeze(0)
    f_b = f_b[0]
    # function value at the boundary (Dirichlet)
    u = f(x_b, func_params)
    loss_b = u - f_b
    return loss_b

def func_loss_if(func_params, x_if, nor, f_if, phi_x, phi_y):
    def f(x, func_params):
        fx = func_model(func_params, x)
        return fx.squeeze(0).squeeze(0)

    # derivatives of u wrt inputs
    d1u = jacrev(f)(x_if, func_params)
    f_if = f_if[0]
    u_z = d1u[2]
    phi_x = phi_x[0]
    phi_y = phi_y[0]
    nor1 = nor[0]
    nor2 = nor[1]
    loss_if = (phi_x * u_z) * nor1 + (phi_y * u_z) * nor2 - f_if
    return loss_if
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
def generate_initial_LM(func_params, Xo1_len, Xo2_len, Xb_len, Xg_len):
    # data_length
    data_length = Xo1_len + Xo2_len + Xb_len + Xg_len # 输入数据长度和

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
    [Data_op, Data_on, Data_b, Data_if, NL, NL_sqrt] = tr_input #训练参数

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
                Lop = vmap((func_loss_op), (None, 0, 0, 0, 0, 0, 0))(func_params, Data_op[0], Data_op[1], Data_op[2], Data_op[3], Data_op[4], Data_op[5]).flatten().detach()
                Lon = vmap((func_loss_on), (None, 0, 0, 0, 0, 0, 0))(func_params, Data_on[0], Data_on[1], Data_on[2], Data_on[3], Data_on[4], Data_on[5]).flatten().detach()
                Lb = vmap((func_loss_b), (None, 0, 0))(func_params, Data_b[0], Data_b[1]).flatten().detach()
                Lif = vmap((func_loss_if), (None, 0, 0, 0, 0, 0))(func_params, Data_if[0], Data_if[1], Data_if[2], Data_if[3], Data_if[4]).flatten().detach()
                L = torch.cat((Lop / NL_sqrt[1], Lon / NL_sqrt[2], Lb / NL_sqrt[3], Lif / NL_sqrt[4]))
                L = L.reshape(NL[0], 1).detach()
                lsdp_sum = torch.sum(Lop * Lop) / NL[1]
                lsdn_sum = torch.sum(Lon * Lon) / NL[2]
                lsb_sum = torch.sum(Lb * Lb) / NL[3]
                lsif_sum = torch.sum(Lif * Lif) / NL[4]
                loss_dbg_old = [lsdp_sum.item(), lsdn_sum.item(), lsb_sum.item(), lsif_sum.item()]

            loss_old = lossval[-1]
            loss_dbg_old = lossval_dbg[-1]

            ### compute the gradinet of loss function for each point
            with torch.no_grad():
                p_vec = get_p_vec(func_params).detach()  # get p_vec for p_vec_old if neccessary

            if criterion:
                per_sample_grads = vmap(jacrev(func_loss_op), (None, 0, 0, 0, 0, 0, 0))(func_params, Data_op[0], Data_op[1], Data_op[2], Data_op[3], Data_op[4], Data_op[5])
                cnt = 0
                for g in per_sample_grads:
                    g = g.detach()
                    J_op = g.reshape(len(g), -1) if cnt == 0 else torch.hstack([J_op, g.reshape(len(g), -1)])
                    cnt = 1

                per_sample_grads = vmap(jacrev(func_loss_on), (None, 0, 0, 0, 0, 0, 0))(func_params, Data_on[0], Data_on[1], Data_on[2], Data_on[3], Data_on[4], Data_on[5])
                cnt = 0
                for g in per_sample_grads:
                    g = g.detach()
                    J_on = g.reshape(len(g), -1) if cnt == 0 else torch.hstack([J_on, g.reshape(len(g), -1)])
                    cnt = 1

                per_sample_grads = vmap(jacrev(func_loss_b), (None, 0, 0))(func_params, Data_b[0], Data_b[1])
                cnt = 0
                for g in per_sample_grads:
                    g = g.detach()
                    J_b = g.reshape(len(g), -1) if cnt == 0 else torch.hstack([J_b, g.reshape(len(g), -1)])
                    cnt = 1

                per_sample_grads = vmap(jacrev(func_loss_if), (None, 0, 0, 0, 0, 0))(func_params, Data_if[0], Data_if[1], Data_if[2], Data_if[3], Data_if[4])
                cnt = 0
                for g in per_sample_grads:
                    g = g.detach()
                    J_if = g.reshape(len(g), -1) if cnt == 0 else torch.hstack([J_if, g.reshape(len(g), -1)])
                    cnt = 1

                J = torch.cat((J_op / NL_sqrt[1], J_on / NL_sqrt[2], J_b / NL_sqrt[3], J_if / NL_sqrt[4])).detach()
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
            Lop = vmap((func_loss_op), (None, 0, 0, 0, 0, 0, 0))(func_params, Data_op[0], Data_op[1], Data_op[2],
                                                                 Data_op[3], Data_op[4], Data_op[5]).flatten().detach()
            Lon = vmap((func_loss_on), (None, 0, 0, 0, 0, 0, 0))(func_params, Data_on[0], Data_on[1], Data_on[2],
                                                                 Data_on[3], Data_on[4], Data_on[5]).flatten().detach()
            Lb = vmap((func_loss_b), (None, 0, 0))(func_params, Data_b[0], Data_b[1]).flatten().detach()
            Lif = vmap((func_loss_if), (None, 0, 0, 0, 0, 0))(func_params, Data_if[0], Data_if[1], Data_if[2],
                                                              Data_if[3], Data_if[4]).flatten().detach()
            L = torch.cat((Lop / NL_sqrt[1], Lon / NL_sqrt[2], Lb / NL_sqrt[3], Lif / NL_sqrt[4]))
            L = L.reshape(NL[0], 1).detach()
            loss_new = torch.sum(L * L).item()
            lsdp_sum = torch.sum(Lop * Lop) / NL[1]
            lsdn_sum = torch.sum(Lon * Lon) / NL[2]
            lsb_sum = torch.sum(Lb * Lb) / NL[3]
            lsif_sum = torch.sum(Lif * Lif) / NL[4]
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
# Essential namedtuples in the model
DataInput = namedtuple( "DataInput" , ["Data_op", "Data_on", "Data_b", "Data_if", "NL" , "NL_sqrt"] )
LM_Setup  = namedtuple( "LM_Setup" , [ 'p_vec_o' , 'dp_o' , 'L_o' , 'J_o' , 'mu0' , 'criterion' ] )

# create names for storages
fname = 'test'
char_id = 'a'

# Network size
n_input = 3
n_hidden = 30
n_output = 1
n_depth = 3  # only used in deep NN
mu_div = 3.
mu_mul = 2.

# number of training and test data points
c_addpt = 1.
num_omega = 2000
num_b = 200
num_if = 200


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

    data_xop, data_xon, data_xb, data_xif, NL, NL_sqrt = generate_data(num_omega, num_b, num_if, device)
    tr_input = DataInput(Data_op=data_xop, Data_on=data_xon, Data_b=data_xb, Data_if=data_xif, NL=NL, NL_sqrt=NL_sqrt)

    # initialization of LM
    p_vec_old, dp_old, L_old, J_old = generate_initial_LM(func_params, NL[1], NL[2], NL[3], NL[4])  # 初始化LM算法
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

'''-------------------------Plot-------------------------'''

def get_theta(x):
    x1 = x[:, 0]
    x2 = x[:, 1]
    theta = torch.atan2(x2, x1)
    return theta.double()

def get_omega_points(num):
    x1 = (2. * torch.rand(num, 1) - 1.).double()
    x2 = (2. * torch.rand(num, 1) - 1.).double()
    x = torch.cat((x1, x2), dim=1)
    r = torch.sqrt(torch.sum(torch.pow(x, 2), 1))
    index = r <= 1
    xo = x[index, :]
    xo = xo[:num, :]
    theta = get_theta(xo)
    r0 = 0.5 + torch.sin(5 * theta) / 7
    r0 = torch.unsqueeze(r0, 1)
    xo = 2 * xo * r0
    return xo

def add_dimension(x):
    m = mark(x)
    index1 = m >= 0
    index1 = index1.float()
    index1 = torch.unsqueeze(index1, 1)
    lf = torch.abs(level_set_fuction(x) * index1)
    x_3 = torch.cat((x, lf), 1)
    return x_3

def mark(x):
    lf = level_set_fuction(x)
    m = torch.squeeze(lf ,1)
    return m


def level_set_fuction(x):
    r = torch.sqrt(torch.sum(torch.pow(x, 2), 1))[:, None]
    theta = get_theta(x)[:, None]
    lf = r - 0.5 - torch.sin(5*theta)/7
    return lf


def u_x(x):
    m = mark(x)
    index = m
    r = torch.sqrt(torch.sum(torch.pow(x, 2), 1))[:, None]
    theta = get_theta(x)[:, None]
    r0 = 0.5 + torch.sin(5 * theta) / 7
    u1 = (r / 2 / r0)**3 + (1 / 1000 - 1) * 0.5 ** 3
    u2 = (r / 2 / r0)**3 / 1000
    ux = torch.where(index >= 0, u1[:, 0], u2[:, 0])
    return ux


xtest = get_omega_points(40000).to(device)
dataz = add_dimension(xtest)
pred = func_model(func_params, dataz).cpu().detach().numpy().flatten()
real = u_x(xtest).cpu().detach().numpy().flatten()

loss_l2 = np.sqrt(np.mean(np.power(pred - real, 2)))
loss_rel_l2 = loss_l2 / np.sqrt(np.mean(np.power(real, 2)))
loss_inf = np.max(np.abs(pred - real))
loss_rel_inf = loss_inf / np.max(real)
print('l2相对误差:', loss_rel_l2.item())
print('无穷范数相对误差:', loss_rel_inf.item())

fig = plt.figure(figsize=plt.figaspect(0.3))
#===============
#  First subplot
#===============
# set up the axes for the first plot
dataz = dataz.cpu().detach().numpy()
ax = fig.add_subplot(1, 1, 1, projection='3d')
surf = ax.scatter(dataz[:,0:1], dataz[:,1:2], np.abs(pred - real), c=np.abs(pred - real) , cmap='plasma')
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Error distribution')
ax.view_init(elev=30, azim=30)
plt.savefig('pred_u.jpg', bbox_inches='tight', dpi=600)
plt.show()

