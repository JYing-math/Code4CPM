import matplotlib.pyplot as plt
import torch
import numpy as np

def get_theta(x):
    x1 = x[:, 0]
    x2 = x[:, 1]
    theta = torch.atan2(x2, x1)
    return theta.double()


def get_omega_points(num):
    x1 = (2. * torch.rand(4000, 1) - 1.).double()
    x2 = (2. * torch.rand(4000, 1) - 1.).double()
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


def get_boundary_points(num, r = 1.):
    theta = (torch.rand(num, 1) * torch.pi * 2).double()
    x1 = r * torch.cos(theta)
    x2 = r * torch.sin(theta)
    xb = torch.cat((x1, x2), dim=1)
    r0 = 0.5 + torch.sin(5 * theta) / 7
    xb = 2 * xb * r0
    return xb


def get_interface_points(num, r = 0.5):
    theta = torch.rand(num, 1) * torch.pi * 2
    x1 = r*torch.cos(theta)
    x2 = r*torch.sin(theta)
    xif = torch.cat((x1, x2), dim=1)
    r0 = 0.5 + torch.sin(5 * theta) / 7
    xif = 2 * xif * r0
    return xif

def get_omega_p_points(x):
    m = mark(x)
    index = m > 0
    xo1 = x[index, :]
    return xo1

def get_omega_n_points(x):
    m = mark(x)
    index = m < 0
    xo2 = x[index, :]
    return xo2

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


def up_x(x):
    r = torch.sqrt(torch.sum(torch.pow(x, 2), 1))[:, None]
    theta = get_theta(x)[:, None]
    r0 = 0.5 + torch.sin(5 * theta) / 7
    ux = (r / 2 / r0) ** 3 + (1 / 1000 - 1) * 0.5 ** 3
    return ux


def un_x(x):
    r = torch.sqrt(torch.sum(torch.pow(x, 2), 1))[:, None]
    theta = get_theta(x)[:, None]
    r0 = 0.5 + torch.sin(5 * theta) / 7
    ux = (r / 2 / r0)**3 / 1000
    return ux

def fp_x(x):
    h = 1e-4
    x1 = x[:, 0][:, None]
    x2 = x[:, 1][:, None]
    xl = torch.cat((x1 - h, x2), 1)
    xr = torch.cat((x1 + h, x2), 1)
    xt = torch.cat((x1, x2 + h), 1)
    xb = torch.cat((x1, x2 - h), 1)
    f = (up_x(xl) + up_x(xr) + up_x(xt) + up_x(xb) - 4 * up_x(x)) / (h ** 2)
    return f


def fn_x(x):
    h = 1e-4
    x1 = x[:, 0][:, None]
    x2 = x[:, 1][:, None]
    xl = torch.cat((x1 - h, x2), 1)
    xr = torch.cat((x1 + h, x2), 1)
    xt = torch.cat((x1, x2 + h), 1)
    xb = torch.cat((x1, x2 - h), 1)
    f = (un_x(xl) + un_x(xr) + un_x(xt) + un_x(xb) - 4 * un_x(x)) / (h ** 2)
    return f

def dphi_dx(x):
    r = torch.sqrt(torch.sum(torch.pow(x, 2), 1))
    r = torch.unsqueeze(r, 1)
    theta = get_theta(x)[:, None]
    phix = torch.cos(theta) + 5 * torch.cos(5 * theta) * torch.sin(theta) / r / 7
    return phix

def dphi_dy(x):
    r = torch.sqrt(torch.sum(torch.pow(x, 2), 1))
    r = torch.unsqueeze(r, 1)
    theta = get_theta(x)[:, None]
    phiy = torch.sin(theta) - 5 * torch.cos(5 * theta) * torch.cos(theta) / r / 7
    return phiy

def d2phi_dx2(x):
    r = torch.sqrt(torch.sum(torch.pow(x, 2), 1))
    r = torch.unsqueeze(r, 1)
    theta = get_theta(x)[:, None]
    phix2 = torch.sin(theta) * torch.sin(theta) / r + 5 / 7 / r * torch.sin(theta) * 5 * torch.sin(
        5 * theta) * torch.sin(theta) / r - 5 / 7 * torch.cos(5 * theta) / r * torch.cos(theta) * torch.sin(
        theta) / r - 5 / 7 * torch.cos(5 * theta) * torch.sin(theta) * torch.cos(theta) / r / r
    return phix2

def d2phi_dy2(x):
    r = torch.sqrt(torch.sum(torch.pow(x, 2), 1))
    r = torch.unsqueeze(r, 1)
    theta = get_theta(x)[:, None]
    phiy2 = torch.cos(theta) * torch.cos(theta) / r + 5 / 7 / r * torch.cos(theta) * 5 * torch.sin(
        5 * theta) * torch.cos(theta) / r + 5 / 7 * torch.cos(5 * theta) / r * torch.sin(theta) * torch.cos(
        theta) / r + 5 / 7 * torch.cos(5 * theta) * torch.cos(theta) * torch.sin(theta) / r / r
    return phiy2

def d2phi_dxy(x):
    r = torch.sqrt(torch.sum(torch.pow(x, 2), 1))
    r = torch.unsqueeze(r, 1)
    theta = get_theta(x)[:, None]
    phixy = - torch.sin(theta) * torch.cos(theta) / r - 5 / 7 / r * torch.sin(theta) * 5 * torch.sin(
        5 * theta) * torch.cos(theta) / r + 5 / 7 * torch.cos(5 * theta) / r * torch.cos(theta) * torch.cos(
        theta) / r - 5 / 7 * torch.cos(5 * theta) * torch.cos(theta) * torch.sin(theta) / r / r
    return phixy

def get_normal_interface(x):
    r = torch.sqrt(torch.sum(torch.pow(x, 2), 1))
    r = torch.unsqueeze(r, 1)
    theta = get_theta(x)[:, None]
    n1 = torch.cos(theta) + 5 * torch.cos(5*theta) * torch.sin(theta) / r / 7
    n2 = torch.sin(theta) - 5 * torch.cos(5*theta) * torch.cos(theta) / r / 7
    e1 = n1 / torch.sqrt(n1 * n1 + n2 * n2)
    e2 = n2 / torch.sqrt(n1 * n1 + n2 * n2)
    n = torch.cat((e1, e2), dim=1)
    return n

def psi_x(x, n):
    h = 1e-6
    xn = x + h*n
    psi = (up_x(xn) - up_x(x))/h - (un_x(xn) - un_x(x))/h
    return psi

def generate_data(num_omega, num_b, num_if, device):
    xo = get_omega_points(num_omega)
    xop = get_omega_p_points(xo)
    xon = get_omega_n_points(xo)
    xb = get_boundary_points(num_b)
    xif = get_interface_points(num_if)
    xop_tr = add_dimension(xop)
    xon_tr = add_dimension(xon)
    xb_tr = add_dimension(xb)
    xif_tr = add_dimension(xif)
    fop_tr = fp_x(xop)
    fon_tr = fn_x(xon)
    fb_tr = up_x(xb)
    nor = get_normal_interface(xif)
    fif_tr = psi_x(xif, nor)
    phix_op = dphi_dx(xop)
    phiy_op = dphi_dy(xop)
    phix2_op = d2phi_dx2(xop)
    phiy2_op = d2phi_dy2(xop)
    phix_on = dphi_dx(xon)
    phiy_on = dphi_dy(xon)
    phix2_on = d2phi_dx2(xon)
    phiy2_on = d2phi_dy2(xon)
    phix_if = dphi_dx(xif)
    phiy_if = dphi_dy(xif)
    len_xop = len(xop)
    len_xon = len(xon)
    len_sum = len_xon + len_xop + num_b + num_if
    NL = [len_sum, len_xop, len_xon, num_b, num_if]
    NL_sqrt = np.sqrt(NL)
    xop_tr = torch.tensor(xop_tr, requires_grad=True).double().to(device)
    xon_tr = torch.tensor(xon_tr, requires_grad=True).double().to(device)
    xb_tr = torch.tensor(xb_tr, requires_grad=True).double().to(device)
    xif_tr = torch.tensor(xif_tr, requires_grad=True).double().to(device)
    fop_tr = torch.tensor(fop_tr).double().to(device)
    fon_tr = torch.tensor(fon_tr).double().to(device)
    fb_tr = torch.tensor(fb_tr).double().to(device)
    nor = torch.tensor(nor).double().to(device)
    fif_tr = torch.tensor(fif_tr).double().to(device)
    phix_op = torch.tensor(phix_op).double().to(device)
    phiy_op = torch.tensor(phiy_op).double().to(device)
    phix2_op = torch.tensor(phix2_op).double().to(device)
    phiy2_op = torch.tensor(phiy2_op).double().to(device)
    phix_on = torch.tensor(phix_on).double().to(device)
    phiy_on = torch.tensor(phiy_on).double().to(device)
    phix2_on = torch.tensor(phix2_on).double().to(device)
    phiy2_on = torch.tensor(phiy2_on).double().to(device)
    phix_if = torch.tensor(phix_if).double().to(device)
    phiy_if = torch.tensor(phiy_if).double().to(device)
    data_xop = (xop_tr, fop_tr, phix_op, phiy_op, phix2_op, phiy2_op)
    data_xon = (xon_tr, fon_tr, phix_on, phiy_on, phix2_on, phiy2_on)
    data_xb = (xb_tr, fb_tr)
    data_xif = (xif_tr, nor, fif_tr, phix_if, phiy_if)
    return data_xop, data_xon, data_xb, data_xif, NL, NL_sqrt