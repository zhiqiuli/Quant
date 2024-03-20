'''
https://github.com/cantaro86/Financial-Models-Numerical-Methods/blob/master/2.1%20Black-Scholes%20PDE%20and%20sparse%20matrices.ipynb
'''
import numpy as np

from matplotlib import pyplot as plt
from matplotlib import cm

from scipy import sparse
from scipy.sparse.linalg import splu
from scipy.sparse.linalg import spsolve

r = 0.05
sig = 0.2
S0 = 100
X0 = np.log(S0)
K = 105
Texpir = 1

Nspace = 3000  # M space steps
Ntime = 2000  # N time steps

S_max = 3 * float(K)
S_min = float(K) / 3

x_max = np.log(S_max)  # A2
x_min = np.log(S_min)  # A1

x, dx = np.linspace(x_min, x_max, Nspace, retstep=True)  # space discretization
T, dt = np.linspace(0, Texpir, Ntime, retstep=True)  # time discretization
Payoff = np.maximum(np.exp(x) - K, 0)  # Call payoff


###
### implicit scheme
###

V = np.zeros((Nspace, Ntime))  # grid initialization
offset = np.zeros(Nspace - 2)  # vector to be used for the boundary terms

V[: , -1] = Payoff  # terminal conditions
# boundary condition, note that T[::-1] is the same as time-to-maturity or tau
V[-1, : ] = np.exp(x_max) - K * np.exp(-r * T[::-1])
V[0 , : ] = 0  # boundary condition

# construction of the tri-diagonal matrix D
sig2 = sig * sig
dxx  = dx * dx

a =  (dt / 2) * ((r - 0.5 * sig2) / dx - sig2 / dxx)
b = 1 + dt * (sig2 / dxx + r)
c = -(dt / 2) * ((r - 0.5 * sig2) / dx + sig2 / dxx)

D = sparse.diags([a, b, c], [-1, 0, 1], shape=(Nspace - 2, Nspace - 2)).tocsc()

# Backward iteration
for i in range(Ntime - 2, -1, -1):
    offset[0]  = a * V[0, i]
    offset[-1] = c * V[-1, i]
    V[1:-1, i] = spsolve(D, (V[1:-1, i + 1] - offset))

option_price = np.interp(X0, x, V[:, 0])
print(f"European call option price (npde w/ implicit scheme): {option_price}")


S = np.exp(x)
fig = plt.figure(figsize=(15, 6))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122, projection="3d")

ax1.plot(S, Payoff, color="blue", label="Payoff")
ax1.plot(S, V[:, 0], color="red", label="BS curve")
ax1.set_xlim(60, 170)
ax1.set_ylim(0, 50)
ax1.set_xlabel("S")
ax1.set_ylabel("price")
ax1.legend(loc="upper left")
ax1.set_title("BS price at t=0")

X, Y = np.meshgrid(T, S)
ax2.plot_surface(Y, X, V, cmap=cm.ocean)
ax2.set_title("BS price surface")
ax2.set_xlabel("S")
ax2.set_ylabel("t")
ax2.set_zlabel("V")
ax2.view_init(30, -100)  # this function rotates the 3d plot
plt.show()

###
### CN
###

V = np.zeros((Nspace, Ntime))  # grid initialization
offset = np.zeros(Nspace - 2)  # vector to be used for the boundary terms

V[: , -1] = Payoff  # terminal conditions
# boundary condition, note that T[::-1] is the same as time-to-maturity or tau
V[-1, : ] = np.exp(x_max) - K * np.exp(-r * T[::-1])
V[0 , : ] = 0  # boundary condition

sig2 = sig * sig
dx2  = dx * dx

alpha = 0.25 * dt / dx * (r - 0.5 * sig2)
beta  = 0.25 * sig2 * dt / dx2
theta = 0.5 * dt * r

a_0 = - alpha + beta
a_1 = 1 - 2 * beta - theta
a_2 =   alpha + beta

d_0 =   alpha - beta
d_1 = 1 + 2 * beta + theta
d_2 = - alpha - beta

A = sparse.diags([a_0, a_1, a_2], [-1, 0, 1], shape=(Nspace - 2, Nspace - 2)).tocsc()

D = sparse.diags([d_0, d_1, d_2], [-1, 0, 1], shape=(Nspace - 2, Nspace - 2)).tocsc()

# Backward iteration
for i in range(Ntime - 2, -1, -1):
    offset[0]  = a_0 * V[0 , i] - d_0 * V[0 , i]
    offset[-1] = a_2 * V[-1, i] - d_2 * V[-1, i]
    LHS = A * V[1:-1,i+1] + offset
    V[1:-1, i] = spsolve(D, LHS)

option_price = np.interp(X0, x, V[:, 0])
print(f"European call option price (npde w/ CN): {option_price}")

S = np.exp(x)
fig = plt.figure(figsize=(15, 6))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122, projection="3d")

ax1.plot(S, Payoff, color="blue", label="Payoff")
ax1.plot(S, V[:, 0], color="red", label="BS curve")
ax1.set_xlim(60, 170)
ax1.set_ylim(0, 50)
ax1.set_xlabel("S")
ax1.set_ylabel("price")
ax1.legend(loc="upper left")
ax1.set_title("BS price at t=0")

X, Y = np.meshgrid(T, S)
ax2.plot_surface(Y, X, V, cmap=cm.ocean)
ax2.set_title("BS price surface")
ax2.set_xlabel("S")
ax2.set_ylabel("t")
ax2.set_zlabel("V")
ax2.view_init(30, -100)  # this function rotates the 3d plot
plt.show()
