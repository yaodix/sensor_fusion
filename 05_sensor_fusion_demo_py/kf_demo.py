# 参考： https://adamshan.blog.csdn.net/article/details/78248421
# 卡尔曼滤波示例

import numpy as np
# %matplotlib inline
import matplotlib.pyplot as plt
from scipy.stats import norm


# 初始状态均设为0
x = np.matrix([[0.0, 0.0, 0.0, 0.0]]).T

################################ 初始化R、P、Q ################################
# 因为初始状态设为0，所以初始P矩阵“误差”设置要大。如果初始状态用首次测量值初始化，则设置较小误差
P = np.diag([1000.0, 1000.0, 1000.0, 1000.0]) 
print(P, P.shape)

dt = 0.1 # Time Step between Filter Steps
F = np.matrix([[1.0, 0.0, dt, 0.0],
              [0.0, 1.0, 0.0, dt],
              [0.0, 0.0, 1.0, 0.0],
              [0.0, 0.0, 0.0, 1.0]])

H = np.matrix([[0.0, 0.0, 1.0, 0.0],
              [0.0, 0.0, 0.0, 1.0]])

# ra = 10.0**2
# R = np.matrix([[ra, 0.0],
#               [0.0, ra]])

ra = 0.09
R = np.matrix([[ra, 0.0],
              [0.0, ra]])
print(R, R.shape)

sv = 0.5
G = np.matrix([[0.5*dt**2],
               [0.5*dt**2],
               [dt],
               [dt]])
Q = G*G.T*sv**2
I = np.eye(4)

# 产生随机数
m = 200 # Measurements
vx= 20 # in X
vy= 10 # in Y

mx = np.array(vx+np.random.randn(m))
my = np.array(vy+np.random.randn(m))
measurements = np.vstack((mx,my))


# print(measurements.shape)
# print('Standard Deviation of Acceleration Measurements=%.2f' % np.std(mx))
# print('You assumed %.2f in R.' % R[0,0])

# 绘制测量数据
# fig = plt.figure(figsize=(16,5))
# plt.step(range(m),mx, label='$\dot x$')
# plt.step(range(m),my, label='$\dot y$')
# plt.ylabel(r'Velocity $m/s$')
# plt.title('Measurements')
# plt.legend(loc='best',prop={'size':18})
# plt.show()

# 一些过程值，用于显示结果
xt = []
yt = []
dxt= []
dyt= []
Zx = []
Zy = []
Px = []
Py = []
Pdx= []
Pdy= []
Rdx= []
Rdy= []
Kx = []
Ky = []
Kdx= []
Kdy= []

def savestates(x, Z, P, R, K):
    xt.append(float(x[0]))
    yt.append(float(x[1]))
    dxt.append(float(x[2]))
    dyt.append(float(x[3]))
    Zx.append(float(Z[0]))
    Zy.append(float(Z[1]))
    Px.append(float(P[0,0]))
    Py.append(float(P[1,1]))
    Pdx.append(float(P[2,2]))
    Pdy.append(float(P[3,3]))
    Rdx.append(float(R[0,0]))
    Rdy.append(float(R[1,1]))
    Kx.append(float(K[0,0]))
    Ky.append(float(K[1,0]))
    Kdx.append(float(K[2,0]))
    Kdy.append(float(K[3,0]))    


# 关于速度的估计结果
def plot_x():
    fig = plt.figure(figsize=(16,9))
    plt.step(range(len(measurements[0])),dxt, label='$estimateVx$')
    plt.step(range(len(measurements[0])),dyt, label='$estimateVy$')
    
    plt.step(range(len(measurements[0])),measurements[0], label='$measurementVx$')
    plt.step(range(len(measurements[0])),measurements[1], label='$measurementVy$')

    plt.axhline(vx, color='#999999', label='$trueVx$')
    plt.axhline(vy, color='#999999', label='$trueVy$')

    plt.xlabel('Filter Step')
    plt.title('Estimate (Elements from State Vector $x$)')
    plt.legend(loc='best',prop={'size':11})
    plt.ylim([0, 30])
    plt.ylabel('Velocity')
    plt.show()

# 位置的估计结果
def plot_xy():
    fig = plt.figure(figsize=(16,16))
    plt.scatter(xt,yt, s=20, label='State', c='k')
    plt.scatter(xt[0],yt[0], s=100, label='Start', c='g')
    plt.scatter(xt[-1],yt[-1], s=100, label='Goal', c='r')

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Position')
    plt.legend(loc='best')
    plt.axis('equal')
    plt.show()

if __name__ == "__main__":
    # 卡尔曼滤波
    for n in range(len(measurements[0])):
        # Time Update (Prediction)
        # ========================
        # Project the state ahead
        x = F*x
        
        # Project the error covariance ahead
        P = F*P*F.T + Q
        
        # Measurement Update (Correction)
        # ===============================
        # Compute the Kalman Gain
        S = H*P*H.T + R
        K = (P*H.T) * np.linalg.pinv(S)
        
        # Update the estimate via z
        Z = measurements[:,n].reshape(2,1)
        y = Z - (H*x)                            # Innovation or Residual
        x = x + (K*y)
        
        # Update the error covariance
        P = (I - (K*H))*P
        
        # Save states (for Plotting)
        savestates(x, Z, P, R, K)
    
    plot_x()
    plot_xy()
