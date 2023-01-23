def update(mean1, var1, mean2, var2):
    new_mean = float(var2 * mean1 + var1 * mean2) / (var1 + var2)  # 新均值位于mean1和mean2之间
    new_var = 1./(1./var1 + 1./var2)  # 新方差要比var1和var2均小，高斯曲线更加高耸，陡峭
    return [new_mean, new_var]

def predict(mean1, var1, mean2, var2):
    new_mean = mean1 + mean2
    new_var = var1 + var2
    return [new_mean, new_var]

measurements = [5., 6., 7., 9., 10.]
measurement_sig = 4.
motion = [1., 1., 2., 1., 1.]
motion_sig = 2.

# 状态初始值分布
mu = 0.
sig = 10000.

for n in range(len(measurements)):
    [mu, sig] = update(mu, sig, measurements[n], measurement_sig)
    print('update: ', [mu, sig])
    [mu, sig] = predict(mu, sig, motion[n], motion_sig)
    print('predict: ', [mu, sig])
