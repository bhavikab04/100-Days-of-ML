import numpy as np

np.random.seed(0)
x = np.random.uniform(-10, 10, size = 100)
y_truth = 3*x + 7 + np.random.normal(0,1, size = 100)

w = np.random.randn()
b = np.random.randn()
alpha = 1e-2

itr = 1000

for i in range(itr):
    y_pred = w*x + b
    dl_dypred = y_pred - y_truth
    dypred_dw = x
    dypred_db = 1
    dL_dw = np.mean(dl_dypred*dypred_dw)
    dL_db = np.mean(dl_dypred*dypred_db)
    w = w - alpha*(dL_dw)
    b = b - alpha*(dL_db)
    if(i%100 == 0):
        L = np.mean((y_pred-y_truth)** 2)/2
        print(f"iteration: {i}, loss: {L:.6f}, w = {w:.4f}, b = {b:.4f}")



