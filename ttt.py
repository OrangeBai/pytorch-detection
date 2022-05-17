import numpy as np

a = np.random.randn(5, 5)
b = np.random.randn(5, 5)
ab = np.matmul(a,b)

singluar = np.linalg.svd(ab)
lmd = singluar[2][0]

cc = np.matmul(b, lmd)
output = np.matmul(a, cc)
print(1)