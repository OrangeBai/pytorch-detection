import numpy as np
import matplotlib.pyplot as plt

# no_random = np.load(r'F:\Code\Computer Science\fast_ensemble\plots\static_non.npy')
# ran = np.load(r'F:\Code\Computer Science\fast_ensemble\plots\static_random.npy')
normal = np.load(r'F:\Code\Computer Science\fast_ensemble\plots\data\record2.npy')

xx = range(1, 121)
# plt.plot(xx, no_random[1, :], label='without noise', color='r')
# plt.plot(xx, ran[1, :], label='with noise', color='g')
plt.plot(xx, normal[3, :], label='Natural ACC', color='r')

# plt.plot(xx, no_random[3, :], linestyle='-.', color='r')
# plt.plot(xx, ran[3, :], linestyle='-.', color='g')
plt.plot(xx, normal[1, :], label='FGSM ACC', linestyle='-.', color='r')

# plt.plot(xx, no_random[7, :], linestyle=':', color='r')
# plt.plot(xx, ran[7, :], linestyle=':', color='g')
plt.plot(xx, normal[7, :], label='PGD ACC', linestyle=':', color='r')

plt.legend()
plt.savefig('plot4.png')
print(1)
