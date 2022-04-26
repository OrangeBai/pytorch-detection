import numpy as np
import matplotlib.pyplot as plt


def plot_erb(plt, data, name, color):
    x = np.array(range(1, 101))
    y = data.mean(axis=0)
    e = data.std(axis=0)
    plt.plot(x, y, label=name, color=color)
    plt.scatter(x, y+e, color=color, s=5)
    plt.scatter(x, y-e, color=color, s=5)

    return


pgd_adv = np.load('pgd.npy')
pgd_test = np.load('pgd_test.npy')
fgsm_adv = np.load('fgsm.npy')
fgsm_test = np.load('fgsm_test.npy')

natural = np.load('natural.npy')
natural_test = np.load('natural_test.npy')

plot_erb(plt, pgd_test, 'PGD', 'red')
plot_erb(plt, fgsm_test, 'FGSM', 'blue')
plot_erb(plt, natural_test, 'Natural', 'green')

plt.legend()
plt.show()
print(1)
