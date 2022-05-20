from core.utils import *
from settings.cifar_settings import *
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as colors


a4 = []
aa = torch.load('/home/orange/Main/Experiment/mnist/mnist_dnn_1/rate')
a4 = []
for key, val in aa.items():
    if 'noise_4' in key:
        a4.append(np.array(val).mean())
print(1)

import matplotlib.pyplot as plt
large = 22
med = 16
small = 12
params = {'axes.titlesize': large,
          'legend.fontsize': med,
          'figure.figsize': (16, 9),
          'axes.labelsize': med,
          # 'xtick.labelsize': med,
          # 'ytick.labelsize': med,
          'figure.titlesize': large}
plt.rcParams.update(params)
plt.style.use('seaborn-dark')
sns.set_theme(style="darkgrid")
figure, ax = plt.subplots()
xx = np.array([64, 128, 256, 512, 1024, 1536, 2048])
ax.plot(xx, 1-np.array(a4))
ax.plot(xx, 1-np.array(a8))
ax.plot(xx, 1-np.array(a12))
ax.plot(xx, 1-np.array(a16))
ax.plot(xx, 1-np.array(a24))
ax.plot(xx, 1-np.array(a32))
ax.plot(xx, 1-np.array(a64))


ax.legend(['4', '8', '12', '16', '24', '32', '64'])
ax.set_xlabel('# points')
ax.set_ylabel(r'# float neurons / # neurons')
plt.savefig()