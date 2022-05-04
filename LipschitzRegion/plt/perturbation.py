from core.utils import *
from settings.cifar_settings import *
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as colors


def tableau_color_list():
    color_dict = colors.TABLEAU_COLORS
    return list(color_dict.values())


args = set_up_training(False)
a = load_result(args.model_dir, '0.2_portion.npy').mean(axis=1)
b = load_result(args.model_dir, '0.5_portion.npy').mean(axis=1)
cc = load_result(args.model_dir, '1_portion.npy').mean(axis=1)

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
c = tableau_color_list()
figure, ax = plt.subplots()

xx = [100 * i for i in range(31)]
ax.plot(xx, a.mean(axis=0), color=c[0])
ax.fill_between(xx, a.mean(axis=0) - a.var(axis=0), a.mean(axis=0) + a.var(axis=0), color=c[0], alpha=0.3)

ax.plot(xx, b.mean(axis=0), color=c[1])
ax.fill_between(xx, b.mean(axis=0) - b.var(axis=0), b.mean(axis=0) + b.var(axis=0), color=c[1], alpha=0.3)

ax.plot(xx, cc.mean(axis=0), color=c[2])
ax.fill_between(xx, cc.mean(axis=0) - cc.var(axis=0), cc.mean(axis=0) + cc.var(axis=0), color=c[2], alpha=0.3)


ax.legend(['$\epsilon$=0.2', '$\epsilon$=0.5', '$\epsilon$=1'])
ax.set_xlabel('# points')
ax.set_ylabel(r'# float neurons / # neurons')
plt.savefig()

print(0)
