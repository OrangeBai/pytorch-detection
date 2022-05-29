import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
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


a = np.linspace(1.5, 4.5, 300)
b = np.ones(300)
b[30:40] = np.linspace(1, 0.5, 10)
b[40:60] = 0.5
b[60:70] = np.linspace(0.5, 1, 10)

b[150:160] = 1
b[240:250] = np.linspace(1, 0.8, 10)
b[250:] = 0.8


c = np.ones(300)
c[0:40] = 0.5
c[40:140] = 0.5
c[130:140] = np.linspace(0.5, 1, 10)

c[140:160] = 1
c[160:170] = np.linspace(1, 0.8, 10)
c[170:] = 0.8



plt.plot(a, b)
plt.plot(a, c)
plt.savefig('./sup')
print(1)