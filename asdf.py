
import numpy as np
import matplotlib.pyplot as pp

n_true = 30  # number of pixels we want to compute
n_boundary = 15  # number of pixels to extend the image in all directions
c = 4
d = 4

# First compute g and lapg including boundary extenstion
n = n_true + n_boundary * 2
x = np.arange(-n // 2, n // 2) /1
y = np.arange(-n // 2, n // 2) / 10

def gaus2d(x=0, y=0, mx=0, my=0, sx=1, sy=1):
    return 1. / (2. * np.pi * sx * sy) * np.exp(-((x - mx)**2. / (2. * sx**2.) + (y - my)**2. / (2. * sy**2.)))

xx, yy = np.meshgrid(x, y) # get 2D variables instead of 1D
g = gaus2d(xx, yy)

kx = 2 * np.pi * np.fft.fftfreq(n)
ky = 2 * np.pi * np.fft.fftfreq(n)
lapg = np.real(np.fft.ifft2(np.fft.fft2(g) * (-kx[None, :] ** 2 - ky[:, None] ** 2)))

# Now crop the two images to our desired size
x = x[n_boundary:-n_boundary]
y = y[n_boundary:-n_boundary]
g = g[n_boundary:-n_boundary, n_boundary:-n_boundary]
lapg = lapg[n_boundary:-n_boundary, n_boundary:-n_boundary]

# Display
fig = pp.figure()
ax = fig.add_subplot(121, projection='3d')
ax.plot_surface(x[None, :], y[:, None], g)
# ax.set_zlim(0, 800)
ax = fig.add_subplot(122, projection='3d')
ax.plot_surface(x[None, :], y[:, None], lapg)
# ax.set_zlim(0, 800)
pp.show()
print(1)