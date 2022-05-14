import numpy as np


def ff_dis(inmap, padding=1):
    xn, yn = inmap.shape[-2] + padding * 2, inmap.shape[-1] + padding * 2
    inmap = np.pad(inmap, ((0, 0), (0, 0), (padding, padding), (padding, padding)), 'constant', constant_values=(0, 0))
    xk = (2 * np.pi / xn) * np.arange(-xn // 2, xn // 2)
    yk = (2 * np.pi / yn) * np.arange(-yn // 2, yn // 2)

    kappa_x = np.fft.fftshift(xk)
    kappa_y = np.fft.fftshift(yk)

    grid_map = (-kappa_x[None, None, None, :]**2 - kappa_y[None, None, :, None] ** 2)
    grid_map = np.repeat(np.repeat(grid_map, inmap.shape[1], axis=1), inmap.shape[0], axis=0)
    df = np.real(np.fft.ifft2(np.fft.fft2(inmap, axes=[-2, -1]) * grid_map))
    df = df[:, :, padding: -padding, padding: -padding]

    return df


if __name__ == '__main__':
    ff_dis(np.random.randn(64, 64, 32))
