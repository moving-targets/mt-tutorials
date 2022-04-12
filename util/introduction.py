import numpy as np
from matplotlib import pyplot as plt


def function_plot(x, y, true_fn=None, pred_fn=None, figsize=(7, 7)):
    plt.figure(figsize=figsize)
    plt.scatter(x, y, color='tab:red', label='measures')
    span = x[1] - x[0]
    space = np.linspace(x[0] - 0.5 * span, x[1] + 0.5 * span)
    if pred_fn is not None:
        plt.scatter(x, pred_fn(x), color='tab:orange', label='predictions')
        plt.plot(space, pred_fn(space), color='tab:orange', label='estimated function', linestyle=':')
    if true_fn is not None:
        plt.plot(space, true_fn(space), linestyle=':', color='tab:blue', label='true function')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(linestyle=':')
    plt.axis('equal')
    plt.legend()
    plt.show()


def output_plot(x, y, p=None, z=None, bound_fn=None, figsize=(7, 7)):
    plt.figure(figsize=figsize)
    y0 = np.linspace(y[0] - 0.02, y[0] + 0.22)
    y1 = x[1] ** (np.log(y0) / np.log(x[0]))
    plt.gca().set_facecolor((0.95, 0.95, 0.95))
    plt.axvspan(y0[0], y0[-1], color='white')
    plt.plot(y0, y1, color='tab:orange', label='ML model bias')
    plt.scatter(y[0], y[1], color='tab:red', label='measured values', zorder=2)
    if p is not None:
        p = np.array(p).reshape(-1, 2)
        plt.scatter(p[:, 0], p[:, 1], color='tab:orange', label='predictions', zorder=2)
    ylim = None
    if z is not None:
        z = np.array(z).reshape(-1, 2)
        plt.scatter(z[:, 0], z[:, 1], color='tab:blue', label='adjusted target', zorder=2)
    if bound_fn:
        y1_bound = bound_fn(y0)
        ylim = plt.ylim()
        plt.fill_between(y0, ylim[0], y1_bound, zorder=2, alpha=0.2, label='feasible output')
    if z is not None and p is not None:
        tmp = np.empty((len(p) + len(z), p.shape[1]))
        tmp[0::2, :] = p
        tmp[1::2, :] = z
        plt.plot(tmp[:, 0], tmp[:, 1], linestyle=':', color='0.5', zorder=1)
    plt.xlabel('y0')
    plt.ylabel('y1')
    plt.grid(linestyle=':')
    plt.axis('equal')
    plt.legend()
    if ylim is not None:
        plt.ylim(*ylim)
    plt.show()
