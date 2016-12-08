from pylab import *
import numpy as np

def fun_1():
    X = np.linspace(-np.pi, np.pi, 256, endpoint = True)
    C, S = np.cos(X), np.sin(X)
    plot(X, C)
    plot(X, S)
    show()

def fun_2():
    figure(figsize = (8, 6), dpi = 80)
    subplot(1, 1, 1)
    X = np.linspace(-np.pi, np.pi, 256, endpoint = True)
    C, S = np.cos(X), np.sin(X)

    plot(X, C, color = "blue", linewidth = 1.0, linestyle = "-")
    plot(X, S, color = "green", linewidth = 1.0, linestyle = "-")

    #xlim(-4.0, 4.0)
    xlim(X.min() * 1.1, X.max() * 1.1)

    xticks(np.linspace(-4, 4, 9, endpoint = True))

    #ylim(-1.0, 1.0)
    ylim(C.min() * 1.1, C.max() * 1.1)

    yticks(np.linspace(-1, 1, 5, endpoint = True))

    show()

if __name__ == "__main__":
    fun_2()