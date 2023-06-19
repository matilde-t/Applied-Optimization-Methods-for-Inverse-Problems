from aomip import ProximalOperators
import tifffile
import matplotlib.pyplot as plt
import numpy as np
import os
from challenge.utils import load_htc2022data, segment, calculate_score

os.environ["CUDA_VISIBLE_DEVICES"] = "3"


def test_operators():
    x = np.arange(-10, 10)
    # Indicator function
    c = [-4, 4]
    y = [ProximalOperators().indicator(i, c) for i in x]
    plt.figure()
    plt.plot(x, y)
    plt.title("Indicator function, c = [-4, 4]")
    plt.savefig("./homework/hw05/indicator.png")

    # Constant function
    y = [ProximalOperators().constant(i) for i in x]
    plt.figure()
    plt.plot(x, y)
    plt.title("Constant function")
    plt.savefig("./homework/hw05/constant.png")

    # Translation
    y = [
        ProximalOperators().translation(i, ProximalOperators().l2, 5, sigma=1)
        for i in x
    ]
    plt.figure()
    plt.plot(x, y)
    plt.title("Translation, f(x) = 1/2 * ||x - 5||^2")
    plt.savefig("./homework/hw05/translation.png")

    sigma = [0, 5, 10, 15, 20]
    # l2 norm
    fig = plt.figure()
    legend = []
    for s in sigma:
        y = [ProximalOperators().l2(i, sigma=s) for i in x]
        plt.plot(x, y)
        legend.append("sigma = {}".format(s))
    plt.legend(legend)
    fig.suptitle("L2 norm")
    plt.savefig("./homework/hw05/l2.png")

    # Huber
    fig = plt.figure()
    legend = []
    sigma = [5, 20]
    delta = [0.1, 0.5, 1, 5]
    for d in delta:
        for s in sigma:
            y = [ProximalOperators().huber(i, d, s) for i in x]
            plt.plot(x, y)
            legend.append("d = {}, s = {}".format(d, s))
    plt.legend(legend)
    fig.suptitle("Huber")
    plt.savefig("./homework/hw05/huber.png")

    # L1 norm
    fig = plt.figure()
    legend = []
    sigma = [0, 5, 10, 15, 20]
    for s in sigma:
        y = [ProximalOperators().l1(i, sigma=s) for i in x]
        plt.plot(x, y)
        legend.append("sigma = {}".format(s))
    plt.legend(legend)
    fig.suptitle("L1 norm")
    plt.savefig("./homework/hw05/l1.png")

    return


def test_all():
    test_operators()
    return


if __name__ == "__main__":
    test_operators()
