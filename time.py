import timeit
import numpy as np


def timer(sig):
    z = np.random.randint(-2, 2)
    return sig(z)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def exp(z):
    return np.exp(-z)


def sigmoid_2nd(z):
    return 1.0 / (1.0 + 2 ** (-z))


def relu(z):
    if z > 0:
        return z
    return 0.01 * z


def base(z):
    return z


print('base\t', timeit.timeit(lambda: timer(base), number=1000000))
print('sigmoid exp\t', timeit.timeit(lambda: timer(sigmoid), number=1000000))
print('exp\t', timeit.timeit(lambda: timer(exp), number=1000000))
print('sigmoid 2nd pwr\t', timeit.timeit(lambda: timer(sigmoid_2nd), number=1000000))
print('leaky relu\t', timeit.timeit(lambda: timer(relu), number=1000000))
