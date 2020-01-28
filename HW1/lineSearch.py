from scipy import optimize
import scipy as sp
import numpy as np
import random
from numpy.linalg import norm
import matplotlib.pyplot as plt

c = 1

def f(x_k):
    return (c * x_k[0] - 2 ) ** 4 + x_k[1] * (c * x_k[0] - 2) ** 2 + (x_k[1] + 1) ** 2

def f_prime(x_k):
    partial_f_partial_x1 = 4 * c * (c * x_k[0] - 2) ** 3 + 2 * c * x_k[1] ** 2 * (c * x_k[0] - 2)
    partial_f_partial_x2 = 2 * x_k[1] * (c * x_k[0] - 2) ** 2 + 2 * (x_k[1] + 1)
    return np.array([partial_f_partial_x1, partial_f_partial_x2])

def test_func(x):
    return (x[0])**2+(x[1])**2




def plotError(delta):
    ax = plt.axes()
    x_points = [i for i in range(len(delta))]
    ax.scatter(x_points, delta)
    plt.show()


def main():
    epsilon = 1e-16
    alpha_max = 1 - 1e-7
    alpha_min = 1e-7

    alpha_0 = 0.0001
    alpha_1 = random.uniform(alpha_0, alpha_max)
    answer = [2, -1]
    #x_k = 2
    x_k = np.array([-1.9, 0.9])
    tolerance = 0.001

    alpha_i = alpha_1
    alpha_i_minus_1 = alpha_0
    delta = []
    delta.append(norm(x_k - answer))
    i = 0
    while delta[i] > tolerance:
        p_k = -1 * f_prime(x_k)
        alpha = sp.optimize.line_search(f, f_prime, x_k, p_k, amax = 1000, c1=0.01, c2=0.8)[0]
        print('alpha = ' + str(alpha))
        x_k = x_k + alpha * -1 * f_prime(x_k)
        delta.append(norm(x_k - answer))
        i += 1

    print('x = ' + str(x_k) + ' with a delta = ' + str(delta[i]) )
    print('num iterations = ' + str(i))
    plotError(delta)

if __name__ == "__main__":
    main()




