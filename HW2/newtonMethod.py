import random
import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
import math
from scipy.optimize import fmin
import sys

epsilon = 1e-16
alpha_max = 1
alpha_min = 0
c1 = 0.0001
c2 = 0.8
max_iterations = 1e+3

def f(x_k):
    #return 100 * (x_k[1] - x_k[0]**2)**2 + (1 - x_k[0])**2
    return 1/4 * x_k[0]**4 + 1/2 * (x_k[1] - x_k[2])**2 + 1/2 * x_k[1]**2

def grad_f(x_k):
    '''partial_x1 = -400 * x_k[0] * x_k[1] + 400 * x_k[0] **3 + 2 * x_k[0] - 2
    partial_x2 = 200 * x_k[1] - 200 * x_k[0]**2
    return np.array([partial_x1, partial_x2])'''
    partial_x1 = x_k[0]**3
    partial_x2 = 2 * x_k[1] - x_k[2]
    partial_x3 = x_k[2] - x_k[1]
    return np.array([partial_x1, partial_x2, partial_x3])

def grad_grad_f(x_k):
    '''partial_x11 = -400 * x_k[1] + 1200 * x_k[0]**2 + 2
    partial_x21 = -400 * x_k[0]
    partial_x12 = -400 * x_k[0]
    partial_x22 = 200
    return np.array([[partial_x11, partial_x21], [partial_x12, partial_x22]])'''
    partial_x11 = 3 * x_k[0] **2
    partial_x12 = 0
    partial_x13 = 0
    partial_x21 = 0
    partial_x22 = 2
    partial_x23 = -1
    partial_x31 = 0
    partial_x32 = -1
    partial_x33 = 1
    return np.array([[partial_x11, partial_x12, partial_x13], [partial_x21, partial_x22, partial_x23],[partial_x31, partial_x32, partial_x33]])


def plotError(delta):
    ax = plt.axes()
    ax.set_ylabel("norm(x_k - x*)")
    ax.set_xlabel("iteration")
    x_points = [i for i in range(len(delta))]
    ax.scatter(x_points, delta)
    plt.show()


def findSolution(answer, xk):
    tolerance = 1e-8

    delta = []
    d = math.log(norm(xk - answer))
    delta.append(d)
    i = 0
    while norm(grad_f(xk)) > tolerance and i < max_iterations:
        xk = xk + -1.0 * np.linalg.inv(grad_grad_f(xk)).dot(grad_f(xk))
        try:
            d = math.log(norm(xk - answer))
            delta.append(d)
        except:
            pass
        i += 1
    return xk, delta, norm(grad_f(xk))

def main():

    x_star = fmin(f, np.array([5, 5, 5]))
    #x_star = fmin(f, np.array([1.1, 1.2]))
    print('x* = ' + str(x_star))
    #xk_0 = np.array([1.1, 1.2])
    xk_0 = np.array([5, 5, 5])
    x_k, delta, normGradF = findSolution(x_star, xk_0)
    print('xk = ' + str(x_k))
    print('num iterations = ' + str(len(delta)))
    f_xk = f(x_k)
    f_xstar = f(x_star)
    print('f(x*) = ' + str(f_xstar))
    print('f(xk) = ' + str(f_xk))
    grad_f_xk = grad_f(x_k)
    grad_f_x_star = grad_f(x_star)
    print('norm(grad_f(x*)) = ' + str(norm(grad_f_x_star)))
    print('norm(grad_f(xk)) = ' + str(norm(grad_f_xk)))
    plotError(delta)

if __name__ == "__main__":
    main()


