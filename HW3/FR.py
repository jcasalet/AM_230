import numpy as np
from numpy.linalg import qr
import random
import matplotlib.pyplot as plt
from numpy.linalg import norm
import math
import scipy.optimize
import sys
#import jax -- has grad() built-in that you can compile with jit()

max_iterations = 1e+2
epsilon = 1e-4
alpha_max = 1
alpha_min = 0
c1 = 0.1
c2 = 0.4

def plotError(deltas, graphTitle):
    ax = plt.axes()
    ax.set_ylabel("log(norm(error))")
    ax.set_xlabel("iteration")
    x_points = [i+1 for i in range(len(deltas))]
    ax.scatter(x_points, deltas)
    plt.title(graphTitle)
    plt.show()

def f(x):
    sum = 0.0
    for i in range(len(x)-1):
        sum += 100*(x[i]**2 - x[i+1])**2 + (x[i] - 1)**2
    return sum


def grad_f(x):
    n = len(x)
    gradF = np.zeros(n)
    gradF[0] = 400 * x[0]**3 - 400 * x[0]*x[1] + 2 * x[0] - 2
    gradF[n-1] = 200 * x[n-1] - 200 * x[n-2]**2

    for i in range(1, n-1):
        gradF[i] = 200*(x[i-1]**2 -x[i])*(-1) + 200*(x[i]**2 - x[i+1])*(2*x[i]) + 2*(x[i] - 1)

    return gradF

def phi(xk, alpha):
    return f(xk - alpha * grad_f(xk))


def phi_prime(xk, alpha):
    return np.dot(grad_f(xk - alpha * grad_f(xk)), -grad_f(xk))


def main():
    n = 10
    x_k = np.random.rand(n)
    p_k = -grad_f(x_k)
    deltas = list()
    k=1

    while norm(grad_f(x_k)) > epsilon:

        alpha, fc, gc, new_fval, old_fval, new_slope = scipy.optimize.line_search(f=f, myfprime=grad_f, xk=x_k, pk=p_k, gfk = grad_f(x_k), amax=1, maxiter=5, c1=c1, c2=c2)
        x_kplus1 = x_k + alpha * p_k

        gradf_k = grad_f(x_k)
        gradf_kplus1 = grad_f(x_kplus1)

        b_kplus1 = np.dot(gradf_kplus1, gradf_kplus1) / np.dot(gradf_k, gradf_k)

        p_k = -gradf_kplus1 + b_kplus1 * p_k
        x_k = x_kplus1
        k+=1
        deltas.append(math.log(norm(gradf_k)))

    print('x* = ' + str(x_k))
    plotError(deltas, 'fr')


if __name__ == "__main__":
    main()