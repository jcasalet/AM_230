import numpy as np
from numpy.linalg import qr
import random
import matplotlib.pyplot as plt
from numpy.linalg import norm
import math
import sys

max_iterations = 1e+3
threshold = 1e-8

def plotGradient(grad, ylabel):
    ax = plt.axes()
    ax.set_ylabel(ylabel)
    ax.set_xlabel("iteration")
    x_points = [i+1 for i in range(len(grad))]
    ax.scatter(x_points, grad)
    plt.show()

def generateMatrix(n, dist):
    Y = np.random.rand(n, n)
    Q, R = qr(Y)
    D = np.zeros((n, n))
    if dist is 'uniform':
        for i in range(len(D)):
            D[i][i] = random.randint(10, 10**3)
    elif dist is 'four':
        for i in range(len(D)):
            # pick a dist
            newRand = np.random.random()
            if newRand < 0.25:
                # pick value from  (9, 11)
                D[i][i] = np.random.randint(9, 11)
            elif newRand < 0.5:
                # pick value from  (99, 101)
                D[i][i] = np.random.randint(99, 101)
            elif newRand < 0.75:
                # pick value from  (999, 1001)
                D[i][i] = np.random.randint(999, 1001)
            else:
                # pick value from (9999, 10001)
                D[i][i] = np.random.randint(9999, 10001)
    elif dist is 'two':
        for i in range(len(D)):
            newRand = np.random.random()
            if newRand < 0.5:
                D[i][i] = np.random.randint(9, 11)
            else:
                D[i][i] = np.random.randint(999, 1001)

    else:
        print('dist: ' + str(dist) + ' is unknown.  use uniform, two, or four')
        sys.exit(1)

    return Q.transpose().dot(D).dot(Q)

def f(x, A, b):
    return 0.5 * x.transpose().dot(A).dot(x) - b.transpose().dot(x)

def grad_f(x, A, b):
    return A.dot(x) - b

def runConjugateGradient(n, eigenDist):
    b = np.zeros(n)
    A = generateMatrix(n, eigenDist)
    x_k = np.array([random.random() for i in range(0, n)])
    x0 = x_k
    r_k = grad_f(x_k, A, b)
    p_k = -1 * r_k
    k = 0
    normErrors = list()

    while norm(A.dot(x_k) - b) > threshold:
        normErrors.append(math.log(norm(A.dot(x_k) - b)))

        alpha_k = r_k.transpose().dot(r_k) / p_k.transpose().dot(A).dot(p_k)
        x_kplus1 = x_k + alpha_k * p_k
        r_kplus1 = r_k + alpha_k * A.dot(p_k)
        b_kplus1 = r_kplus1.transpose().dot(r_kplus1) / r_k.transpose().dot(r_k)
        p_k = -1 * r_kplus1 + b_kplus1 * p_k
        k = k+1
        x_k = x_kplus1
        r_k = r_kplus1
    return A, x0, x_k, normErrors


def main():
    A, x0, xstar, normErrors = runConjugateGradient(10**3, 'four')
    print('x* = : ' + str(xstar))
    plotGradient(normErrors, "norm(Ax - b)")


if __name__ == "__main__":
    main()
