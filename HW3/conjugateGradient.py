import numpy as np
from numpy.linalg import qr
import random
import matplotlib.pyplot as plt
from numpy.linalg import norm
from numpy.linalg import inv
import math
import sys

max_iterations = 1e+3
threshold = 1e-8

def plotGradient(grad, ylabel, graphTitle):
    ax = plt.axes()
    ax.set_ylabel(ylabel)
    ax.set_xlabel("iteration")
    x_points = [i+1 for i in range(len(grad))]
    ax.scatter(x_points, grad)
    plt.title(graphTitle)
    plt.show()

def generateMatrix(n, dist):
    Y = np.random.rand(n, n)
    Q, R = qr(Y)
    D = np.zeros((n, n))
    lambda_max = -sys.maxsize
    lambda_min = sys.maxsize
    if dist is 'uniform':
        for i in range(len(D)):
            lambda_i = random.randint(10, 10**3)
            D[i][i] = lambda_i
            if lambda_i < lambda_min:
                lambda_min = lambda_i
            if lambda_i > lambda_max:
                lambda_max = lambda_i
    elif dist is 'four':
        for i in range(len(D)):
            # pick a dist
            newRand = np.random.random()
            if newRand < 0.25:
                # pick value from  (9, 11)
                lambda_i = np.random.randint(9, 11)
                D[i][i] = lambda_i

            elif newRand < 0.5:
                # pick value from  (99, 101)
                lambda_i = np.random.randint(99, 101)
                D[i][i] = lambda_i

            elif newRand < 0.75:
                # pick value from  (999, 1001)
                lambda_i = np.random.randint(999, 1001)
                D[i][i] = lambda_i

            else:
                # pick value from (9999, 10001)
                lambda_i = np.random.randint(9999, 10001)
                D[i][i] = lambda_i

            if lambda_i < lambda_min:
                lambda_min = lambda_i
            if lambda_i > lambda_max:
                lambda_max = lambda_i

    elif dist is 'two':
        for i in range(len(D)):
            newRand = np.random.random()
            if newRand < 0.5:
                lambda_i = np.random.randint(9, 11)
                D[i][i] = lambda_i
            else:
                lambda_i = np.random.randint(999, 1001)
                D[i][i] = lambda_i
            if lambda_i < lambda_min:
                lambda_min = lambda_i
            if lambda_i > lambda_max:
                lambda_max = lambda_i

    else:
        print('dist: ' + str(dist) + ' is unknown.  use uniform, two, or four')
        sys.exit(1)

    return Q.transpose().dot(D).dot(Q), lambda_min, lambda_max

def f(x, A, b):
    return 0.5 * x.transpose().dot(A).dot(x) - b.transpose().dot(x)

def grad_f(x, A, b):
    return A.dot(x) - b

def runConjugateGradient(n, eigenDist):
    b = np.zeros(n)
    A, lambda_min, lambda_max = generateMatrix(n, eigenDist)
    x_k = np.array([random.random() for i in range(0, n)])
    x0 = x_k
    r_k = grad_f(x_k, A, b)
    p_k = -1 * r_k
    k = 0
    normErrors = list()

    while norm(A.dot(x_k) - b.transpose().dot(x_k)) > threshold:
        normErrors.append(math.log(norm(A.dot(x_k) - b)))

        alpha_k = r_k.transpose().dot(r_k) / p_k.transpose().dot(A).dot(p_k)
        x_kplus1 = x_k + alpha_k * p_k
        r_kplus1 = r_k + alpha_k * A.dot(p_k)
        b_kplus1 = r_kplus1.transpose().dot(r_kplus1) / r_k.transpose().dot(r_k)
        p_k = -1 * r_kplus1 + b_kplus1 * p_k
        k = k+1
        x_k = x_kplus1
        r_k = r_kplus1
    return A, x0, x_k, normErrors, k, lambda_min, lambda_max


def main():
    eigenDist = 'two'
    solution = np.zeros(10**3)
    A, x0, xstar, normErrors, iterations, lambda_min, lambda_max = runConjugateGradient(10**3, eigenDist)

    # now calculate theoretical convergence rate
    conditionNumber = lambda_max / lambda_min
    print('k(A) = ' + str(conditionNumber))

    # show 5.36 holds for convergence
    LHS = norm(xstar - solution)
    RHS = 2 * ( (math.sqrt(conditionNumber) - 1) / (math.sqrt(conditionNumber) + 1) ) ** iterations * norm(x0 - solution)
    print(str(LHS) + ' <= ' + str(RHS))
    if LHS <= RHS:
        print('formula 5.36 holds!')
    else:
        print('formula 5.36 failed!')

    plotGradient(normErrors, "log(norm(error))", eigenDist)


if __name__ == "__main__":
    main()
