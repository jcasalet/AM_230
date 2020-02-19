import random
import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
import math
from scipy.optimize import fmin


max_iterations = 1e+4
epsilon = 1e-16
alpha_max = 1
alpha_min = 0
c=1
c1 = 1e-4
c2 = 0.9

'''
def f(x_k):
    return (c * x_k[0] - 2.0 ) ** 4 + x_k[1] * (c * x_k[0] - 2.0) ** 2 + (x_k[1] + 1.0) ** 2
def grad_f(x_k):
    partial_f_partial_x1 = 4.0 * c * (c * x_k[0] - 2.0) ** 3 + 2.0 * c * x_k[1] ** 2 * (c * x_k[0] - 2.0)
    partial_f_partial_x2 = 2.0 * x_k[1] * (c * x_k[0] - 2) ** 2 + 2.0 * (x_k[1] + 1)
    return np.array([partial_f_partial_x1, partial_f_partial_x2])
'''

def f(x_k):
    return 100 * (x_k[1] - x_k[0]**2)**2 + (1 - x_k[0])**2

def grad_f(x_k):
    partial_f_partial_x1 = -400 * x_k[0] * x_k[1] + 400 * x_k[0] **3 + 2 * x_k[0] - 2
    partial_f_partial_x2 = 200 * x_k[1] - 200 * x_k[0]**2
    return np.array([partial_f_partial_x1, partial_f_partial_x2])

def grad_grad_f(x_k):
    partial_x11 = -400 * x_k[1] + 1200 * x_k[0]**2 + 2
    partial_x21 = -400 * x_k[0]
    partial_x12 = -400 * x_k[1]
    partial_x22 = 200
    return np.array([[partial_x11, partial_x21], [partial_x12, partial_x22]])

'''def f(x_k):
    return 1/4 * x_k[0]**4 + 1/2 * (x_k[1] - x_k[2])**2 + 1/2 * x_k[1]**2
def grad_f(x_k):
    partial_f_partial_x1 = x_k[0]**3
    partial_f_partial_x2 = 2 * x_k[1] - x_k[2]
    partial_f_partial_x3 = x_k[2] - x_k[1]
    return np.array([partial_f_partial_x1, partial_f_partial_x2, partial_f_partial_x3])
def grad_grad_f(x_k):
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
'''

def phi(x_k, alpha):
    return f(x_k + alpha * -1 * grad_f(x_k))

def phi_prime(x_k, alpha):
    return grad_f(x_k + alpha * -1 * grad_f(x_k)).transpose().dot(-1 * grad_f(x_k))

def zoom(x_k, alpha_lo, alpha_hi):
    while abs(alpha_hi - alpha_lo) > epsilon:
    #while True:
        alpha_j = (alpha_lo + alpha_hi) / 2
        phi_alpha_j =  phi(x_k, alpha_j)
        # if φ(αj ) > φ(0) + c1αjφ(0) or φ(αj ) ≥ φ(αlo)
        if phi_alpha_j > phi(x_k, 0) + c1 * alpha_j * phi_prime(x_k, 0) or phi_alpha_j >= phi(x_k, alpha_lo):
            alpha_hi = alpha_j
        else:
            phi_prime_alpha_j = phi_prime(x_k, alpha_j)
            if abs(phi_prime_alpha_j) <= -1.0 * c2 * phi_prime(x_k, 0):
                return alpha_j
            if phi_prime_alpha_j * (alpha_hi - alpha_lo) >= 0:
                alpha_hi = alpha_lo
            alpha_lo = alpha_j

    return alpha_j

def lineSearch(x_k, alpha_i, alpha_i_minus_1):

    i = 1
    while abs(alpha_i - alpha_i_minus_1) > epsilon and i < max_iterations:
        phi_alpha_i = phi(x_k, alpha_i)
        # if φ(αi ) > φ(0) + c1αiφ(0) or [φ(αi ) ≥ φ(αi−1) and i > 1]
        if phi_alpha_i > phi(x_k, 0) + c1 * alpha_i * phi_prime(x_k, 0) or \
            (phi_alpha_i >= phi(x_k, alpha_i_minus_1) and i>1):
            return zoom(x_k, alpha_i_minus_1, alpha_i)
        # if |φ(αi )| ≤ −c2φ(0)
        if abs(phi_prime(x_k, alpha_i)) <= -1.0 * c2 * phi_prime(x_k, 0):
            return alpha_i
        # if φ(αi ) ≥ 0
        if phi_prime(x_k, alpha_i) >= 0:
            return zoom(x_k, alpha_i, alpha_i_minus_1)
        alpha_i_minus_1 = alpha_i
        alpha_i = random.uniform(alpha_i, alpha_max)
        i = i + 1

    return alpha_i

def plotError(delta):
    ax = plt.axes()
    ax.set_ylabel("log(norm(x_k - x*))")
    ax.set_xlabel("iteration")
    x_points = [i for i in range(len(delta))]
    ax.scatter(x_points, delta)
    plt.show()

def steepestDescent(alpha_0, alpha_1, answer, xk):
    tolerance = 1e-8
    alpha_i = alpha_1
    alpha_i_minus_1 = alpha_0
    delta = []
    try:
        d = math.log(norm(xk - answer))
        delta.append(d)
    except:
        delta.append(0)
    i = 0
    while norm(grad_f(xk)) > tolerance and i < max_iterations:
        alpha = lineSearch(xk, alpha_i, alpha_i_minus_1)
        xk = xk + alpha * -1.0 * grad_f(xk)
        alpha_i_minus_1 = alpha_i
        alpha_i = alpha
        try:
            d = math.log(norm(xk - answer))
            delta.append(d)
        except:
            pass
        i += 1
    return xk, delta

def main():
    alpha_0 = alpha_min
    alpha_1 = random.uniform(alpha_0, alpha_max)
    #answer = [2/c, -1]
    #answer=[1,1]
    answer = fmin(f, np.array([2, 2]))
    #answer = fmin(f, np.array([2, 2, 2]))
    print('answer = ' + str(answer))
    #xk_0 = np.array([2, 2, 2])
    xk_0 = np.array([2,2])
    x_k, delta = steepestDescent(alpha_0, alpha_1, answer, xk_0)
    print('x = ' + str(x_k) + ' with a delta = ' + str(delta[len(delta)-1]) )
    print('num iterations = ' + str(len(delta)))
    plotError(delta)

if __name__ == "__main__":
    main()

