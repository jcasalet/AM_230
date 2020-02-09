import random
import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
import math
import sys
import scipy.io


max_iterations = 1e+4
epsilon = 1e-16
alpha_max = 1
alpha_min = 0
c=1
c1 = 1e-4
c2 = 0.9

# define probability function
def p_of_L_given_x(L, x, t):
    return 1 / (1 + math.exp(-1 * L.dot(t.transpose().dot(x))))

# define gradient of probability function
def grad_f(L, x, t):
    f_0 = f_1 = 0.0
    for i in range(len(x)):
        x0_num = L[i] * x[i][0] * math.exp(-1 * L[i] * t.transpose().dot(x[i]))
        x1_num = L[i] * x[i][1] * math.exp(-1 * L[i] * t.transpose().dot(x[i]))
        den = 1 + math.exp(-1 * L[i] * t.transpose().dot(x[i]))
        f_0 += np.ndarray.item(x0_num)/ den
        f_1 += np.ndarray.item(x1_num) / den
    return t - np.array([f_0, f_1])

# define objective function
def f(L, x, t):
    sumOfLogs = 0
    for i in range(len(x)):
        p = p_of_L_given_x(L[i], x[i], t)
        sumOfLogs += math.log(p)
    return 1/2 * norm(t) **2 - sumOfLogs


def phi(L, x, t, alpha):
    return f(L, x, t + alpha * -1 * grad_f(L, x, t))

def phi_prime(L, x, t, alpha):
    return grad_f(L, x, t + alpha * -1 * grad_f(L, x, t)).transpose().dot(-1 * grad_f(L, x, t))

def zoom(L, x, t, alpha_lo, alpha_hi):
    while abs(alpha_hi - alpha_lo) > epsilon:
    #while True:
        alpha_j = (alpha_lo + alpha_hi) / 2
        phi_alpha_j =  phi(L, x, t, alpha_j)
        # if φ(αj ) > φ(0) + c1αjφ(0) or φ(αj ) ≥ φ(αlo)
        if phi_alpha_j > phi(L, x, t, 0) + c1 * alpha_j * phi_prime(L, x, t, 0) or phi_alpha_j >= phi(L, x, t, alpha_lo):
            alpha_hi = alpha_j
        else:
            phi_prime_alpha_j = phi_prime(L, x, t, alpha_j)
            if abs(phi_prime_alpha_j) <= -1.0 * c2 * phi_prime(L, x, t, 0):
                return alpha_j
            if phi_prime_alpha_j * (alpha_hi - alpha_lo) >= 0:
                alpha_hi = alpha_lo
            alpha_lo = alpha_j

    return alpha_j

def lineSearch(L, x, t, alpha_i, alpha_i_minus_1):

    phi_alpha_i = phi(L, x, t, alpha_i)
    i = 1
    while abs(alpha_i - alpha_i_minus_1) > epsilon:
        # if φ(αi ) > φ(0) + c1αiφ(0) or [φ(αi ) ≥ φ(αi−1) and i > 1]
        if phi_alpha_i > phi(L, x,t, 0) + c1 * alpha_i * phi_prime(L, x, t, 0) or \
            (phi_alpha_i >= phi(L, x, t, alpha_i_minus_1) and i>1):
            return zoom(L, x, t, alpha_i_minus_1, alpha_i)
        # if |φ(αi )| ≤ −c2φ(0)
        if abs(phi_prime(L, x, t, alpha_i)) <= -1.0 * c2 * phi_prime(L, x, t, 0):
            return alpha_i
        # if φ(αi ) ≥ 0
        if phi_prime(L, x, t, alpha_i) >= 0:
            return zoom(L, x, t, alpha_i, alpha_i_minus_1)
        alpha_i_minus_1 = alpha_i
        alpha_i = random.uniform(alpha_i, alpha_max)
        i = i + 1

    return alpha_i

def steepestDescent(L, x, t, alpha_0, alpha_1):
    tolerance = 1e-8
    alpha_i = alpha_1
    alpha_i_minus_1 = alpha_0
    i = 0
    while norm(grad_f(L, x, t)) > tolerance and i < max_iterations:
        alpha = lineSearch(L, x, t, alpha_i, alpha_i_minus_1)
        t = t + alpha * -1.0 * grad_f(L, x, t)
        alpha_i_minus_1 = alpha_i
        alpha_i = alpha

        i += 1
    return t, i

# calculate accuracy for model t
def accuracy(L, x, t):
    numCorrect = 0
    for i in range(len(x)):
        y_hat = predict(x[i], t)
        if y_hat == np.ndarray.item(L[i]):
            numCorrect +=1
    return (1.0 * numCorrect) / (1.0 * len(x))

# predict label for single  point
def predict(x, t):
    # predict for L=1
    p_pos1 = 1/(1 - t.dot(x))
    # predict for L=-1
    p_neg1 = 1/(1 + t.dot(x))
    # select higher probability
    # TODO somehow my signs are reversed
    if p_pos1 > p_neg1:
        return -1
    else:
        return 1

def main():
    if len(sys.argv) != 2:
        print("usage: newtonMethodClassifier.py [path-to-input-files]")
        exit(1)

    # data and labels are read in as numpy arrays
    data = scipy.io.loadmat(sys.argv[1] + '/DATA.mat')['DATA']
    labels = scipy.io.loadmat(sys.argv[1] + '/LABELS.mat')['LABELS']
    alpha_0 = alpha_min
    alpha_1 = random.uniform(alpha_0, alpha_max)
    theta_0 = np.array([1.2,1.2])
    theta, i = steepestDescent(labels, data, theta_0, alpha_0, alpha_1)
    # print results
    print('using steepest descent:')
    print('theta = ' + str(theta))
    print('norm(grad_f = ' + str(norm(grad_f(labels, data, theta))))
    print('objective = ' + str(f(labels, data, theta)))
    # now try a test point on the +1
    print('predict(' + str([-1, -1]) + ') = ' + str(predict(np.array([-1, -1]).transpose(), theta)))
    # now try a test point on the -1
    print('predict(' + str([2, 1]) + ') = ' + str(predict(np.array([2, 1]).transpose(), theta)))
    # print accuracy
    print('accuracy = ' + str(accuracy(labels, data, theta)))
    print('num iterations = ' + str(i))

if __name__ == "__main__":
    main()

