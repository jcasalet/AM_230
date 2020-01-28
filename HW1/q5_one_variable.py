import random
import numpy as np
alpha_max = 1 - 1e-7
alpha_min = 1e-7
import matplotlib.pyplot as plt

c=1
#c=10
c1 = 1e-4
c2 = 0.9
epsilon = 1e-8

'''def f(x_k, c):
    return (c * x_k[0] - 2 ) ** 4 + x_k[1] * (c * x_k[0] - 2) ** 2 + (x_k[1] + 1) ** 2

def f_prime(x_k, c):
    partial_f_partial_x1 = 4 * c * (c * x_k[0] - 2) ** 3 + 2 * c * x_k[1] ** 2 * (c * x_k[0] - 2)
    partial_f_partial_x2 = 2 * x_k[1] * (c * x_k[0] - 2) ** 2 + 2 * (x_k[1] + 1)
    return np.array([partial_f_partial_x1, partial_f_partial_x2])'''

def f(x_k):
    return x_k ** 2 - 6 * x_k + 9

def f_prime(x_k):
    return 2 * x_k - 6

def phi(x_k, p_k, alpha):
    return f(x_k + alpha * p_k)

def phi_prime(x_k, p_k, alpha):
    return 2 * p_k * (alpha ** 2 + x_k - 1)

def zoom(x_k, p_k, alpha_lo, alpha_hi):
    while True:

        alpha_j = (alpha_lo + alpha_hi) / 2
        phi_alpha_j =  phi(x_k, p_k, alpha_j)
        # if φ(αj ) > φ(0) + c1αjφ(0) or φ(αj ) ≥ φ(αlo)
        if phi_alpha_j > phi(x_k, p_k, 0) + c1 * alpha_j * phi_prime(x_k, p_k, 0) or phi_alpha_j >= phi(x_k, p_k, alpha_lo):
            alpha_hi = alpha_j
        else:
            phi_prime_alpha_j = phi_prime(x_k, p_k, alpha_j)
            if abs(phi_prime_alpha_j) <= -1 * c2 * phi_prime(x_k, p_k, 0):
                return alpha_j
            if phi_prime_alpha_j * (alpha_hi - alpha_lo) >= 0:
                alpha_hi = alpha_lo
            alpha_lo = alpha_j

        return alpha_lo

def lineSearch(x_k, alpha_i, alpha_i_minus_1):

    p_k = -1 * f_prime(x_k)
    phi_alpha_i = phi(x_k, p_k, alpha_i)
    i = 1
    while True:
        # if φ(αi ) > φ(0) + c1αiφ(0) or [φ(αi ) ≥ φ(αi−1) and i > 1]
        if phi_alpha_i > phi(x_k, p_k, 0) + c1 * alpha_i * phi_prime(x_k, p_k, 0) or \
            (phi_alpha_i >= phi(x_k, p_k, alpha_i_minus_1) and i>1):
            return zoom(x_k, p_k, alpha_i_minus_1, alpha_i)
        # if |φ(αi )| ≤ −c2φ(0)
        if abs(phi_prime(x_k, p_k, alpha_i)) <= -1 * c2 * phi_prime(x_k, p_k, 0):
            return alpha_i
        # if φ(αi ) ≥ 0
        if phi_prime(x_k, p_k, alpha_i) >= 0:
            return zoom(x_k, p_k, alpha_i, alpha_max)
        alpha_i_minus_1 = alpha_i
        alpha_i = random.uniform(alpha_i, alpha_max)
        i = i + 1

    return alpha_min

def plotError(delta):
    ax = plt.axes()
    x_points = [i for i in range(len(delta))]
    ax.scatter(x_points, delta)
    plt.show()

def main():
    alpha_0 = 0
    alpha_1 = random.uniform(alpha_0, alpha_max)
    answer = 3
    #x_k = 2
    x_k = np.array([200])
    #x_k = np.array([0, 0])
    tolerance = 0.001

    alpha_i = alpha_1
    alpha_i_minus_1 = alpha_0
    delta = []
    delta.append(abs(x_k - answer))
    i = 0
    while delta[i] > tolerance:
    #while abs(alpha_i_minus_1 - alpha_i) > epsilon:
        alpha = lineSearch(x_k, alpha_i, alpha_i_minus_1)
        print('alpha = ' + str(alpha))
        print('delta = ' + str(delta))
        print('x = ' + str(x_k))
        x_k = x_k + alpha * -1 * f_prime(x_k)
        alpha_i_minus_1 = alpha_i
        alpha_i = alpha
        delta.append(abs(x_k - answer))
        i += 1

    print('x = ' + str(x_k) + ' with a delta = ' + str(delta))
    plotError(delta)

if __name__ == "__main__":
    main()


