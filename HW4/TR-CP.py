import math
import numpy as np
import random
import matplotlib.pyplot as plt



max_iterations = 50
epsilon = 1e-8

def plotError(deltas, graphTitle):
    ax = plt.axes()
    ax.set_ylabel("log(norm(error))")
    ax.set_xlabel("iteration")
    x_points = [i+1 for i in range(len(deltas))]
    ax.scatter(x_points, deltas)
    plt.title(graphTitle)
    plt.show()


def generate_Q(n):
    Y = np.random.rand(n, n)
    Q, R = np.linalg.qr(Y)
    D = np.zeros((n, n))
    for i in range(len(D)):
        D[i][i] = random.random()
    return np.matmul(np.matmul(Q.transpose(), D), Q)

def f(x, Q):
    return math.log(1 + x.transpose().dot(Q.dot(x)))

def grad_f(x, Q):
    return (x.transpose().dot(Q)) / (1 + x.transpose().dot(Q.dot(x)))

def hessian(x, Q):
    # u'v + v'u
    u = x.transpose().dot(Q)
    v = (1 + x.transpose().dot(Q.dot(x))) ** -1
    u_prime = Q
    v_prime = -1 * (1 + x.transpose().dot(Q.dot(x))) ** -2 * x.transpose().dot(Q)
    return u_prime.dot(v) + v_prime.dot(u)

def m(x_k, Q, p, B_k):
    # m(p) = fk + gk^Tp + 1/2 p^TBkp
    return f(x_k, Q) + grad_f(x_k, Q).transpose().dot(p) + 0.5 * p.transpose().dot(B_k.dot(p))

def m_prime(x_k, Q, p, B_k):
    # d/dp m(p) = gk^T + 1/2Bkp
    return grad_f(x_k, Q).transpose + 0.5 * B_k.dot(p)

def getCauchyPoint(gk, Bk, deltak):
    gt_B_g = gk.transpose().dot(Bk.dot(gk))
    norm_gk = np.linalg.norm(gk)
    if gt_B_g <= 0:
        tau_k = 1.0
    else:
        tau_k = min(norm_gk**3 / (deltak * gt_B_g), 1.0)

    return -1.0 * tau_k * (deltak / norm_gk) * gk


def calculate_rho(x, p, Q, B, n):
    numerator = f(x, Q) - f(x + p, Q)
    denominator = m(x, Q, np.zeros(n), B) - m(x, Q, p, B)
    return numerator / denominator

def runTrustRegion(x0, Q, delta_hat, n):
    xk = x0
    deltak = random.uniform(0, delta_hat)
    eta = random.uniform(0, 0.25)
    k = 0
    error = list()
    while np.linalg.norm(grad_f(xk, Q)) > epsilon and k < max_iterations:
        Bk = hessian(xk, Q)
        pk = getCauchyPoint(grad_f(xk, Q), Bk, deltak)
        print(pk)
        rho_k = calculate_rho(xk, pk, Q, Bk, n)
        if rho_k < 0.25:
            deltak = 0.25 * deltak

        elif rho_k > 0.75 and np.isclose(np.linalg.norm(pk), deltak, epsilon):
            deltak = min(2 * deltak, delta_hat)

        if rho_k > eta:
            xk = xk + pk

        k += 1
        error.append(math.log(np.linalg.norm(xk - np.zeros(n))))

    return xk, error



def main():
    n=10
    Q = generate_Q(n)
    x0=np.random.rand(n)
    delta_hat = 1
    x_star, error = runTrustRegion(x0, Q, delta_hat, n)
    print('x* = ' + str(x_star) + ' num iterations: ' + str(len(error)))
    plotError(error, 'TR-cauchy: n=' + str(n))


if __name__ == "__main__":
    main()