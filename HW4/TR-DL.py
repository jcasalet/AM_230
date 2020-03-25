import math
import numpy as np
import random
import matplotlib.pyplot as plt



max_iterations = 50
delta_hat = 1
eta = random.uniform(0, 0.25)
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
    return Q.transpose().dot(D.dot(Q))

def f(x, Q):
    return math.log(1 + x.transpose().dot(Q.dot(x)))

def grad_f(x, Q):
    return (2 * x.transpose().dot(Q)) / (1 + x.transpose().dot(Q.dot(x)))


def hessian(x, Q):
    u = x.transpose().dot(Q)
    v = (1 + x.transpose().dot(Q.dot(x))) ** -1
    u_prime = Q
    v_prime = -1 * (1 + x.transpose().dot(Q.dot(x))) ** -2 * grad_f(x, Q)
    return u_prime.dot(v) + v_prime.dot(u)

def m(x_k, Q, p, B_k):
    # m(p) = fk + gk^Tp + 1/2 p^TBkp
    return f(x_k, Q) + grad_f(x_k, Q).transpose().dot(p) + 0.5 * p.transpose().dot(B_k.dot(p))

def m_prime(x_k, Q, p, B_k):
    # d/dp m(p) = gk^T + 1/2Bkp
    return grad_f(x_k, Q).transpose + 0.5 * B_k.dot(p)

def getTau(pu, pb, delta):
    # use quadratic formula to solve for tau
    a = np.linalg.norm(pb -pu)**2
    b = 2 * (pb - pu).transpose().dot(pu)
    c = np.linalg.norm(pu)**2 - delta**2
    return 1 + (-b + math.sqrt(b**2 - 4 * a *c))/(2 * a)


def dogleg(gk, B, delta):
    pb = -1.0 * np.linalg.inv(B).dot(gk)
    if np.linalg.norm(pb) <= delta:
        return pb

    pu = -1.0 * (gk.transpose().dot(gk)) / (gk.transpose().dot(B).dot(gk)) * gk
    if np.linalg.norm(pu) >= delta:
        return -1.0 * (delta / np.linalg.norm(pu))  * gk

    tau = getTau(pu, pb, delta)
    if tau <= 1:
        return tau * pu
    else:
        return pu + (tau - 1)*(pb - pu)

def calculate_rho(x, p, Q, B, n):
    numerator = f(x, Q) - f(x + p, Q)
    denominator = m(x, Q, np.zeros(n), B) - m(x, Q, p, B)
    return numerator / denominator

def runTrustRegion(x0, Q, delta_hat, n):
    xk = x0
    deltak = random.uniform(0, delta_hat)
    k = 0
    error = list()
    while k < max_iterations:
        Bk = hessian(xk, Q)
        pk = dogleg(grad_f(xk, Q), Bk, deltak)
        rho_k = calculate_rho(xk, pk, Q, Bk, n)

        if rho_k < 0.25:
            deltak = 0.25 * deltak
        elif rho_k > 0.75 and np.isclose(np.linalg.norm(pk), deltak, 1e-4):
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

    x_star, error = runTrustRegion(x0, Q, delta_hat, n)
    print('x* = ' + str(x_star) + ' num iterations: ' + str(len(error)))
    plotError(error, 'TR dogleg: n=' + str(n))


if __name__ == "__main__":
    main()