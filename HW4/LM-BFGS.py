import scipy.optimize
import math
import numpy as np
import random
import matplotlib.pyplot as plt


threshold = 1e-4
c1 = 1e-4
c2 = 0.9
alpha_max = 1
alpha_min = 0
max_iterations = 1e+4
alpha = 100

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
    n = x.shape[0]
    if n % 2 == 0:
        end = math.floor(n/2)
    else:
        end = math.floor(n/2) + 1
    for i in range(0, end):
        sum += alpha * ( (x[2*i+1] - x[2*i]**2)**2 ) + (1 - x[2*i])**2
    return sum

def grad_f(x):
    n = x.shape[0]
    grad = np.zeros(n)
    if n % 2 == 0:
        end = n - 1
    else:
        end = n - 2
    for i in range(0, end, 2):
        grad[i] = 2 * alpha * (x[i+1] - x[i]**2) * (-2 * x[i]) + 2 * (1 - x[i]) * -1
        grad[i+1] = 2 * alpha * (x[i+1] - x[i]**2)
    if n % 2 != 0:
        grad[n-1] = 2 * alpha * (x[n-1] - x[n-2]**2) * (-2 * x[n-2]) + 2 * (1 - x[n-2]) * -1

    return grad



def compute_Hk0(s, y, k, n):
    end = len(s) - 1
    if k == 0:
        return np.identity(n)
    else:
        gamma_k = s[end-1].transpose().dot(y[end-1]) / y[end-1].transpose().dot(y[end-1])
        return gamma_k * np.identity(n)


def compute_pk(gradf, s, y, k, m, n):
    end = len(s) - 1
    rho = np.zeros(k)
    q = gradf
    alpha = np.zeros(k)
    for i in range(end, -1, -1):
        rho[i] = 1 / (y[i].transpose().dot(s[i]))
        alpha[i] = (rho[i] * np.dot(s[i], q))
        q = q - alpha[i] *  y[i]
    r = compute_Hk0(s, y, k, n).dot(q)

    for i in range(0, end+1):
        beta = rho[i] * y[i].transpose().dot(r)
        r = r + s[i] * (alpha[i] - beta)
    return -1.0 * r

def main():

    n = 1000
    m = 10

    s = list()
    y = list()
    xk = np.random.rand(n)

    delta = list()

    k = 0
    while k < max_iterations:

        pk = compute_pk(grad_f(xk), s, y, k, m, n)

        alphak, fc, gc, new_fval, old_fval, new_slope = scipy.optimize.line_search(f=f, myfprime=grad_f, xk=xk, pk=pk, amax=1, maxiter=100, c1=c1, c2=c2)

        x_kplus1 = xk + alphak * pk

        sk = x_kplus1 - xk

        yk = grad_f(x_kplus1) - grad_f(xk)

        if k > m:
            s.pop(0)
            s.append(sk)
            y.pop(0)
            y.append(yk)
        else:
            s.append(sk)
            y.append(yk)

        xk = x_kplus1

        k += 1

        delta.append(math.log(np.linalg.norm(xk - np.ones(n))))

        if f(xk) <= threshold:
            break


    print('xk = ' + str(xk))
    plotError(delta, 'LM-BFGS: n=' + str(n) + ', m=' + str(m) + ', alpha=' + str(alpha))

if __name__ == "__main__":
    main()