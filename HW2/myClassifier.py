import scipy.io
import numpy as np
import math
from numpy.linalg import norm
import sys
import ad

max_iterations = 1e+2
tolerance = 1e-8

def main():
    if len(sys.argv) != 2:
        print("usage: myClassifier.py [path-to-input-files]")
        exit(1)

    # data and labels are read in as numpy arrays
    data = scipy.io.loadmat(sys.argv[1] + '/DATA.mat')['DATA']
    labels = scipy.io.loadmat(sys.argv[1] + '/LABELS.mat')['LABELS']

    # run newton's method
    theta = getThetaFromNewton(labels, data)

    # print results
    print('norm(grad_f = ' + str(norm(grad_f(labels, data, theta))))
    print('theta = ' + str(theta))
    print('objective = ' + str(objective(labels, data, theta)))

    # now try a test point on the +1
    print('predict(' + str([-1, -1]) +') = ' + str(predict(np.array([-1, -1]).transpose(),theta)))

    # now try a test point on the -1
    print('predict(' + str([2, 1]) + ') = ' + str(predict(np.array([2, 1]).transpose(),theta)))

    # print accuracy
    print('accuracy = ' + str(accuracy(labels, data, theta)))

def getThetaFromNewton(L, x):
    # create theta parameter with random numbers in interval (0, 1)
    theta = np.random.rand(2)
    i = 0
    while norm(grad_f(L, x, theta)) > tolerance and i < max_iterations:
        theta = theta - np.linalg.inv(grad_grad_f(L, x, theta)).dot(grad_f(L, x, theta))
        i += 1
    return theta

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

# define hessian of probability function
def grad_grad_f(L, x, t):
    f00 = f01 = f10 = f11 = 0.0
    for i in range(len(x)):
        commonNumeratorPart = -1 * (L[i]**2 * math.exp(-1 * L[i] * t.transpose().dot(x[i])))
        denom = (1 + math.exp(-1 * L[i] * t.transpose().dot(x[i]))) ** 2
        f00 += ((x[i][0] **2) * commonNumeratorPart) / denom
        f01 += ((x[i][0] * x[i][1]) * commonNumeratorPart) / denom
        f10 += ((x[i][1] * x[i][0]) * commonNumeratorPart) / denom
        f11 += ((x[i][1] **2) * commonNumeratorPart) / denom
    hessian = np.array([[np.ndarray.item(f00), np.ndarray.item(f01)],
                     [np.ndarray.item(f10), np.ndarray.item(f11)]])

    return np.array([[1, 0], [0, 1]]) - hessian

# define objective function
def objective(L, x, t):
    sumOfLogs = 0
    for i in range(len(x)):
        p = p_of_L_given_x(L[i], x[i], t)
        sumOfLogs += math.log(p)
    return 1/2 * norm(t) **2 - sumOfLogs

if __name__ == "__main__":

    main()