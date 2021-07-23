import numpy as np

def softmax(a):
    b = np.copy(a)
    b = np.exp(b)
    b /= np.sum(b)
    return b

def relu(a):
    relu_der = np.ones_like(a)
    for i in range(len(a)):
        for j in range(len(a)):
            if a[i][j] < 0:
                relu_der[i][j] = 0
                a[i][j] = 0

    return a, relu_der
