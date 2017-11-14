import numpy as np


def train(X, Y, lamb):
    A = X.T.dot(X)
    A = A + lamb*np.identity(A.shape[0])
    b = X.T.dot(Y)
    W_hat = np.linalg.solve(A, b)
    return(W_hat)

def predict(W, X):
    ret = W.T.dot(X.T)
    return(list(np.argmax(ret, axis=0)))

def accuracy(y_true, y_hat):
    ones = []
    for i in range(0,len(y_hat)):
        b = y_hat[i]
    
        if y_true[i][b] == 1:
            ones.append(1.0)
        else:
            ones.append(0.0)
    return(np.mean(np.array(ones)))
