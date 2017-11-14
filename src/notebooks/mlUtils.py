import numpy as np

def accuracy(y_true, y_hat):
    ones = []
    for i in range(0,len(y_hat)):
        b = y_hat[i]

        if y_true[i][b] == 1:
            ones.append(1.0)
        else:
            ones.append(0.0)
    return(np.mean(np.array(ones)))

#linear regression
class linear:
    def __init__(self):
        pass

    def train(self, X, Y, lamb):
        A = X.T.dot(X)
        A = A + lamb*np.identity(A.shape[0])
        b = X.T.dot(Y)
        W_hat = np.linalg.solve(A, b)
        return(W_hat)

    def predict(self, W, X):
        ret = W.T.dot(X.T)
        return(list(np.argmax(ret, axis=0)))

#logistic regression
class logistic:
    def __init__(self):
        pass
    
    def sigmoid(self, X, Y, W, b):
        denom_ = 1.0 + np.exp(-1.0 * Y * (b + X.dot(W))) #element-wise multiplication of labels Y
        return(1.0/denom_)

    def grad_w(self, X, Y, W, b, lam):
        n = float(X.shape[0])
        gw = (-1.0/n) * X.T.dot(Y*(1.0 - sigmoid(X,Y,W,b))) + 2*lam*W
        return(gw)

    def grad_b(self, X, Y, W, b):
        n = float(X.shape[0])
        gb = -1.0*(Y * (1.0 - sigmoid(X,Y,W,b)))
        return(gb)

    def log_likelihood(self, X, Y, W, b, lam):
        n = float(X.shape[0])
        reg = lam * np.power(np.linalg.norm(W),2)
        exp_ = np.log(1.0 + np.exp(-1.0 * Y * (b + X.dot(W))))
        out = (1.0/n) * np.sum(exp_) + reg
        return(out)

    def classify(self, X, W, b):
        return(np.sign(b + X.dot(W)))

    def reg_gradient_desc(self, X, Y, X_test, Y_test, step, lam, eps, maxiter):
        W = np.zeros((X.shape[1],1))
        b = 0.0

        #output data
        ll = log_likelihood(self, X, Y, W, b, lam)
        test_ll = log_likelihood(X_test, Y_test, W, b, lam)
        likelihoods = []
        test_likelihoods = []
        likelihoods.append(ll)
        test_likelihoods.append(test_ll)
        ll_old = np.power(10,5)
        accs = []
        test_accs = []
        it = 0

        #gradient descent
        while np.abs(ll_old - ll) > eps:
            if it > maxiter:
                print("maximum iterations reached")
                break
            W -= step_size * grad_w(X, Y, W, b, lam)
            b -= np.mean(step_size * grad_b(X, Y, W, b))

            ll_old = ll
            ll = log_likelihood(X, Y, W, b, lam)
            likelihoods.append(ll)
            test_ll = log_likelihood(X_test, Y_test, W, b, lam)
            test_likelihoods.append(test_ll)

            it += 1

            accs.append(accuracy(Y, classify(X, W, b)))
            test_accs.append(accuracy(Y_test, classify(X_test, W, b)))

        return(W, b, likelihoods, test_likelihoods, accs, test_accs)

    def stoch_reg_gradient_desc(self, X, Y, X_test, Y_test, step, lam, eps, maxiter, batch_size):
        W = np.zeros((X.shape[1],1))
        b = 0.0

        #output data
        ll = log_likelihood(X, Y, W, b, lam)
        test_ll = log_likelihood(X_test, Y_test, W, b, lam)
        likelihoods = []
        test_likelihoods = []
        likelihoods.append(ll)
        test_likelihoods.append(test_ll)
        ll_old = np.power(10,5)
        accs = []
        test_accs = []
        it = 0

        while True:
            if it > maxiter:
                print("maximum iterations reached")
                break
            batch = np.random.choice(X.shape[0], batch_size, replace=False)
            X_batch = X[batch, :]
            Y_batch = Y[batch, :]
            W -= step_size * grad_w(X_batch, Y_batch, W, b, lam)
            b -= np.mean(step_size * grad_b(X_batch, Y_batch, W, b))


            ll_old = ll
            ll = log_likelihood(X, Y, W, b, lam)
            likelihoods.append(ll)
            test_ll = log_likelihood(X_test, Y_test, W, b, lam)
            test_likelihoods.append(test_ll)

            it += 1

            accs.append(accuracy(Y, classify(X, W, b)))
            test_accs.append(accuracy(Y_test, classify(X_test, W, b)))

        return(W, b, likelihoods, test_likelihoods, accs, test_accs)
