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

#converts one hot encoding to a one v. all ting
#oneHot is the nxd matrix (n=number of samples and d=number of labels)
#col is the column that should be +1
#all other columns will be -1
#will return a nx1 vector 
def oneHotToLogLabels(oneHot,col):
    curIndex = col
    newY = np.empty((1))
    for i in range(0,len(oneHot)):
        if np.nonzero(oneHot[i])[0] == curIndex:
            tmp = np.array([1.0])
            newY = np.vstack((newY,tmp))
        else:
            tmp = np.array([-1.0])
            newY = np.vstack((newY,tmp))
    newY = np.delete(newY, (0), axis=0)
    return(newY)

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

    def log_accuracy(self, y_true, y_hat):
        ones = []
        for i in range(len(y_true)):
            if int(y_true[i]) == int(y_hat[i]):
                ones.append(1.0)
            else:
                ones.append(0.0)
        return(np.mean(np.array(ones)))
    
    def sigmoid(self, X, Y, W, b):
        denom_ = 1.0 + np.exp(-1.0 * Y * (b + X.dot(W))) #element-wise multiplication of labels Y
        return(1.0/denom_)

    def grad_w(self, X, Y, W, b, lam):
        n = float(X.shape[0])
        gw = (-1.0/n) * X.T.dot(Y*(1.0 - self.sigmoid(X,Y,W,b))) + 2*lam*W
        return(gw)

    def grad_b(self, X, Y, W, b):
        n = float(X.shape[0])
        gb = -1.0*(Y * (1.0 - self.sigmoid(X,Y,W,b)))
        return(gb)

    def log_likelihood(self, X, Y, W, b, lam):
        n = float(X.shape[0])
        reg = lam * np.power(np.linalg.norm(W),2)
        exp_ = np.log(1.0 + np.exp(-1.0 * Y * (b + X.dot(W))))
        out = (1.0/n) * np.sum(exp_) + reg
        return(out)

    def classify(self, X, W, b):
        return(np.sign(b + X.dot(W)))

    def reg_gradient_desc(self, X, Y, X_test, Y_test, step_size, lam, eps, maxiter):
        W = np.zeros((X.shape[1],1))
        b = 0.0

        #output data
        ll = self.log_likelihood(X, Y, W, b, lam)
        test_ll = self.log_likelihood(X_test, Y_test, W, b, lam)
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
            W -= step_size * self.grad_w(X, Y, W, b, lam)
            b -= np.mean(step_size * self.grad_b(X, Y, W, b))

            ll_old = ll
            ll = self.log_likelihood(X, Y, W, b, lam)
            likelihoods.append(ll)
            test_ll = self.log_likelihood(X_test, Y_test, W, b, lam)
            test_likelihoods.append(test_ll)

            it += 1

            accs.append(self.log_accuracy(Y, self.classify(X, W, b)))
            test_accs.append(self.log_accuracy(Y_test, self.classify(X_test, W, b)))

        return(W, b, likelihoods, test_likelihoods, accs, test_accs)

    def stoch_reg_gradient_desc(self, X, Y, X_test, Y_test, step_size, lam, eps, maxiter, batch_size):
        W = np.zeros((X.shape[1],1))
        b = 0.0

        #output data
        ll = self.log_likelihood(X, Y, W, b, lam)
        test_ll = self.log_likelihood(X_test, Y_test, W, b, lam)
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
            W -= step_size * self.grad_w(X_batch, Y_batch, W, b, lam)
            b -= np.mean(step_size * self.grad_b(X_batch, Y_batch, W, b))


            ll_old = ll
            ll = self.log_likelihood(X, Y, W, b, lam)
            likelihoods.append(ll)
            test_ll = self.log_likelihood(X_test, Y_test, W, b, lam)
            test_likelihoods.append(test_ll)

            it += 1

            accs.append(self.log_accuracy(Y, self.classify(X, W, b)))
            test_accs.append(self.log_accuracy(Y_test, self.classify(X_test, W, b)))

        return(W, b, likelihoods, test_likelihoods, accs, test_accs)
