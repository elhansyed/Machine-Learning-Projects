#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import numpy as np
import time
import pickle
import matplotlib.pyplot as plt
import numpy as np
import sklearn.linear_model as lin


    
####1####
#A
A = np.random.randint(10, size=(3,4))

#B
x = np.random.randint(10, size=(1,4))
 
#C
B = np.reshape(A, (6,2))

#D
C = x+A

#E
y = x.reshape(4)

#F
A[0] = y

#G
A[1] = A[2]-y

#H
print(A[:,[0,1,2]])

#I
print(A[[0,2],:])

#J
print(A.min())

#K
print(np.mean(A,axis=1))

#L
print(np.cos(A))

#M
print(np.sum(A, axis=0)**2)

#N
print(np.dot(A,A.T))

#O
print(np.dot(A,x.T).mean())


####2####

#A

def AATrans(A):
    N, M = len(A[0]), len(A)
    AAT = np.zeros((M,M))
    for i in range(M):
        for j in range(M):
            x = 0
            for k in range(N):
                x+= A[i][k] * A[j][k]
            AAT[i][j] += x
    return AAT
    
def myAdd(A, B):
    cols, rows = len(A[0]), len(A)
    for r in range(rows):
        for c in range(cols):
            A[r][c] += B[r][c]
    return A
    
def myfun(A, B):
    AAT = AATrans(A)
    AAT_added = myAdd(AAT,B)
    return AAT_added

def mymeasure(M,N):
    A = np.random.rand(M,N)
    B = np.random.rand(M,M)
    start_time_myfun = time.time()
    C1 = myfun(A,B)
    total_time_myfun = time.time() - start_time_myfun
    print("Execution time using myfun: " + str(total_time_myfun))
    start_time_numpy = time.time()
    C2 = np.add((np.matmul(A,A.T)),B)
    total_time_numpy = time.time() - start_time_numpy
    print("Execution time using numpy: " + str(total_time_numpy))
    mag = np.sum(np.abs(C1-C2))
    print('Magnitude of C1-C2: ' + str(mag))
    
#D
mymeasure(200,400)
mymeasure(1000,2000)

####4####
with open('data1.pickle','rb') as f:
    dataTrain, dataTest = pickle.load(f)

def get_xy(dataset):
    X_train = zip(*dataset)[0]
    y_train = zip(*dataset)[1]
    X_train, y_train = zip(*sorted(zip(X_train, y_train)))
    return X_train, y_train

X_train, y_train = get_xy(dataTrain)
X_test, y_test = get_xy(dataTest)
    
#A
def feature_matrix(X, alpha, beta):
    Z = np.zeros((len(alpha),len(X)))
    for i in range(len(alpha)):
        Z[i] = 1/(1+np.exp(beta*(alpha[i]-X)))
    return Z

#B
def plot_basis(alpha, beta):
    Z = feature_matrix(np.linspace(np.min(X_train), np.max(X_train), num=1000), alpha, beta)
    plt.plot(Z.T)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()

alpha = np.linspace(np.min(X_train), np.max(X_train), num=6)
beta = 2
figure_3b = plt.figure(figsize = (4,4))
figure_3b.suptitle("Question 4(b): 6 basis functions")
plot_basis(alpha, beta)

#C
def get_zaug(alpha, beta):
    Z = feature_matrix(X_train, alpha, beta)
    Zaug = np.vstack([np.ones(Z.shape[1]), Z]).T
    return Zaug

def errors(y, predicted):
    num = ((y-predicted)**2).sum()
    return num/len(y)

def predict(w0, w, alpha, beta, X):
    Z = feature_matrix(X, alpha, beta)
    summedZ = np.matmul(w, Z)
    predicted = w0*np.ones(Z.shape[1]) + summedZ
    return predicted

def my_fit(alpha, beta):
    Zaug = get_zaug(alpha, beta)
    fitted = np.linalg.lstsq(Zaug, y_train, rcond=None)        
    weights = fitted[0]
    w0, w = weights[0], weights[1:]        
    predicted_train = predict(w0, w, alpha, beta, X_train)
    predicted_test = predict(w0, w, alpha, beta, X_test)
    return w, w0, errors(y_train, predicted_train), errors(y_test, predicted_test)

#D
def plotY(w, w0, alpha, beta):
    tempX = np.linspace(np.min(X_test), np.max(X_test), num=1000)
    predicted = predict(w0, w, alpha, beta, tempX)
    plt.scatter(X_train, y_train, c = 'b')
    plt.plot(tempX, predicted, 'r')

#E
alpha = np.linspace(np.min(X_train), np.max(X_train), num=5)
beta = 1
figure_4e = plt.figure(figsize = (4,4))
figure_4e.suptitle("Question 4(e): the fitted function (5 basis functions)")
w, w0, err_train, err_test = my_fit(alpha, beta)
plotY(w, w0, alpha, beta)
print("Training error: " + str(err_train))
print("Testing error: " + str(err_test))

#F
alpha = np.linspace(np.min(X_train), np.max(X_train), num=12)
beta = 1
figure_4f = plt.figure(figsize = (4,4))
figure_4f.suptitle("Question 4(f): the fitted function (12 basis functions)")
w, w0, err_train, err_test = my_fit(alpha, beta)
plotY(w, w0, alpha, beta)
print("Training error: " + str(err_train))
print("Testing error: " + str(err_test))

#G
alpha = np.linspace(np.min(X_train), np.max(X_train), num=19)
beta = 1
figure_4g = plt.figure(figsize = (4,4))
figure_4g.suptitle("Question 4(g): the fitted function (19 basis functions)")
w, w0, err_train, err_test = my_fit(alpha, beta)
plotY(w, w0, alpha, beta)
print("Training error: " + str(err_train))
print("Testing error: " + str(err_test))



#####5#####
with open('data2.pickle','rb') as f:
    dataVal, dataTest = pickle.load(f)
    
X_test, y_test = get_xy(dataTest)
X_val, y_val = get_xy(dataVal)

#A
def myfit_reg(alpha, beta, gamma):
    Zaug = get_zaug(alpha, beta)
    ridge = lin.Ridge(gamma)
    ridge.fit(Zaug, y_train)
    w = ridge.coef_
    w = w[1:]
    w0 = ridge.intercept_
    predicted_train = predict(w0, w, alpha, beta, X_train)
    predicted_val = predict(w0, w, alpha, beta, X_val)
    return w, w0, errors(y_train, predicted_train), errors(y_val, predicted_val)

#B
gamma = 10e-9
alpha = np.linspace(np.min(X_train), np.max(X_train), num=19)
beta = 1
figure_5b = plt.figure(figsize = (4,4))
figure_5b.suptitle("Question 5(b): the fitted function, 19 basis functions, gamma 10^-9")
w, w0, err_train, err_val = myfit_reg(alpha, beta, gamma)
plotY(w, w0, alpha, beta)
print("Training error: " + str(err_train))
print("Validation error: " + str(err_val))

#C
gamma = 0
alpha = np.linspace(np.min(X_train), np.max(X_train), num=19)
beta = 1
figure_5c = plt.figure(figsize = (4,4))
figure_5c.suptitle("Question 5(c): the fitted function, 19 basis functions, gamma 0")
w, w0, err_train, err_val = myfit_reg(alpha, beta, gamma)
plotY(w, w0, alpha, beta)
print("Training error: " + str(err_train))
print("Validation error: " + str(err_val))

#D
def best_gamma(alpha, beta):
    gamma_list = np.exp(np.arange(-26, 5, 2))
    err_tra_list = []
    err_val_list = []
    weights = []
    
    best_error_val = 10000
    best_gamma = 0
    best_gamma_index=10000
    best_w0 = 0
    best_w = 0
    
    f = plt.figure(figsize = (10,10))
    
    for i in range(gamma_list.shape[0]):
        w, w0, err_tra, err_val = myfit_reg(alpha, beta, gamma_list[i])
        f.add_subplot(4, 4, i+1)
        f.suptitle("Question 5(d): best-fitting functions for log(gamma) = -26,-24, ..., 0, 2, 4")
        plotY(w, w0, alpha, beta)
        weights.append(w)
        err_tra_list.append(err_tra)
        err_val_list.append(err_val)
        if(err_val < best_error_val):
            best_error_val = err_val
            best_error_train = err_tra
            best_gamma = gamma_list[i]
            best_gamma_index = i
            best_w0 = w0
            best_w = w
            
    f2 = plt.figure(figsize = (4,4))
    f2.suptitle("Question 5(d): training and validation error")
    plt.xlabel("gamma")
    plt.ylabel("error")
    plt.semilogx(gamma_list,err_tra_list, 'b')
    plt.semilogx(gamma_list,err_val_list, 'r')
    
    f5 = plt.figure(figsize = (4,4))
    f5.suptitle("Question 5(d): weights for smallest gamma")
    plt.plot(weights[0])
    
    f4 = plt.figure(figsize = (4,4))
    f4.suptitle("Question 5(d): optimal weights")
    plt.plot(weights[best_gamma_index])
    
    f3 = plt.figure(figsize = (4,4))
    f3.suptitle("Question 5(d): weights for largest gamma")
    plt.plot(weights[-1])
    
    f6 = plt.figure(figsize = (4,4))
    f6.suptitle("Question 5(d): best-fitting function (gamma = "
                + str(best_gamma) + ")")
    plt.xlabel("x")
    plt.ylabel("predicted")
    plotY(best_w, best_w0, alpha, beta)
    
    print("Optimal gamma: " + str(best_gamma))
    print("Optimal w0: " + str(best_w0))
    
    predicted_test = predict(best_w0, best_w, alpha, beta, X_test)
    best_error_test = errors(y_test, predicted_test)
    
    print("Optimal training error: " + str(best_error_train))
    print("Optimal validation error: " + str(best_error_val))
    print("Optimal testing error: " + str(best_error_test))

alpha = np.linspace(np.min(X_train), np.max(X_train), num=19)
beta = 1
best_gamma(alpha, beta)

#####6#####

#C
def grad_reg(Z, T, w, w0, gamma):
    summedZ = np.matmul(w, Z)
    predicted = w0*np.ones(Z.shape[1]) + summedZ
    ones_row = np.ones((Z.shape[1],1)).T
    grad_w0 = 2*np.dot(ones_row,(predicted - y_train))
    grad_w = 2*(Z.dot(predicted - y_train) + gamma*w.sum())
    return grad_w, grad_w0
    
#D
def myfit_reg_gd(alpha, beta, gamma, lrate):
    epochs = 3000000
    errors_list = []
    epoch_graph_list = [1, 6, 36, 216, 1296, 7776, 46656, 279936, 1679616]
    graph_loc = 1
    w = np.random.randn(alpha.shape[0])
    w0 = np.random.randn()
    f = plt.figure(figsize = (10,10))

    for epoch in range(1, epochs+1):
        predicted_train = predict(w0, w, alpha, beta, X_train)
        predicted_test = predict(w0, w, alpha, beta, X_test)
        errors_list.append([errors(y_train, predicted_train), errors(y_test, predicted_test)])
        Z = feature_matrix(X_train, alpha, beta)
        grad_w, grad_w0 = grad_reg(Z, y_train, w, w0, gamma)
        w = w - (lrate*grad_w)
        w0 = w0 - (lrate*grad_w0)
        
        if(epoch in epoch_graph_list):
            f.add_subplot(3,3,graph_loc)
            f.suptitle("Question 6: fitted function as iterations increase")
            plotY(w, w0, alpha, beta)
            graph_loc +=1
        if(epoch == epochs):
            f2 = plt.figure(figsize = (4,4))
            f2.add_subplot(1,1,1)
            f2.suptitle("Question 6: fitted function")
            plotY(w, w0, alpha, beta)
            
    f3 = plt.figure(figsize = (4,4))
    f3.add_subplot(1,1,1)
    f3.suptitle('Question 6: training and test error v.s. iterations')        
    train_errors, test_errors = zip(*errors_list)[0], zip(*errors_list)[1]
    plt.xlabel("number of iterations")
    plt.ylabel("error")
    plt.plot(np.arange(epochs), train_errors, 'b')
    plt.plot(np.arange(epochs), test_errors, 'r')
    
    f4 = plt.figure(figsize = (4,4))
    f4.add_subplot(1,1,1)
    f4.suptitle('Question 6: training and test error v.s. iterations (log scale)')        
    plt.xlabel("number of iterations")
    plt.ylabel("error (log)")
    plt.semilogx(np.arange(epochs), train_errors, 'b')
    plt.semilogx(np.arange(epochs), test_errors, 'r')

    f5 = plt.figure(figsize = (4,4))
    f5.add_subplot(1,1,1)
    f5.suptitle('Question 6: last 1,000,000 training errors')        
    plt.plot(np.arange(2000000,epochs+1), train_errors[2000000-1:])

    print('Final training error: ' + str(train_errors[-1]))
    print('Final testing error: ' + str(test_errors[-1]))
    
    mfg_w, mfg_w0, mfg_errors_train, _ = myfit_reg(alpha, beta, gamma)
    predicted_test = predict(mfg_w0, mfg_w, alpha, beta, X_test)
    mfg_errors_test = errors(y_test, predicted_test)
    print('myfit_reg train: ' + str(mfg_errors_train))
    print('myfit_reg test: ' + str(mfg_errors_test))
    
    delta_diff = train_errors[-1] - mfg_errors_train
    print('Delta difference (myfit_reg_gd & myfit_reg): ' + str(delta_diff))
    print('Learning Rate: ' + str(lrate))
    
alpha = np.linspace(np.min(X_train), np.max(X_train), num=19)
beta = 1
gamma = 0.001
lrate = 1e-3
