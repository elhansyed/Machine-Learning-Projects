#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 18:25:41 2019

@author: syed
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import bonnerlib2
from sklearn.neural_network import MLPClassifier
import pickle

#1
#a
def gen_data(mu0, mu1, cov0, cov1, N0, N1):
    x0 = np.random.multivariate_normal(mu0, [[1,cov0],[cov0,1]], N0)
    x1 = np.random.multivariate_normal(mu1, [[1,cov1],[cov1,1]], N1)
    X = np.hstack((zip(*x0),zip(*x1)))
    t = np.hstack((np.zeros(N0), np.ones(N1)))
    return shuffle(X.transpose(),t)


Xtrain,Ttrain = gen_data(mu0=(1, 1), mu1=(2, 2), cov0=0, cov1=0.9, N0=1000, N1=500)
Xtest, Ttest = gen_data(mu0=(1, 1), mu1=(2, 2), cov0=0, cov1=0.9, N0=10000, N1=5000)

#b
print('\n')
print('Question 1(b).')
print('-------------')

def plot_train(Xtrain, Ttrain): 
    X,y = Xtrain.T
    colors = []
    for i in range(len(Ttrain)):
        if(Ttrain[i]==0):
            colors.append('red')
        else: 
            colors.append('blue')
    plt.scatter(X,y,c=colors,s=2)
    
def precision(train, target):
    true_pos = sum((train == 1) & (target == 1))
    false_pos = sum((train == 1) & (target == 0))
    return (float(true_pos)/float(false_pos+true_pos))

def recall(train, target):
    true_pos = sum((train == 1) & (target == 1))
    false_neg = sum((train == 0) & (target == 1))
    return (float(true_pos)/float(true_pos + false_neg))

nn = MLPClassifier(hidden_layer_sizes = 1, activation = 'logistic',
                   solver = 'sgd', learning_rate_init = 0.01, max_iter = 1000, 
                   tol = 0.0000001)

nn.fit(Xtrain, Ttrain)
prediction = nn.predict(Xtest) 

fig_1b = plt.figure()
fig_1b.suptitle('Question 1(b): Neural net with 1 hidden unit')
plt.xlim(-3,6)
plt.ylim(-3,6)
plot_train(Xtrain,Ttrain)
bonnerlib2.dfContour(nn)

print('Test accuracy: ' + str(nn.score(Xtest, Ttest)))
print('Test precision: ' + str(precision(prediction, Ttest)))
print('Test recall: ' + str(recall(prediction, Ttest)))


#c
print('\n')
print('Question 1(c).')
print('-------------')

def accuracy(train, target): 
    correct = sum(train == target)
    return float(correct)/float(len(target))

def best_acc_recall_prec_print(accuracy, recall, precision):
    print('Best test accuracy: ' + str(accuracy))
    print('Best test recall: ' + str(recall))
    print('Best test precision: ' + str(precision))


def neural_net_trial(Xtrain, Ttrain, hidden_units, question_part):
    fig = plt.figure(figsize = (4,3))
    plt.suptitle('Question 1(' + str(question_part) + '): Neural nets with ' +
                 str(hidden_units) +' hidden units')
    best_acc = 0
    nn_trial = MLPClassifier(hidden_layer_sizes = (hidden_units,), activation = 'logistic',
                   solver = 'sgd', learning_rate_init = 0.01, max_iter = 1000, 
                   tol = 0.0000001)
    for i in range(12):
        nn_trial.fit(Xtrain, Ttrain)
        test_acc = nn_trial.score(Xtest, Ttest)
      #  print(test_acc)
        fig.add_subplot(4,3,i+1)
        plt.xlim(-3,6)
        plt.ylim(-3,6)
        plot_train(Xtrain,Ttrain)
        bonnerlib2.dfContour(nn_trial)
        if(test_acc > best_acc):
            best_acc = test_acc
            best_nn = nn_trial
    best_predict = best_nn.predict(Xtest)
    best_precision = precision(best_predict, Ttest)
    best_recall = recall(best_predict, Ttest)
    return best_nn, best_acc, best_precision, best_recall

def plt_nn(nn, question_part, num_layers):
    fig = plt.figure()
    plt.suptitle('Question 1(' + str(question_part) + '): Best neural net with ' 
              + str(num_layers) + ' hidden units')
    plot_train(Xtrain,Ttrain)
    bonnerlib2.dfContour(nn)
    
best_nn_trial, best_nn_acc, best_nn_prec, best_nn_rec = neural_net_trial(Xtrain, Ttrain, 2, 'c')
plt_nn(best_nn_trial, 'c', 2)
best_acc_recall_prec_print(best_nn_acc, best_nn_rec, best_nn_prec)
        
#d
print('\n')
print('Question 1(d).')
print('-------------')
best_nn_trial_d, best_nn_acc_d, best_nn_prec_d, best_nn_rec_d = neural_net_trial(Xtrain, Ttrain, 3, 'd')
plt_nn(best_nn_trial_d, 'd', 3)
best_acc_recall_prec_print(best_nn_acc_d, best_nn_rec_d, best_nn_prec_d)

#e
print('\n')
print('Question 1(e).')
print('-------------')
best_nn_trial_e, best_nn_acc_e, best_nn_prec_e, best_nn_rec_e = neural_net_trial(Xtrain, Ttrain, 4, 'e')
plt_nn(best_nn_trial_e, 'e', 4)
best_acc_recall_prec_print(best_nn_acc_e, best_nn_rec_e, best_nn_prec_e)


#3
with open('mnist.pickle','rb') as f:
    Xtrain,Ytrain,Xtest,Ytest = pickle.load(f)
    
Xvalidation, Yvalidation = Xtrain[:10000], Ytrain[:10000]
Xtrain_reduced, Ytrain_reduced = Xtrain[10000:20000], Ytrain[10000:20000]

#a
print('\n')
print('Question 3(a).')
print('-------------')

def cross_entropy_3(output, y, classes=10):
    y = np.eye(classes)[y]
    return -np.sum(y*output)

best_val_acc = 0
for i in range(10):
    nn_3a = MLPClassifier(hidden_layer_sizes = (30,), activation = 'logistic',
                           solver = 'sgd', batch_size = 100, learning_rate_init = 1, max_iter = 100, 
                           tol = 0.0000001)
    nn_3a.fit(Xtrain_reduced, Ytrain_reduced)
    val_acc_3a = nn_3a.score(Xvalidation, Yvalidation)
    if(val_acc_3a > best_val_acc):
        best_model = nn_3a
        best_val_acc = val_acc_3a
    print('Validation accuracy net '+ str(i+1) + ': ' + str(val_acc_3a))

test_pred_bm = best_model.predict_log_proba(Xtest)
print('Best Net Validation Accuracy: ' + str(best_model.score(Xvalidation, Yvalidation)))
print('Best Net Test Accuracy: ' + str(best_model.score(Xtest, Ytest)))
print('Best Net Cross Entropy: ' + str(cross_entropy_3(test_pred_bm, Ytest)))
print('Learning Rate: 1')

#b
print('\n')
print('Question 3(b).')
print('-------------')

best_val_acc = 0
for i in range(10):
    nn_3b = MLPClassifier(hidden_layer_sizes = (30,), activation = 'logistic',
                           solver = 'sgd', batch_size = Xtrain_reduced.shape[0], learning_rate_init = 1, max_iter = 100, 
                           tol = 0.0000001)
    nn_3b.fit(Xtrain_reduced, Ytrain_reduced)
    val_acc_3b = nn_3b.score(Xvalidation, Yvalidation)
    if(val_acc_3b > best_val_acc):
        best_model = nn_3b
        best_val_acc = val_acc_3b
    print('Validation accuracy net '+ str(i+1) + ': ' + str(val_acc_3b))

test_pred_bm = best_model.predict_log_proba(Xtest)
print('Best Net Validation Accuracy: ' + str(best_model.score(Xvalidation, Yvalidation)))
print('Best Net Test Accuracy: ' + str(best_model.score(Xtest, Ytest)))
print('Best Net Cross Entropy: ' + str(cross_entropy_3(test_pred_bm, Ytest)))
print('Learning Rate: 1')

#c
print('\n')
print('Question 3(c).')
print('-------------')
nn_3c = MLPClassifier(hidden_layer_sizes = (30,), activation = 'logistic',
                           solver = 'sgd', batch_size = Xtrain_reduced.shape[0], learning_rate_init = 1, max_iter = 50, 
                           tol = 0.0000001)
nn_3c.fit(Xtrain_reduced, Ytrain_reduced)

test_pred_3c = nn_3c.predict_log_proba(Xtest)

train_acc_3c = nn_3c.score(Xtrain_reduced, Ytrain_reduced)
test_acc_3c = nn_3c.score(Xtest, Ytest)
cross_ent_3c = cross_entropy_3(test_pred_3c, Ytest)

print('Training Accuracy (50 Iterations): ' + str(train_acc_3c))
print('Test Accuracy (50 Iterations): ' + str(test_acc_3c))
print('Cross Entropy (50 Iterations): ' + str(cross_ent_3c))

nn_3c_2 = MLPClassifier(hidden_layer_sizes = (30,), activation = 'logistic',
                           solver = 'sgd', batch_size = Xtrain_reduced.shape[0], learning_rate_init = 1, max_iter = 200, 
                           tol = 0.0000001)
nn_3c_2.fit(Xtrain_reduced, Ytrain_reduced)

test_pred_3c_2 = nn_3c_2.predict_log_proba(Xtest)

train_acc_3c_2 = nn_3c_2.score(Xtrain_reduced, Ytrain_reduced)
test_acc_3c_2 = nn_3c_2.score(Xtest, Ytest)
cross_ent_3c_2 = cross_entropy_3(test_pred_3c_2, Ytest)

print('Training Accuracy (200 Iterations): ' + str(train_acc_3c_2))
print('Test Accuracy (200 Iterations): ' + str(test_acc_3c_2))
print('Cross Entropy (200 Iterations): ' + str(cross_ent_3c_2))

#d
def mxb(X, w, w0):
    return np.dot(X, w) + w0
    
def sigmoid(z):
    return 1/(1+np.exp(-z))
    
def softmax(z):
    num = np.exp(z)
    den = np.sum(np.exp(z), axis=0)
    return num/den
    
def cross_entropy(output, y):
    classes = output.shape[1]
    one_hot = np.eye(classes)[y]
    return np.sum(-np.sum(one_hot*np.ma.log(output).filled(0)))

print('\n')
print('Question 3(d).')
print('-------------')
hidden_units = 30
feature_nums = Xtrain_reduced.shape[1]
N = feature_nums
output_nums = 10

best_acc = 0
for i in range(1,10+1):
    V = np.random.normal(size=(feature_nums, hidden_units))
    v0 = np.zeros((1, hidden_units))
    W = np.random.normal(size=(hidden_units, output_nums))
    w0 = np.zeros((1, output_nums))
    lr = 1e-6/N
    epochs = 100
    
    for epoch in range(epochs):
        #forward pass
        U = mxb(Xtrain_reduced, V, v0)
        H = sigmoid(U)
        Z = mxb(H, W, w0)
        O = softmax(Z)
        C = cross_entropy(O, Ytrain_reduced)
        
        #backward pass
        dCdZ = O - Ytrain_reduced.reshape(-1,1)
        dCdW = np.dot(H.T, dCdZ)
        dCdw0 = np.sum(dCdZ, axis=0).reshape(1,-1)
        dCdH = np.dot(dCdZ, W.T)
        dCdU = np.dot(np.dot(H, (1-H).T), dCdH)
        dCdV = np.dot(Xtrain_reduced.T, dCdU)
        dCdv0 = np.sum(dCdU, axis=0).reshape(1,-1)
        
        #updating weights
        W -= lr*dCdW
        w0 -= lr*dCdw0
        V -= lr*dCdV
        v0 -= lr*dCdv0
    U = mxb(Xvalidation, V, v0)
    H = sigmoid(U)
    Z = mxb(H, W, w0)
    O = softmax(Z)  
    val_acc = accuracy(np.argmax(O,axis=1), Yvalidation)*100
    print('Net ' + str(i) + ' Validation Accuracy: ' + str(val_acc))
    if(val_acc>best_acc):
        best_acc = val_acc
        best_V = V
        best_v0 = v0
        best_W = W
        best_w0 = w0
        
        
##Best Results
U = mxb(Xtest, best_V, best_v0)
H = sigmoid(U)
Z = mxb(H, best_W, best_w0)
O = softmax(Z)
C = cross_entropy(O, Ytest)
test_acc = accuracy(np.argmax(O,axis=1), Ytest)*100
print('Best Trained Net Results')
print('Validation accuracy: ' + str(best_acc))
print('Test accuracy: ' + str(test_acc))
print('Cross Entropy: ' + str(C))
print('Learning rate: ' + str(lr))

#e
print('\n')
print('Question 3(e).')
print('-------------')
hidden_units = 30
feature_nums = Xtrain.shape[1]
output_nums = 10
batch_size = 100
N = batch_size

best_acc = 0
for k in range(1,10+1):
    V = np.random.normal(size=(feature_nums, hidden_units))
    v0 = np.zeros((1, hidden_units))
    W = np.random.normal(size=(hidden_units, output_nums))
    w0 = np.zeros((1, output_nums))
    lr = 1e-6/N
    epochs = 100
    #Epoch begin
    for epoch in range(epochs):
        Xtrain, Ytrain = shuffle(Xtrain, Ytrain)
        for i in xrange(0, Xtrain.shape[0], batch_size):
            #Mini batch creation
            X = Xtrain[np.arange(i,i+batch_size)]
            T = Ytrain[np.arange(i,i+batch_size)]
            #forward pass
            U = mxb(X, V, v0)
            H = sigmoid(U)
            Z = mxb(H, W, w0)
            O = softmax(Z)
            C = cross_entropy(O, T)
            
            #backward pass
            dCdZ = O - T.reshape(-1,1)
            dCdW = np.dot(H.T, dCdZ)
            dCdw0 = np.sum(dCdZ, axis=0).reshape(1,-1)
            dCdH = np.dot(dCdZ, W.T)
            dCdU = np.dot(np.dot(H, (1-H).T), dCdH)
            dCdV = np.dot(X.T, dCdU)
            dCdv0 = np.sum(dCdU, axis=0).reshape(1,-1)
            
            #updating weights
            W -= lr*dCdW
            w0 -= lr*dCdw0
            V -= lr*dCdV
            v0 -= lr*dCdv0
    U = mxb(Xvalidation, V, v0)
    H = sigmoid(U)
    Z = mxb(H, W, w0)
    O = softmax(Z)  
    val_acc = accuracy(np.argmax(O,axis=1), Yvalidation)*100
    print('Net ' + str(k) + ' Validation Accuracy: ' + str(val_acc))
    if(val_acc>best_acc):
        best_V = V
        best_v0 = v0
        best_W = W
        best_w0 = w0

##Best Results
U = mxb(Xtest, best_V, best_v0)
H = sigmoid(U)
Z = mxb(H, best_W, best_w0)
O = softmax(Z)
C = cross_entropy(O, Ytest)
test_acc = accuracy(np.argmax(O,axis=1), Ytest)*100
print('Best Trained Net Results')
print('Validation accuracy: ' + str(best_acc))
print('Test accuracy: ' + str(test_acc))
print('Cross Entropy: ' + str(C))
print('Learning rate: ' + str(lr))


#f
print('\n')
print('Question 3(f).')
print('-------------')

with open('mnist.pickle','rb') as f:
    Xtrain,Ytrain,Xtest,Ytest = pickle.load(f)
    
hidden_units = 100
feature_nums = Xtrain.shape[1]
output_nums = 10
batch_size = 100
N = batch_size

V = np.random.normal(size=(feature_nums, hidden_units))
v0 = np.zeros((1, hidden_units))
W = np.random.normal(size=(hidden_units, output_nums))
w0 = np.zeros((1, output_nums))
lr = 1e-6/N
epochs = 100
#Epoch begin
for epoch in range(1,epochs+1):
    Xtrain, Ytrain = shuffle(Xtrain,Ytrain)
    for i in xrange(0, Xtrain.shape[0], batch_size):
        #Mini batch creation
        X = Xtrain[np.arange(i,i+batch_size)]
        T = Ytrain[np.arange(i,i+batch_size)]
        #forward pass
        U = mxb(X, V, v0)
        H = sigmoid(U)
        Z = mxb(H, W, w0)
        O = softmax(Z)
        C = cross_entropy(O, T)
        
        #backward pass
        dCdZ = O - T.reshape(-1,1)
        dCdW = np.dot(H.T, dCdZ)
        dCdw0 = np.sum(dCdZ, axis=0).reshape(1,-1)
        dCdH = np.dot(dCdZ, W.T)
        dCdU = np.dot(np.dot(H, (1-H).T), dCdH)
        dCdV = np.dot(X.T, dCdU)
        dCdv0 = np.sum(dCdU, axis=0).reshape(1,-1)
        
        #updating weights
        W -= lr*dCdW
        w0 -= lr*dCdw0
        V -= lr*dCdV
        v0 -= lr*dCdv0
    if(epoch%10==0):
        U = mxb(Xtest, V, v0)
        H = sigmoid(U)
        Z = mxb(H, W, w0)
        O = softmax(Z)
        print('Epoch ' + str(epoch) + ' test accuracy: ' + str(accuracy(np.argmax(O,axis=1), Ytest)*100))

U = mxb(X, V, v0)
H = sigmoid(U)
Z = mxb(H, W, w0)
O = softmax(Z)
print('Final training accuracy: ' + str(accuracy(np.argmax(O,axis=1), T)*100))
U = mxb(Xtest, V, v0)
H = sigmoid(U)
Z = mxb(H, W, w0)
O = softmax(Z)
print('Final test accuracy: ' + str(accuracy(np.argmax(O,axis=1), Ytest)*100))
print('Final cross entropy: ' + str(cross_entropy(O, Ytest)))


#g
#DOES NOT RUN
print('\n')
print('Question 3(g).')
print('-------------')
hidden_units = 100
X,T = Xtrain,Ytrain
feature_nums = X.shape[1]
N = feature_nums
output_nums = 10
V = np.random.normal(size=(feature_nums, hidden_units))
v0 = np.zeros((1, hidden_units))
W = np.random.normal(size=(hidden_units, output_nums))
w0 = np.zeros((1, output_nums))
lr = 1e-6/N
epochs = 1
#Epoch begin
for epoch in range(1,epochs+1):
    #forward pass
    U = mxb(X, V, v0)
    H = sigmoid(U)
    Z = mxb(H, W, w0)
    O = softmax(Z)
    C = cross_entropy(O, T)
    
    #backward pass
    dCdZ = O - T.reshape(-1,1)
    dCdW = np.dot(H.T, dCdZ)
    dCdw0 = np.sum(dCdZ, axis=0).reshape(1,-1)
    dCdH = np.dot(dCdZ, W.T)
    dCdU = np.dot(np.dot(H, (1-H).T), dCdH)
    dCdV = np.dot(X.T, dCdU)
    dCdv0 = np.sum(dCdU, axis=0).reshape(1,-1)
    
    #updating weights
    W -= lr*dCdW
    w0 -= lr*dCdw0
    V -= lr*dCdV
    v0 -= lr*dCdv0
    if(epoch%10==0):
        U = mxb(Xtest, V, v0)
        H = sigmoid(U)
        Z = mxb(H, W, w0)
        O = softmax(Z)
        print('Epoch ' + str(epoch) + ' test accuracy: ' + str(accuracy(np.argmax(O,axis=1), Ytest)*100))


U = mxb(X, V, v0)
H = sigmoid(U)
Z = mxb(H, W, w0)
O = softmax(Z)
print('Final training accuracy: ' + str(accuracy(np.argmax(O,axis=1), T)*100))
U = mxb(Xtest, V, v0)
H = sigmoid(U)
Z = mxb(H, W, w0)
O = softmax(Z)
print('Final test accuracy: ' + str(accuracy(np.argmax(O,axis=1), Ytest)*100))
print('Final cross entropy: ' + str(cross_entropy(O, Ytest)))


