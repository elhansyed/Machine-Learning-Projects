#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import numpy as np 
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
import bonnerlib2
import pickle
import random
import time


#1
#a
def gen_data(mu0, mu1, cov0, cov1, N0, N1):
    x0 = np.random.multivariate_normal(mu0, [[1,cov0],[cov0,1]], N0)
    x1 = np.random.multivariate_normal(mu1, [[1,cov1],[cov1,1]], N1)
    X = np.hstack((zip(*x0),zip(*x1)))
    t = np.hstack((np.zeros(N0), np.ones(N1)))
    return shuffle(X.transpose(),t)
    
    
#b
X,t = gen_data(mu0=(1, 1), mu1=(2, 2), cov0=0, cov1=-0.9, N0=10000, N1=5000)


X,y = X.T

#c 
figure_1c = plt.figure(figsize = (4,4))
figure_1c.suptitle('Question 1(c): sample cluster data')
plt.xlim(-3,6)
plt.ylim(-3,6)
colors = []
for i in range(len(t)):
    if(t[i]==0):
        colors.append('red')
    else: 
        colors.append('blue')
plt.scatter(X,y,c=colors,s=2)

#2
#a
X, t = gen_data(mu0=(1, 1), mu1=(2, 2), cov0=0, cov1=-0.9, N0=1000, N1=500)

#b
model = LogisticRegression()
model.fit(X,t)
print('w: ' + str(model.coef_[0]) + ', w0: ' + str(model.intercept_[0]))

#c
def my_predict(X, w, w0, threshold=0.5):
    z = np.dot(X,w.transpose()) + w0
    sigmoid = 1/(1+np.exp(-z))
    return 1*(sigmoid >= threshold)

def my_score(pred, y):
    correct = 1*(pred == y).sum()
    return float(correct)/float(len(y))
    
accuracy1 = model.score(X,t)

predicted = my_predict(X, model.coef_[0], model.intercept_[0])
accuracy2 = my_score(predicted, t)

print('accuracy1: ' + str(accuracy1))
print('accuracy2: ' + str(accuracy2))
print('difference between accuracy1 and accuracy2: ' + str(accuracy2-accuracy1))

#d
figure_2d = plt.figure(figsize = (4,4))
figure_2d.suptitle('Question 2(d):  training data and decision boundary')
plt.xlim(-3,6)
plt.ylim(-3,6)

weights = model.coef_[0]
x_pts = np.linspace(-3,6)
a = -weights[0]/weights[1]
intercept = model.intercept_[0]
y_pts = a*x_pts - intercept/weights[1]

plt.plot(x_pts,y_pts,c='black')

X, y = X.T
colors = []
for i in range(len(t)):
    if(t[i]==0):
        colors.append('red')
    else: 
        colors.append('blue')
plt.scatter(X,y,c=colors,s=2)

#e
figure_2e = plt.figure(figsize = (4,4))
figure_2e.suptitle('Question 2(e): three contours')
plt.xlim(-3,6)
plt.ylim(-3,6)

y_pts1 = a*x_pts - (intercept + np.log((1/0.5) - 1))/weights[1]
y_pts2 = a*x_pts - (intercept + np.log((1/0.6) - 1))/weights[1]
y_pts3 = a*x_pts - (intercept + np.log((1/0.05) - 1))/weights[1]

plt.plot(x_pts,y_pts1, c = 'black')
plt.plot(x_pts,y_pts2, c = 'blue')
plt.plot(x_pts,y_pts3, c = 'red')

plt.scatter(X,y,c=colors,s=2)

#f
X, t = gen_data(mu0=(1, 1), mu1=(2, 2), cov0=0, cov1=-0.9, N0=10000, N1=5000)

#g
model = LogisticRegression()
model.fit(X,t)

w = model.coef_[0]
w0 = model.intercept_[0]

def precision(train, target):
    true_pos = sum((train == 1) & (target == 1))
    false_pos = sum((train == 1) & (target == 0))
    return (float(true_pos)/float(false_pos+true_pos))

def recall(train, target):
    true_pos = sum((train == 1) & (target == 1))
    false_neg = sum((train == 0) & (target == 1))
    return (float(true_pos)/float(true_pos + false_neg))

predict_1 = my_predict(X, w, w0, threshold=0.05)
predict_2 = my_predict(X, w, w0, threshold=0.5)
predict_3 = my_predict(X, w, w0, threshold=0.6)

precision_1 = precision(predict_1, t)
precision_2 = precision(predict_2, t)
precision_3 = precision(predict_3, t)

recall_1 = recall(predict_1,t)
recall_2 = recall(predict_2,t)
recall_3 = recall(predict_3,t)

print('P(C = 1|x) = 0.05, precision: ' + str(precision_1) + ', recall: ' + str(recall_1))
print('P(C = 1|x) = 0.5, precision: ' + str(precision_2) + ', recall: ' + str(recall_2))
print('P(C = 1|x) = 0.6, precision: ' + str(precision_3) + ', recall: ' + str(recall_3))

    
#4
#a
X, t = gen_data(mu0=(1, 1), mu1=(2, 2), cov0=0, cov1=-0.9, N0=1000, N1=500)
X_repeat, t_repeat = X,t ## used in 4e

clf = QuadraticDiscriminantAnalysis()
model = clf.fit(X,t)
accuracy4a = clf.score(X,t)
print('accuracy: ' + str(accuracy4a))

fig_4a = plt.figure()
fig_4a.suptitle('Question 4(a): Decision boundary and contours')
X,y = X.T
colors = []
for i in range(len(t)):
    if(t[i]==0):
        colors.append('red')
    else: 
        colors.append('blue')
plt.scatter(X,y,c=colors,s=2)
bonnerlib2.dfContour(clf)

#c
Xc, tc = gen_data(mu0=(1, 1), mu1=(2, 2), cov0=0, cov1=0.9, N0=1000, N1=500)
clf2 = QuadraticDiscriminantAnalysis()
model = clf2.fit(Xc,tc)
accuracy = clf2.score(Xc,tc)
print('accuracy: ' + str(accuracy))

fig_4c = plt.figure()
fig_4c.suptitle('Question 4(c): Decision boundary and contours')
Xc,yc = Xc.T
colors = []
for i in range(len(t)):
    if(t[i]==0):
        colors.append('red')
    else: 
        colors.append('blue')
plt.scatter(Xc,yc,c=colors,s=2)
bonnerlib2.dfContour(clf2)

#d
Xd_train, td_train = gen_data(mu0=(1, 1), mu1=(2, 2), cov0=0, cov1=0.9, N0=1000, N1=500)
Xd_test, td_test = gen_data(mu0=(1, 1), mu1=(2, 2), cov0=0, cov1=0.9, N0=10000, N1=50000)

clf3 = QuadraticDiscriminantAnalysis()
model = clf3.fit(Xd_train,td_train)
accuracy = clf3.score(Xd_train,td_train)
print('accuracy: ' + str(accuracy))

fig_4d = plt.figure()
fig_4d.suptitle('Question 4(d): Decision boundary and contours')
Xd_test,yd_test = Xd_test.T
colors = []
for i in range(len(td_test)):
    if(td_test[i]==0):
        colors.append('red')
    else: 
        colors.append('blue')
plt.scatter(Xd_test,yd_test,c=colors,s=2)
bonnerlib2.dfContour(clf3)

#e
X_test, y_test = gen_data(mu0=(1, 1), mu1=(2, 2), cov0=0, cov1=-0.9, N0=1000, N1=500)

def my_mean(feat):
    return np.sum(feat)/len(feat)

def extract_label_indices(y, label):
    return np.where(y==label)[0]

def get_feats(X):
    return X.T

def get_data_mean(X):
    feat0, feat1 = get_feats(X)
    return np.array([my_mean(feat0), my_mean(feat1)])

def my_cov(X, mu):
    N = len(X)
    Y = X-mu.T
    return np.dot(Y.T, Y)/(N) #change to N

def my_prior(X, classk):
    return float(len(classk))/float(len(X))

def my_mahalanobis(x, mu, cov):
    Y = x-mu
    inv = np.linalg.inv(cov)
    return np.max(np.dot(np.dot(Y,inv), Y.T), axis=1)

def p_xy(d, cov, mu, x):
    den = (-d/2)*np.log(2*np.pi) - (1/2)*(np.log(np.linalg.det(cov))) #((2*np.pi)**(d/2))*np.sqrt(np.linalg.det(cov))
    num = -my_mahalanobis(x, mu, cov) #np.exp(-my_mahalanobis(x, mu, cov))
    return num/den

def p_y(prior):
    return 1-prior#(prior**y)*((1-prior)**(1-y))

def p_yx(X, cov, mu, x, prior, y):
    return p_xy(X, cov, mu, x)*np.log(p_y(prior))

def predict(class0, class1, cov0, cov1, mu0, mu1, prior0, prior1, x):
    d = len(class0) + len(class1)
    p0 = p_yx(d, cov0, mu0, x, prior0, 0)
    p1 = p_yx(d, cov1, mu1, x, prior1, 1)
    probs = np.array([p0, p1])
    return np.argmax(probs.T,axis=1)

def myGDA(Xtrain,Ttrain,Xtest,Ttest):
    class0 = Xtrain[extract_label_indices(Ttrain, 0)]
    mu0 = get_data_mean(class0)
    cov0 = my_cov(class0, mu0)
    
    class1 = Xtrain[extract_label_indices(Ttrain, 1)]
    mu1 = get_data_mean(class1)
    cov1 = my_cov(class1, mu1)
    
    prior0 = my_prior(Xtrain, class0)
    prior1 = my_prior(Xtrain, class1)
    
    a = predict(class0, class1, cov0, cov1, mu0, mu1, prior0, prior1, Xtrain)
    b = predict(class0, class1, cov0, cov1, mu0, mu1, prior0, prior1, Xtest)
    
    right_train = (1*(a==Ttrain)).sum()   
    my_accuracy_train = float(right_train)/len(a)   
    print('|accuracy4a - accuracy4e(train)|: ' + str(abs(accuracy4a-my_accuracy_train)))
    
    right_test = (1*(b==Ttest)).sum()  
    my_accuracy_test = float(right_test)/len(b)   
    print('|accuracy4a - accuracy4e(test)|: ' + str(abs(accuracy4a-my_accuracy_test)))
    
myGDA(X_repeat, t_repeat, X_test, y_test)


#5
with open('mnist.pickle','rb') as f:
    Xtrain,Ytrain,Xtest,Ytest = pickle.load(f)
    
#a
def MNIST(Xtrain, question_part):
    randList = random.sample(range(Xtrain.shape[0]),25)
    figure_5 = plt.figure(figsize = (5,5))
    plt.suptitle('Question 5(' + str(question_part)+'): 25 random MNIST images')
    for i in range(len(randList)):
        image = Xtrain[randList[i]].reshape([28,28])
        figure_5.add_subplot(5,5,i+1)
        plt.axis('off')
        plt.imshow(image, cmap='Greys', interpolation='nearest')
        
MNIST(Xtrain, 'a')

#b
def modelling(model_type, Xtrain, Ytrain, Xtest, Ytest):
    start_time = time.time()
    model_type.fit(Xtrain, Ytrain)
    total_time = time.time() - start_time
    accuracy_train = model_type.score(Xtrain, Ytrain)
    accuracy_test = model_type.score(Xtest, Ytest)
    print('Training accuracy: ' + str(accuracy_train))
    print('Testing accuracy: ' + str(accuracy_test))
    print('Time taken to fit the model to the training data: ' + str(total_time) + ' seconds')

clf5b = QuadraticDiscriminantAnalysis()
modelling(clf5b, Xtrain, Ytrain, Xtest, Ytest)

#c
gnb_5c = GaussianNB()
modelling(gnb_5c, Xtrain, Ytrain, Xtest, Ytest)

#adding noise
sigma = 0.1
noise = sigma*np.random.normal(size=np.shape(Xtrain))
Xtrain = Xtrain + noise

#d
MNIST(Xtrain, 'd')

#e
clf5e = QuadraticDiscriminantAnalysis()
modelling(clf5e, Xtrain, Ytrain, Xtest, Ytest)
gnb_5e = GaussianNB()
modelling(gnb_5e, Xtrain, Ytrain, Xtest, Ytest)

#f
Xtrain, Ytrain = Xtrain[:6000], Ytrain[:6000]
clf5f = QuadraticDiscriminantAnalysis()
modelling(clf5f, Xtrain, Ytrain, Xtest, Ytest)

gnb_5f = GaussianNB()
modelling(gnb_5f, Xtrain, Ytrain, Xtest, Ytest)
gnb_5f_train_acc = gnb_5f.score(Xtrain, Ytrain)
gnb_5f_test_acc = gnb_5f.score(Xtest, Ytest)

#g
figure_5g = plt.figure(figsize = (3,4))
plt.suptitle('Question 5(g): means for each digit class')
for i in range(gnb_5f.theta_.shape[0]):
    image = gnb_5f.theta_[i].reshape([28,28])
    figure_5g.add_subplot(3,4,i+1)
    plt.axis('off')
    plt.imshow(image, cmap='Greys', interpolation='nearest')

#h
def my_mean(class_n):
    return np.sum(class_n, axis=0)/len(class_n)

def extract_label_indices(y, label):
    return np.where(y==label)[0]

def my_var(class_k, mu):
    return np.sum((class_k.T-mu.reshape(-1,1))**2, axis=1)/len(class_k)

def my_p_xy(x, mu, var):
    den = np.sqrt(2*np.pi)*np.sqrt(var)
    num1 = (-(x-mu)**2)
    den1 = (2*var)
    num = np.exp(num1/den1)
    return num/den

def my_p_y(prior):
    return 1-prior

def myGNB(Xtrain,Ttrain,Xtest,Ttest):
    probs_test = []
    probs_train = []
    labels = list(set(Ttrain))
    for k in labels:
        class_k = Xtrain[extract_label_indices(Ttrain, k)]
        mu = my_mean(class_k)
        var = my_var(class_k, mu)
        p_y = my_p_y(float(len(class_k))/float(len(Xtrain)))
        p_xy = my_p_xy(Xtrain, mu, var)
        predict_probs = p_y*np.prod(p_xy, axis=1)
        probs_train.append(predict_probs)
        p_xy = my_p_xy(Xtest, mu, var)
        predict_probs = p_y*np.prod(p_xy, axis=1)
        probs_test.append(predict_probs)
    
    predicted = np.argmax(probs_train, axis=0)
    GNB_training_acc = np.sum(predicted==Ttrain)/float(len(Xtrain))
    print('Training accuracy: ' + str(GNB_training_acc))
    predicted = np.argmax(probs_test, axis=0)
    GNB_testing_acc = np.sum(predicted==Ttest)/float(len(Xtest))
    print('Testing accuracy: '+ str(GNB_testing_acc))
    
    print('Difference in training accuracy: ' + str(abs(GNB_training_acc - gnb_5f_train_acc)))
    print('Difference in testing accuracy: ' + str(abs(GNB_testing_acc - gnb_5f_test_acc)))
        
myGNB(Xtrain,Ytrain,Xtest,Ytest)