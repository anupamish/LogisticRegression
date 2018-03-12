import numpy as np
np.random.seed(0)
import math
data = np.genfromtxt("classification.txt",delimiter=",")
data = np.delete(data,[3],axis=1)
y = data[:,-1]
X = np.delete(data,[3],axis=1)
vectorToAppend=np.ones([X.shape[0],1])
X=np.concatenate((vectorToAppend,X),axis=1)
weights=np.random.randint(4,size=(1,4))
learningRate=0.0001
y = y.reshape(X.shape[0],-1)
#Returns the sigmoid value of the whole vector
def exponent(y,weights,dataPoint):
    return math.exp(-1*y*np.dot(weights,dataPoint))

iteratons=0
while iteratons<7000:
    gradientSum =0
    updateSum = 0
    for i in range(0,X.shape[0]):
        expFactor = exponent(y[i],weights,X[i])
#         expFactor = math.exp(-1*y[i]*np.dot(weights,X[i]))
        factor = float((-1 * expFactor))/(1 + expFactor)
        factor = factor * y[i]* X[i]
        updateSum += factor
    gradientSum=-1*(updateSum)/float(X.shape[0])
    weights=weights-learningRate*gradientSum
    iteratons+=1

print ("Final Weights for logistic regression are:\n ",weights)
    

def sigmoid(a):
    return math.exp(a)/float((1 + math.exp(a)))

preds = np.dot(X,np.transpose(weights))
count=0
for i in range(0,len(preds)):
#     pdf = math.exp(preds[i][0])/(1+ math.exp(preds[i][0]))
    pdf = sigmoid(preds[i][0])
    preds[i][0] = 1 if pdf >= 0.5 else -1
    if preds[i][0] == y[i][0]:
        count += 1
accuracy = (count/float(len(preds))) * 100

print("Accuracy of logistic regression ",accuracy,"%")