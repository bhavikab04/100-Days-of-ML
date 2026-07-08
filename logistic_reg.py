from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import pandas as pd
#import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def softmax(z):
    rowmax = np.max(z,axis = 1, keepdims = True)
    num = np.exp(z-rowmax) #note: keepdims only belongs to reduction operations
    denom = np.sum(num,axis = 1, keepdims = True) # axis = 1 means along each row, sum here is row wise
    return num/denom

def crossentropy(probs, onehot_y):
    N = probs.shape[0] #pleas enote that.size returns a single integer of the total number of elements in the numpy array --> so here it will return N x 10; .shape however returns a tuple
    log_probs = np.log(probs + 1e-12) #need the guard as incase probs is exactly zero it will give -ve inf
    #loss = -(1/N)*np.sum(np.sum((onehot_y*log_probs), axis = 1),axis = 0) --> no need to do this as basically what we are doing here is finding the sumn of all elements, this can be done with just one np.sum as well
    loss = -(1/N)*np.sum(onehot_y*log_probs)
    return loss

digits = load_digits()
X = digits.data     # shape (1797, 64) --> 1797 images, 64 pixels each (flattened 8x8)
y = digits.target
np.random.seed(42)
""" fig,axes = plt.subplots(2,5, figsize=(10,4))
for digit,ax in zip(range(10), axes.flat):
    avg_img = X[digit==y].mean(axis=0).reshape(8,8)
    ax.imshow(avg_img,cmap="gray")
plt.show() """
w = np.random.randn(X.shape[1],10) * 0.01 #using randn instead of rand because rand give only (0,1) values but randn gives negative values too
b = np.zeros(10) #note: bias is set to the outputs not the inputs, so it's one bias value per output class, so only 10 bias values not 64
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scores = X_train @ w + b
probs = softmax(scores) 

""" test = np.array([[1.0, 2.0, 3.0], [1.0, 1.0, 1.0]])
print(softmax(test))
print(softmax(test).sum(axis=1))   # should be [1. 1.] """

num_classes = 10
y_one_hot = np.zeros((y_train.size, num_classes))
#please dont write .size() because size() is not a function, it is an attribute of the object
y_one_hot[np.arange(y_train.size), y_train] = 1
#we need ot onehot because it will come up in the gradient when we are subtracting the ground truth class from the predicted class 

#print(y_one_hot)
""" loss = crossentropy(probs, y_one_hot)
N = probs.shape[0]
dl_dw = X_train.T @ ((1/N)*(probs - y_one_hot))
dl_db = (1/N)*np.sum(probs - y_one_hot, axis = 0)
print(f"shape of dl_dw is {dl_dw.shape} and shape of dl_db is {dl_db.shape}")
 """
#training loop:
itr = 1000 # here one itr is same as one epoch as we are using the entire batch for each update
#if we make mini batches, then one weight update uses that mini batch of say 100 samples, and this is called an itreration
#one epoch is one full pass over all the samples
#1 epoch = (no of samples/batch size)*iterations
alpha = 0.1
N = probs.shape[0]
lamda = 0.01 #regularization

for i in range(itr):

    #froward pass
    scores = X_train @ w + b
    probs = softmax(scores)
    predictions = np.argmax(probs, axis = 1) #argmax returns the index of the highest value, ie the class number, rather than the max prob itself
    train_acc = np.mean(predictions == y_train)
    loss = crossentropy(probs, y_one_hot)

    #backward pass
    dl_dw = X_train.T @ ((1/N)*(probs - y_one_hot)) + lamda * w
    dl_db = (1/N)*np.sum(probs - y_one_hot, axis = 0)

    #parameter update
    w = w - alpha*dl_dw
    b = b - alpha*dl_db

    
    if(i%100 == 0):
        print(f"loss is {loss:.4f} for itr number {i}")
        print(f"train_acc is {train_acc:.4f} for itr number {i}")
    
final_scores = X_test @ w + b
final_probs = softmax(final_scores)
final_predictions = np.argmax(final_probs, axis = 1)
test_accuracy = np.mean(final_predictions == y_test)
print(f"Final test accuracy is {test_accuracy: .2f}")