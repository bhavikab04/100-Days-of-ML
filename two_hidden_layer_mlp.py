from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import numpy as np

def softmax(z):
    rowmax = np.max(z,axis = 1, keepdims = True)
    num = np.exp(z-rowmax) #note: keepdims only belongs to reduction operations
    denom = np.sum(num,axis = 1, keepdims = True) # axis = 1 means along each row, sum here is row wise
    return num/denom

def relu(z):
    return np.maximum(0,z) # note: np.max finds the global maximum of ONE entire array, np.maximum compares TWO arrays element wise and takes the maximum; so np.max is an aggregate function

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

w0 = np.random.randn(X.shape[1],64) * np.sqrt(2/X.shape[1]) #using randn instead of rand because rand give only (0,1) values but randn gives negative values too
w1 = np.random.randn(64,32) * np.sqrt(2/64) #also randn generates values of N(0,1), i.e, mean 0 and variance 1. when it is multiplied by a constant C, the variance scales as C^2(as standard dev gets multiplied by C) so we multiply by the sqrt of 2/n
w2 = np.random.randn(32,10) * np.sqrt(2/32)
b0 = np.zeros(64)
b1 = np.zeros(32)
b2 = np.zeros(10) #note: bias is set to the outputs not the inputs, so it's one bias value per output class, so only 10 bias values not 64
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

num_classes = 10
y_one_hot = np.zeros((y_train.size, num_classes))
y_one_hot[np.arange(y_train.size), y_train] = 1
#training loop:
itr = 1000 # here one itr is same as one epoch as we are using the entire batch for each update
alpha = 0.01 #What's happening: your alpha = 0.1 is too large for this network. When the learning rate is too high, each update overshoots the minimum
N = X_train.shape[0]
lamda = 0.01 #regularization

for i in range(itr):

    #forward pass
    z0 = X_train @ w0 + b0
    a0 = relu(z0)
    z1 = a0 @ w1 + b1
    a1 = relu(z1)
    z2 = a1 @ w2 + b2
    probs = softmax(z2)
    predictions = np.argmax(probs, axis = 1) #argmax returns the index of the highest value, ie the class number, rather than the max prob itself
    train_acc = np.mean(predictions == y_train)
    loss = crossentropy(probs, y_one_hot)
    #print(f"shapes of z0, a0, z1, a1, z2, probs are: {z0.shape}, {a0.shape}, {z1.shape}, {a1.shape}, {z2.shape}, {probs.shape}")
    #shapes of z0, a0, z1, a1, z2, probs are: (1437, 64), (1437, 64), (1437, 32), (1437, 32), (1437, 10), (1437, 10)
    dl_dz2 = (1/N)*(probs - y_one_hot)
    dl_dw2 = (a1.T @ dl_dz2) + lamda * w2
    dl_db2 = np.sum(dl_dz2, axis = 0)

    dl_da1 = dl_dz2 @ w2.T # size is now (N,32) ; this is matrix multiplication
    dl_dz1 = dl_da1 * (z1>0) # this is elementwise multiplication, used only when applying an activation function in numpy, it is NOT Matrix multiplication
    #so, size is still (N,32)
    dl_dw1 = (a0.T @ dl_dz1) + lamda * w1 # final size is( 64,32)
    dl_db1 = np.sum(dl_dz1, axis = 0) #final size is (32, ) as we do column wise sum

    dl_da0 = dl_dz1 @ w1.T # size is (N,32) x (32,64) = (N,64)
    dl_dz0 = dl_da0 * (z0>0) # size is still (N,64)
    dl_dw0 = (X_train.T @ dl_dz0 ) +  lamda * w0# size is now (64,N) x (N, 64) = (64,64)
    dl_db0 = np.sum(dl_dz0, axis = 0)

    #update parameters:
    w0 = w0 - alpha * dl_dw0
    w1 = w1 - alpha * dl_dw1
    w2 = w2 - alpha * dl_dw2
    b0 = b0 - alpha * dl_db0
    b1 = b1 - alpha * dl_db1
    b2 = b2 - alpha * dl_db2


    if(i%100 == 0):
        print(f"loss is {loss:.4f} for itr number {i}")
        print(f"train_acc is {train_acc:.4f} for itr number {i}")
    
z0 = X_test @ w0 + b0
a0 = relu(z0)
z1 = a0 @ w1 + b1
a1 = relu(z1)
final_scores = a1 @ w2 + b2
final_probs = softmax(final_scores)
final_predictions = np.argmax(final_probs, axis = 1)
test_accuracy = np.mean(final_predictions == y_test)
print(f"Final test accuracy is {test_accuracy: .2f}")