# 100-Days-of-ML
Here begins my journey to learn, revise and solidify ML and DL concepts 

## Day 1 — Neural networks from scratch (NumPy)
- Implemented **batch gradient descent linear regression** from scratch in NumPy; derived MSE gradients by hand, converged to correct parameters.
- Implemented **multi-class logistic regression** on the sklearn digits dataset: softmax (with numerical stability), cross-entropy loss, one-hot encoding, full backprop gradient. Reached ~97% test accuracy.
- Added **L2 regularization**.

## Day 2 — Two-hidden-layer MLP from scratch (NumPy)
- Built a genuine **2-hidden-layer MLP** (ReLU activations, softmax output) entirely in NumPy.
- Derived and coded **backpropagation through all layers by hand** — the ReLU gate, chained gradients across layers.
- Learnt **He initialization** and why it matters for ReLU nets.
- Diagnosed a bouncing loss as a **learning-rate issue**; tuned it and hit ~96% test accuracy.

## Day 3 — MLP rebuilt in PyTorch
- Rebuilt the same MLP using `nn.Module`, `nn.Linear`, `F.relu`, `CrossEntropyLoss`, and `optim.SGD`.
- Added **mini-batching** with `DataLoader` and **train/test tracking per epoch**.
- Split the project into `model.py`, `data.py`, `train.py` (modular structure).
- Matched the from-scratch version at ~97% test accuracy; observed mild overfitting live.

## Day 4 — CNNs (in progress)
- Started the CNN rung: convolution forward pass from scratch (NumPy), then a full CNN in PyTorch on CIFAR-10.
