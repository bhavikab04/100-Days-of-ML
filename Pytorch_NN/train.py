import torch
import torch.nn as nn
import torch.optim as optim
from model import MLP
from data import get_dataloader

train_dataloader, test_dataloader = get_dataloader(batch_size=32)
model = MLP()
criterion = nn.CrossEntropyLoss() #loss is an object of the cross entropy loss class now
optimizer = optim.SGD(model.parameters(), lr = 0.01, weight_decay = 0.01, momentum=0.9) #note that weight decay is for l2 regularization , i.e, its lamda value = 0.01
#model parameters are w0,w1,w2,b0,b1,b2
#Momentum: It makes updates accumulate a "velocity" across steps instead of reacting only to the current gradient
num_epochs = 10

for epoch in range(num_epochs):
    model.train() # set train mode (matters once you add dropout/batchnorm)
    running_loss = 0.0
    train_correct = 0
    train_total = 0
    for xb, yb in train_dataloader:
        optimizer.zero_grad() #clear old gradients so that they dont accumulate over multiple batches
        logits = model(xb) # forward pass
        loss = criterion(logits, yb) # computes loss
        loss.backward() # backpropagaation
        optimizer.step() #update parameters

        # to compute training accuracy and loss
        running_loss += loss.item()
        preds = logits.argmax(dim=1)
        train_correct += (preds == yb).sum().item()
        train_total += yb.size(0)

    train_loss = running_loss / len(train_dataloader)   # avg loss per batch
    train_acc = train_correct / train_total     


    model.eval()
    test_correct = 0
    test_total = 0
    #matches = 0
    with torch.no_grad():
        for xb, yb in test_dataloader:
            logits = model(xb)
            preds = torch.argmax(logits, dim = 1) # row wise max arg
            #matches += (preds == yb) # boolean tensor: which are right like this: (preds == yb) → [True, True, False, True, False]
            num_correct_this_batch = (preds == yb).sum().item() #[True, True, False, True, False].sum() → tensor(3) --> so it gives a number but still in tensor form. so use .item() to retireve only the number
            test_correct += num_correct_this_batch
            test_total += yb.size(0)

    accuracy = test_correct/test_total
    print(f"Test accuracy is {accuracy: .2f}, Train loss is {train_loss: .2f} and train accuracy is {train_acc: .2f} for epoch {epoch+1}/{num_epochs}")

print(f"DONE! loop ran {num_epochs} times")