import torch
import torch.nn as nn
import torch.optim as optim

from data import get_data_loader
from model import CIFAR_CNN

train_dataloader, test_dataloader = get_data_loader(batch_size=32)
model = CIFAR_CNN()
print(model)
criterion = nn.CrossEntropyLoss() #loss is an object of the cross entropy loss class now
#optimizer = optim.Adam(model.parameters(), lr = 0.01, weight_decay = 0.01, momentum=0.9)

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(log_dir="/home/bhavika/Users/Bhavika/BITS - Acads/Computer_Vision/logs/CNN_pytorch")
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 50

for epoch in range(num_epochs):
    model.train() 
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

    writer.add_scalar("Loss/train", train_loss, epoch)
    writer.add_scalar("Accuracy/train", train_acc, epoch)

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
    writer.add_scalar("Accuracy/test", accuracy, epoch)
    print(f"Test accuracy is {accuracy: .2f}, Train loss is {train_loss: .2f} and train accuracy is {train_acc: .2f} for epoch {epoch+1}/{num_epochs}")

print(f"DONE! loop ran {num_epochs} times")




writer.close()

