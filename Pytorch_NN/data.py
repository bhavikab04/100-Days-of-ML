from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, TensorDataset


def get_dataloader(batch_size = 32):
    digits = load_digits()
    X = digits.data     # shape (1797, 64) --> 1797 images, 64 pixels each (flattened 8x8)
    y = digits.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train_tensor = torch.tensor(X_train, dtype = torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype = torch.long) #it's in long(int) form because internally they will be converted into one hot vector form?
    X_test_tensor = torch.tensor(X_test, dtype = torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype = torch.long)

    train_ds = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_ds, batch_size = 32, shuffle = True)
    test_ds = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_ds, batch_size = 32, shuffle = False) #we dont want batch shuffling for the test dataset as we want to keep results reproducible, only for train dataset we shuffle batch 
    return train_loader, test_loader

 