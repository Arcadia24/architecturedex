from torchvision.datasets import MNIST
from torchvision import transforms
from torch.cuda.amp import (autocast, GradScaler)
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

import numpy as np
import torch.nn as nn
import torch

# Import MNIST dataset
train_dataset = MNIST(root="data", train=True, transform=transforms.ToTensor(), download=True)
test_dataset = MNIST(root="data", train=False, transform=transforms.ToTensor(), download=True)

# Define the model

class RecNet(nn.Module):
    def __init__(self, input_size : int, hidden_size : int = 128, num_layer : int = 2, num_classes : int = 10) -> None:
        super().__init__()
        
        self.rnn = nn.RNN(input_size, hidden_size, num_layer, batch_first=True)
        
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        x, h = self.rnn(x)
        return self.fc(x[:, -1, :])
    
class LSTMNet(nn.Module):
    def __init__(self, input_size : int, hidden_size : int, num_layer : int, num_classes : int) -> None:
        """_summary_

        Args:
            input_size (int): input size if the input tensor
            hidden_size (int): hidden layer size
            num_layer (int): number of cell for the LSTM
            num_classes (int): number of classes
        """
        super().__init__()
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layer, batch_first=True)
        
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        """_summary_

        Args:
            x (torch.Tensor): input tensor

        Returns:
            torch.Tensor: prediction
        """
        x, (h, c) = self.lstm(x)
        return self.fc(x[:, -1, :])  
    
class MLP(nn.Module):
    def __init__(self, linear_size : list) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            *[[nn.Linear(linear_size[i], linear_size[i + 1]), nn.ReLU(), nn.Dropout(0.2)]  for i in range(len(linear_size) - 1)]
        )

    
    def forward(self, x):
        return self.mlp(x)

def train2(model, loader, criterion, optim, train : bool):
    model.train() if train else model.eval()
    total_loss, total_acc = .0, .0
    with tqdm(loader, desc = "Training" if train else "Testing", unit = "batch") as t:
        for X, y in t:
            X, y = X.cuda(), y.cuda()
            X = X.squeeze(1)
            
            preds = model(X)
            loss = criterion(preds, y)
            acc = (preds.argmax(dim = 1) == y).float().sum()
            
            if train:
                optim.zero_grad()
                loss.backward()
                optim.step()
            
            total_loss += loss.item()/len(loader)
            total_acc += acc.item()/len(loader.dataset)
            
            t.set_postfix(loss = total_loss, acc = total_acc)
    
    return total_loss

def train(model : nn.Module, loader : DataLoader, criterion : nn.Module, optim : nn.Module, train : bool) -> float:
    """_summary_

    Args:
        model (nn.Module): Modek to train or test
        loader (DataLoader): DataLoader
        criterion (nn.Module): Criterion to calculate loss
        optim (nn.Module): Optumizer
        train (bool): to know if train or test

    Returns:
        float: loss
    """
    model.train() if train else model.eval()
    total_loss, total_acc = .0, .0
    with tqdm(loader, desc = "Training" if train else "Testing", unit = "batch") as t:
        for X, y in t:
            X, y = X.cuda(), y.cuda()
            X = X.squeeze(1)
            
            preds = model(X)
            loss = criterion(preds, y)
            acc = (preds.argmax(dim = 1) == y).float().sum()
            
            if train:
                optim.zero_grad()
                loss.backward()
                optim.step()
            
            total_loss += loss.item()/len(loader)
            total_acc += acc.item()/len(loader.dataset)
            
            t.set_postfix(loss = total_loss, acc = total_acc)
    
    return total_loss
            
def fold(model : nn.Module, train_loader : DataLoader, test_loader : DataLoader, criterion : nn.Module, optim : nn.Module, lr : nn.Module, epochs : int, name : str):
    """_summary_

    Args:
        model (nn.Module): Model to train
        train_loader (DataLoader): DataLoader for training
        test_loader (DataLoader): DataLoader for testing
        criterion (nn.Module): Criterion to calculate loss
        optim (nn.Module): optimizer
        lr (nn.Module): Learnin rate scheduler_
        epochs (int): epochs to train
        name (str): name of the model to save
    """
    loss = np.inf
    for i in tqdm(range(epochs), desc = "Epochs", unit = "epoch"):
        train(model, train_loader, criterion, optim, True)
        total_loss = train(model, test_loader, criterion, optim, False)
        lr.step()
        if total_loss < loss:
            loss = total_loss
            print("Saving model")
            torch.save(model.state_dict(), name + ".pth")

def fold2(model, train_loader, test_loader, criterion, optim, lr, epochs : int, name : str):
    loss = np.inf
    for i in tqdm(range(epochs), desc = "Epochs", unit = "epoch"):
        train2(model, train_loader, criterion, optim, True)
        total_loss = train2(model, test_loader, criterion, optim, False)
        lr.step()
        if total_loss < loss:
            loss = total_loss
            print("Saving model")
            torch.save(model.state_dict(), name + ".pth")
    
# Define the hyperparameters

input_size = 28
hidden_size = 128
num_layers = 2
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 10


if __name__ == "__main__":
    # Define the model, loss function and optimizer

    rnn = RecNet(input_size, hidden_size, num_layers, num_classes).cuda()
    lstm = LSTMNet(input_size, hidden_size, num_layers, num_classes).cuda()
    mlp = MLP([28 * 28, 1024, 512, 256, 128, 10]).cuda()
    
    criterion = nn.CrossEntropyLoss().cuda()
    
    optimizer1 = AdamW(rnn.parameters(), lr=learning_rate)
    optimizer2 = AdamW(lstm.parameters(), lr=learning_rate)
    optimizer3 = AdamW(mlp.parameters(), lr=learning_rate)
    
    lr2 = CosineAnnealingLR(optimizer2, num_epochs)
    lr3 = CosineAnnealingLR(optimizer3, num_epochs)

    # Define the data loader

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    
    
    # Train the model
    
    #fold(rnn, train_loader, test_loader, criterion, optimizer1, num_epochs, "rnn")
    
    #fold(lstm, train_loader, test_loader, criterion, optimizer2, lr2, num_epochs, "lstm")
    
    #fold2(mlp, train_loader, test_loader, criterion, optimizer3, lr3, num_epochs, "mlp")
    

            