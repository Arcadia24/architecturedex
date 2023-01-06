import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from tqdm import tqdm

from utils import ALL_LETTERS, N_LETTERS
from utils import load_data, random_training_example, letter_to_tensor, line_to_tensor

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size) -> None:
        super(RNN, self).__init__()
        
        
        self.hidden_size = hidden_size
        
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        
        self.softmax = nn.LogSoftmax(dim = 1)
        
    def forward(self, input_tensor, hidden_tensor):
        combined = torch.cat((input_tensor, hidden_tensor), dim = 1)
        
        return self.softmax(self.i2o(combined)), self.i2h(combined)
    
    def init_hidden(self):
        return torch.zeros(1, self.hidden_size)


#---initilization---
cat_lines, all_cat = load_data()

n_cat = len(all_cat)
print(n_cat)
n_hidden = 128
rnn = RNN(N_LETTERS, n_hidden, n_cat)

#---for one letter---
input_tensor = letter_to_tensor('A')
hidden_tensor = rnn.init_hidden()

output, h = rnn(input_tensor, hidden_tensor)
# print(output.shape, h.shape)

#---for a name/sequence---
input_tensor = line_to_tensor("Albert")
hidden_tensor = rnn.init_hidden()

output, h = rnn(input_tensor[0], hidden_tensor)
# print(output.shape, h.shape)

def category_from_output(output):
    cat_idx = torch.argmax(output).item()
    return all_cat[cat_idx]

# print(category_from_output(output))
criterion = nn.NLLLoss()
optim = torch.optim.AdamW(rnn.parameters(), lr = 5e-3)

def train(loader):
    hidden = rnn.init_hidden()
    total_loss, total_acc = .0,.0
    
    with tqdm(loader, desc="Train") as pbar:
        for X, y in pbar:
            for i in range(X.shape[0]):
                output, hidden = rnn(X[i], hidden)
            
            loss = criterion(output, y)
            acc = ().sum()
                
            optim.zero_grad()
            loss.backward()
            optim.step
        
            total_loss += loss.item() / len(loader)
            total_acc += acc.item() / len(loader.dataset)
            pbar.set_postfix(loss=f"{total_loss:.2e}", acc=f"{total_acc * 100:.2f}%")


