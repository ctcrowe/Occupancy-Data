
import torch
import torch.nn as nn
import time
import os
from torch.nn import functional as F
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from dataclasses import dataclass

# hyperparameters
batch_size = 8 # how many independent sequences will we process in parallel?
block_size = 16
max_iters = 5000
eval_interval = 100
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 64
n_head = 4
n_layer = 4
dropout = 0.2
# ------------


chars = [' ', ',', '_', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
        'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
class_map = {"A-1" : 0, "A-2" : 1, "A-3" : 2, "A-4" : 3, "A-5" : 4, "B" : 5, "E" : 6, "F-1" : 7, "F-2" : 8, "H-1" : 9, "H-2" : 10, "H-3" : 11, "H-4" : 12,
            "H-5" : 13, "I-1" : 14, "I-2" : 15, "I-3" : 16, "I-4" : 17, "L" : 18, "M" : 19, "C" : 20, "R-1" : 21, "R-2" : 22, "R-2.1" : 23, "R-3" : 24, "R-3.1" : 25,
            "R-4" : 26, "S-1" : 27, "S-2" : 28}

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    data_out = train_outputs if split == 'train' else val_outputs
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i] for i in ix])
    y = torch.stack([data_out[i] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def get_Sample(input, printSample=False):
    input = input.strip().upper()
    lines = input.split(',')
    line = lines[0]
    sample = [0] * block_size
    area = [0]
    types = [0] * len(class_map.items())
    for i in range(len(line)):
        if(chars.count(line[i]) > 0):
            sample[i] = chars.index(line[i])
    try :
        area[0] = (float)(lines[1])
    except: area[0] = 0
       
    if len(lines) > 1 :
        classification = lines[-1]
        classification = class_map[classification]
    else :
        classification = 0
    
    if printSample :
        print(input, sample, classification)
    return torch.tensor(sample), torch.tensor(area), torch.tensor(types), torch.tensor(classification)

class OLFDataset(Dataset):
    def __init__(self, lines):
        #self.txt_path = "/workspaces/OLF-Data/OLFNetworkData.txt"
        self.data = []
        self.chars = chars
        self.class_map = class_map
        self.max_len = block_size
        #with open('OLFNetworkData.txt', 'r', encoding='utf-8') as f:
            #text = f.read()
        for line in lines: # text.splitlines():
            name, area, types, sample = get_Sample(line)
            self.data.append([name, area, types, sample])
        self.stoi = {ch:i+1 for i,ch in enumerate(chars)}
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        name, area, types, sample = self.data[idx]
        return name, area, types, sample
    
def create_datasets(input_file):
    with open(input_file, 'r') as f:
        data = f.read()
    inputs = data.splitlines()

    test_set_size = min(1000, int(len(inputs) * 0.1))
    rp = torch.randperm(len(inputs)).tolist()
    train_words = [inputs[i] for i in rp[:-test_set_size]]
    test_words = [inputs[i] for i in rp[-test_set_size:]]
    print(f"split up the dataset into {len(train_words)} training examples and {len(test_words)} test examples")

    train_dataset = OLFDataset(train_words)
    test_dataset = OLFDataset(test_words)
    return train_dataset, test_dataset

class InfiniteDataLoader:
    def __init__(self, dataset, **kwargs):
        train_sampler = torch.utils.data.RandomSampler(dataset, replacement=True, num_samples=int(1e10))
        self.train_loader = DataLoader(dataset, sampler=train_sampler, **kwargs)
        self.data_iter = iter(self.train_loader)
    
    def next(self):
        try:
            batch = next(self.data_iter)
        except StopIteration:
            self.data_iter = iter(self.train_loader)
            batch = next(self.data_iter)
        return batch

def evaluate(model, dataset, max_batches=None):
    model.eval()
    loader = DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=0)
    losses = []
    for i, batch in enumerate(loader):
        batch = [t.to(device) for t in batch]
        X, Y = batch
        logits, loss = model(X, Y)
        losses.append(loss.item())
        if max_batches is not None and i >= max_batches:
            break
    mean_loss = torch.tensor(losses).mean().item()
    model.train()
    return mean_loss

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias = False)
        self.query = nn.Linear(n_embd, head_size, bias = False)
        self.value = nn.Linear(n_embd, head_size, bias = False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)
        q = self.query(x)

        wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class XfmrModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(len(chars), n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.area_head = nn.Linear(1, n_embd)
        self.type_head = nn.Linear(len(class_map.items()), n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head = n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, len(class_map.items()))
        self.apply(self.__init__weights)

    def __init__weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean = 0.0, std = 0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, A, B, C, targets = None):
        B, T = A.shape
        print(A)
        tok_emb = self.token_embedding_table(A)
        pos_emb = self.position_embedding_table(torch.arange(T, device = device))
        area_emb = self.area_head(B)
        type_emb = self.type_head(C)
        x = tok_emb + pos_emb + area_emb + type_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        x = self.lm_head(x)
        logits = torch.sum(x, dim=-2, keepdim=False)

        if targets is None:
            loss = None
        else:
            B, C = logits.shape
            logits = logits.view(B, C)
            loss_targets = torch.nn.functional.one_hot(targets, len(class_map.items()))
            loss_targets = loss_targets.view(B, len(class_map.items()))
            loss = F.cross_entropy(logits, loss_targets.type(torch.FloatTensor))

        return logits, loss

txt_path = "OccupancyNetworkData.txt"

path = "/workspaces/OLF-Data/OccupancyNetwork.pt"
model = XfmrModel()
if os.path.isfile(path):
    statedict = torch.load(path)
    model.load_state_dict(statedict)

m = model.to(device)
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

optimizer = torch.optim.AdamW(model.parameters(), lr = learning_rate)

def RunTraining():
    train_dataset, test_dataset = create_datasets(txt_path)
    batch_loader = InfiniteDataLoader(train_dataset, batch_size = batch_size)

    best_loss = None
    step = 0

    while True:
        t0 = time.time()
        batch = batch_loader.next()
        batch = [t.to(device) for t in batch]
        A, B, C, D = batch
        print(A)
        print(B)
        print(C)
        print(D)

        logits, loss = model(A, B, C, D)

        model.zero_grad(set_to_none = True)
        loss.backward()
        optimizer.step()

        if device.startswith('cuda'):
            torch.cuda.synchronize()
        t1 = time.time()

        if step % 100 == 0:
            print(f"step {step} | loss {loss.item():.4f} | step time {(t1-t0)*1000:.2f}ms")

        if step > 0 and step % 500 == 0:
            train_loss = evaluate(model, train_dataset, max_batches=5 * batch_size)
            test_loss = evaluate(model, test_dataset, max_batches=5 * batch_size)
            print(f"step {step} train loss: {train_loss} test loss: {test_loss}")
            # save the model to disk if it has improved
            if best_loss is None or test_loss < best_loss:
                print(f"test loss {test_loss} is the best so far, saving model to {path}")
                torch.save(model.state_dict(), path)
                best_loss = test_loss
            
            
        #if step > 0 and step % 200 == 0:
        #    print_samples(num=10)

        step+=1

while True:
    usage = input("Train or Test?")
    if usage == "Test":
        test = ""
        while test != "X":
            text = input("Test your room name")
            sample = get_Sample(text, True)
            A, B, C, D = sample
            A = A.view(1, -1)
            B = B.view(1, -1)
            C = C.view(1, -1)
            logits, loss = model(A, B, C, D)
            print(logits)
            max = torch.argmax(logits)
            print(list(class_map.keys())[max])
    elif usage == "Train":
        RunTraining()
