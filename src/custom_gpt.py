import torch
import torch.nn as nn
import torch.nn.functional as F
import tiktoken
import pandas as pd

# set seed
torch.manual_seed(0)

# set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)

# hyperparameters
eval_interval = 100
max_iters = 5000
learning_rate = 3e-4
batch_size = 2

# GPT parameters
block_size = 256
n_embed = 1024
n_head = 4
n_layer = 24
dropout = 0.2

with open("data/input.txt") as f:
    text = f.read()

# prepare tokenisation
enc = tiktoken.encoding_for_model("gpt-4")
n_vocab = enc.n_vocab

# prepare positional encodings
token_embedding_table = nn.Embedding(n_vocab, n_embed)
position_encoding_table = nn.Embedding(block_size, n_embed)

# tokenise data
data = enc.encode(text)
n = int(0.9*len(data))
train_data, valid_data = (data[:n], data[n:])

# batching
def get_batch(source):
    # get batch_size offsets on the data
    ix = torch.randint(len(source)-block_size, (batch_size,))
    # get next predicted token for x in y
    x = torch.stack([torch.tensor(source[i:i+block_size]) for i in ix])
    #x = torch.stack([source[i:i+block_size] for i in ix])
    y = torch.stack([torch.tensor(source[i+1:i+1+block_size]) for i in ix])
    return x.to(device), y.to(device)

# get test batch
xb, yb = get_batch(train_data)

for b in range(batch_size):
    for t in range(block_size):
        context = xb[b,:t+1]
        target = yb[b,t]

# C = embedding size
# B = batch size
# T = block size
B, T, C, = batch_size, block_size, n_embed
"""
x = torch.randn(B,T,C).to(device) #shape (B,T,C)

# compute uniform attention matrix
#normalize mask so that it sums to one. use keepdim to make broadcast operation work later
tril = torch.tril(torch.ones((T,T), dtype=torch.float32, device=device))
wei = tril.masked_fill(tril==0, float('-inf'))
wei = F.softmax(wei, dim=-1)

out = wei @ x

# compute W matrices as a linear projection
head_size = 4
key = nn.Linear(C, head_size, bias=False)
query = nn.Linear(C, head_size, bias=False)
value = nn.Linear(C, head_size, bias=False)

# computing attention(Q, K, V)
k = key(x) # shape (B, T, head_size)
q = query(x) # shape (B, T, head_size)
wei = q @ k.transpose(-2, -1) # shape (B, T, head_size) @ (B, head_size, T) = (B, T, T)
wei *= head_size**-0.5 # sqrt(d_k) so that variance is 1

tril = torch.tril(torch.ones((T, T), dtype=torch.float32, device=device))
wei = tril.masked_fill(tril==0, float('-inf'))
wei = F.softmax(wei, dim=-1)
v = value(x) # shape (B, T, head_size)
out = wei @ v # shape (B, T, T) @ (B, T, C) --> (B, T, C)
"""

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(C, head_size, bias=False)
        self.query = nn.Linear(C, head_size, bias=False)
        self.value = nn.Linear(C, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones((block_size, block_size), dtype=torch.float32, device=device)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x) # shape (B, T, head_size)
        q = self.query(x) # shape (B, T, head_size)
        v = self.value(x) # shape (B, T, head_size)
        
        wei = q @ k.transpose(-2, -1) # shape (B, T, head_size) @ (B, head_size, T) = (B, T, T)
        wei *= C**-0.5 # sqrt(d_k) so that variance is 1
        wei = wei.masked_fill(self.tril[:T,:T]==0, float('-inf'))
        wei = F.softmax(wei, dim=-1) # shape (B, T, T)
        wei = self.dropout(wei)
        out = wei @ v # shape (B, T, T) @ (B, T, C) --> (B, T, C)
        
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out

class FeedForward(nn.Module):
    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, n_embed*4),
            nn.ReLU(),
            nn.Linear(n_embed*4, n_embed),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embed, n_head):
        super().__init__()
        if(n_embed % n_head != 0):
            raise Exception("Embedding size must be a multiple of the number of heads.")
        head_size = n_embed // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embed)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)
        
    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
        
class LiamGPT(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # vocab and positional embeddings
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.positional_embedding_table = nn.Embedding(block_size, n_embed)
        # sequence of GPT blocks
        self.blocks = nn.Sequential(*[Block(n_embed, n_head) for _ in range(n_layer)])
        # add layer normalisation at the end of the blocks
        self.ln = nn.LayerNorm(n_embed)
        # add linear layer at output
        self.lm_head = nn.Linear(n_embed, vocab_size, bias=False)
        
    def forward(self, idx):
        # idx and targets are both shaped (B, T)
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx) # shape (B, T, C)
        pos_emb = self.positional_embedding_table(torch.arange(T, device=idx.device)) # shape (T, C)
        x = tok_emb + pos_emb # shape = (B, T, C)
        x = self.blocks(x)
        x = self.ln(x)
        logits = self.lm_head(x)
        logits = torch.swapaxes(logits, 1, 2) # shape (B, C, T) to comply with cross entropy loss
        return logits
    
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits = self(idx_cond)
            logits = logits[:, :, -1] # shape from (B, C, T) to (B, C)
            probs = F.softmax(logits, dim=-1) # shape (B, C)
            idx_next = torch.multinomial(probs, num_samples=1) # shape (B, 1)
            idx = torch.cat([idx, idx_next], dim=-1) # shape (B, T+1)
        return idx

model = LiamGPT(n_vocab).to(device)

print("Model size :", sum(p.numel() for p in model.parameters() if p.requires_grad))

# data logging
logging = []

optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)
for steps in range(max_iters):
    idx, targets = get_batch(train_data) # get batch
    logits = model(idx) # predict logits
    loss = F.cross_entropy(logits, targets) # get loss
    loss.backward() # backwards pass
    optimiser.step() # update parameters
    optimiser.zero_grad(set_to_none=True) # saves memory by nulling zero tensors
    
    # print progress
    if steps % 100 == 0: print(f"step {steps}, loss {loss.item():.2f}")
    
    # data logging
    logging.append([steps, loss.item()])
    
    @torch.no_grad()
    # evaluation with no gradients
    def eval_loss():
        idx, targets = get_batch(valid_data)
        logits = model(idx)
        loss = F.cross_entropy(logits, targets)
        print(f"step {steps}, eval loss {loss.item():.2f}")
        return loss
    
    if steps % eval_interval == 0: eval_loss().item()

df = pd.DataFrame(logging, columns=["steps", "loss"])
df.to_csv("log.csv")

idx = torch.zeros((1,1), dtype=torch.long, device=device)

result = model.generate(idx, max_new_tokens=500)
listed_result = list(list(result)[0])
decoded = enc.decode(listed_result)
print(decoded)









