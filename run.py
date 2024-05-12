import src.dataset_tools as dataset_tools
import src.multilayer_gpt as gpt
import torch
import torch.nn.functional as F

# set seed
torch.manual_seed(42)

# find device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("DEVICE =", device)

# set default device to cuda if available
torch.set_default_device(device)

# hyperparameters
eval_interval = 100
max_iters = 500
learning_rate = 1e-4
batch_size = 3
block_size = 16

# model parameters
vocab_size = 16384
embed_dim = 2048
num_heads = 8
ff_hidden_layer = 8 * embed_dim
dropout = 0.1
num_layers = 16
context_length = 1024

# reading source data
text = dataset_tools.read_file("input.txt")

text = dataset_tools.split_filter_train_examples(text)

# generating encoding and decoding functions
n_vocab, encode, decode = dataset_tools.byte_pair_encoding(device)

train, test = dataset_tools.train_test_split(text, 0.8)

# encoding data
train_data = []
for d in train:
    train_data.extend(encode(d))
test_data = []
for d in test:
    test_data.extend(encode(d))

train = torch.Tensor(train_data).to(device)
test = torch.Tensor(test_data).to(device)

# batching data
x, y = dataset_tools.get_batch(train, block_size, batch_size, device)

# generating context targets for pretraining
context, target = dataset_tools.get_context_target(x, y, block_size, batch_size)

model = gpt.MultiLayerTransformerDecoder(vocab_size, embed_dim, num_heads, ff_hidden_layer, dropout, num_layers, device).to(device)

optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)

for step in range(max_iters):
    # generate batch
    idx, targets = dataset_tools.get_batch(train, block_size, batch_size, device)
    # forward pass
    logits = model(idx)
    # get loss
    loss = F.cross_entropy(logits, targets)
    # backward pass
    loss.backward()
    # optimizer step
    optimiser.step()
    # zero gradient pass
    optimiser.zero_grad(set_to_none=True)
    
    # print progress
    if step % 100 == 0:
        print(f"step {step}, loss {loss.item():.2f}")
        
    @torch.no_grad()
    # evaluation pass - disable gradients
    def eval_loss():
        idx, targets = dataset_tools.get_batch(test, block_size, batch_size, device)
        logits = model(idx)
        loss = F.cross_entropy(logits, targets)
        print(f"step {step}, eval loss {loss.item():.2f}")
        return loss
    
    if step % eval_interval == 0: eval_loss().item()












