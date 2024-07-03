import torch
from torch.nn import Transformer
import torch.functional as F
import dataset_tools as utils
from datasets import load_dataset

# parameters
d_model = 1024
nhead = 8
num_encoder_layers = 0
num_decoder_layers = 16
dim_feedforward = 2048
dropout = 0.1
activation = "relu"

layer_norm = 1e-5
batch_first = False
norm_first = False
bias = True

encoder = None
decoder = None

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("DEVICE =", device)

# create model
model = Transformer(d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout, activation, encoder, decoder, layer_norm, batch_first, norm_first, bias, device)

print(utils.count_params(model))

def train(train, test, model, num_epochs=1, max_iters=100, learning_rate=1e-4, block_size=1024, batch_size=4):
    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)
    progress = max_iters // 20
    num_params = utils.count_params(model)

    for epoch in range(num_epochs):
        print("=== EPOCH "+str(epoch)+" ===")
        for step in range(max_iters):
            # generate batch
            idx, targets = utils.get_batch(train, block_size, batch_size, device)
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
            if step % progress == 0:
                print(f"step {step}, loss {loss.item():.2f}")
                
            @torch.no_grad()
            # evaluation pass - disable gradients
            def eval_loss():
                idx, targets = utils.get_batch(test, block_size, batch_size, device)
                logits = model(idx)
                loss = F.cross_entropy(logits, targets)
                print(f"step {step}, eval loss {loss.item():.2f}")
                return loss
            
            if step % progress == 0: eval_loss().item()
        # save model
        torch.save(model, "models/liamgpt_"+num_params+"_e"+epoch)



# GET DATA
ds = load_dataset("Salesforce/wikitext", "wikitext-103-v1")
train = ds["train"]["text"]
test = ds["test"]["text"]
val = ds["validation"]["text"]

# Encoding (bpe)
n_vocab, encoder, decoder = utils.byte_pair_encoding(device)

train_data = []
#for line in train:
#    train_data.append(encoder(line))
test_data = []
for line in test:
    test_data.append(encoder(line))
val_data = []
for line in val:
    val_data.append(encoder(line))

# create tensors
#train = torch.Tensor(train_data, torch.int32, device)
test = torch.Tensor(test_data, device)
print(test)

# deallocate memory
train_data = 0
test_data = 0

print(utils.get_batch(train, 4, 1024, device))

# train model
#train()
