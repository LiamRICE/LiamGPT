import sklearn.model_selection
import torch
import tiktoken
import sklearn
import torch.types

def read_file(filename):
    file = "./data/" + filename
    text = ""
    with open(file) as f:
        text = f.read()
    return text

def split_filter_train_examples(text):
    text = text.split('\n')
    final_text = []
    for t in text:
        if not t == '':
            final_text.append(t)
    return final_text

def get_batch(source, block_size, batch_size, device="cpu"):
    # source is an iterable
    # collect data from source of size block size
    ix = torch.randint(len(source) - block_size, (batch_size,))
    x: torch.IntTensor = torch.stack([source[i:i+block_size] for i in ix])#.to(torch.int32)
    y: torch.IntTensor = torch.stack([source[i+1:i+1+block_size] for i in ix])#.to(torch.int32)
    return x.to(device), y.to(device)

def get_context_target(xb, yb, block_size, batch_size):
    for b in range(batch_size):
        for t in range(block_size):
            context = xb[b,:t+1]
            target = yb[b,t]
    return context, target

def simple_encoding(text, device):
    chars = sorted(list(set(text)))
    # string-to-int (stoi) and int-to-string (itos) dictionaries
    stoi = {ch:i for i,ch in enumerate(chars)}
    itos = {i:ch for i,ch in enumerate(chars)}
    # create encode and decode functions
    encode = lambda x: torch.tensor([stoi[ch] for ch in x], dtype=torch.long).to(device)
    decode = lambda x: ''.join([itos[i] for i in x.tolist()]).to(device)
    # get vocab_size
    vocab_size = len(stoi)
    return vocab_size, encode, decode

def byte_pair_encoding(device):
    # get encoding model
    enc = tiktoken.encoding_for_model("gpt-4")
    encode = lambda x: torch.tensor(enc.encode(x)).to(device)
    decode = lambda x: enc.decode(x)
    # return vocabulary size, encoding and decoding functions
    return enc.n_vocab, encode, decode

def train_test_split(data, train_percent):
    return sklearn.model_selection.train_test_split(data, train_size=train_percent)


"""
n_vocab, encode, decode = simple_encoding(read_file("input", "txt"))
print(n_vocab)
rep = encode("Hello world!")
print(rep)
print(decode(rep))

n_vocab, encode, decode = byte_pair_encoding()
print(n_vocab)
rep = encode("Hello world!")
print(rep)
print(decode(rep))
"""