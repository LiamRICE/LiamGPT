import src.multilayer_gpt as gpt
import src.dataset_tools as dataset_tools
import torch
import torch.nn.functional as F

CUDA_LAUNCH_BLOCKING=1

# find device
def detect_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("DEVICE =", device)
    # set default device to cuda if available
    torch.set_default_device(device)
    return device



def init_model(device: str="cpu", vocab_size: int=16384, embed_dim: int=2048, num_heads: int=4, dropout: float=0.1, num_layers: int=16):
    # parameters
    ff_hidden_layer = 8 * embed_dim

    # create input tensor
    #input_tensor = torch.randint(0, vocab_size, (context_length, batch_size)).to(device)

    # initialise the model
    model = gpt.MultiLayerTransformerDecoder(vocab_size, embed_dim, num_heads, ff_hidden_layer, dropout, num_layers, device)

    print(f"the model has {gpt.count_params(model):,} trainable parameters")
    
    return model



def train(data_src: str, model, device: str = "cpu", num_epochs: int=16, max_iters: int=1000, learning_rate: float=1e-4, context_length: int=1024, batch_size: int=2, eval_interval:int = 100):
    
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

    #print(train_data)

    train = torch.tensor(train_data).to(device)
    test = torch.tensor(test_data).to(device)
    
    print(train.dtype)
    
    model.to(device)
    
    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    for e in range(num_epochs):
        print("[", sep='')
        for step in range(max_iters):
            # generate batch
            idx, targets = dataset_tools.get_batch(train, context_length, batch_size, device)
            # forward pass
            logits = model(idx)
            print(logits.size())
            # get loss
            loss = F.cross_entropy(logits, targets)
            # backward pass
            loss.backward()
            # optimizer step
            optimiser.step()
            # zero gradient pass
            optimiser.zero_grad(set_to_none=True)
            
            # print progress
            if step % (eval_interval / 100) == 0:
                print("=", sep='')
            if step % eval_interval == 0:
                print("]")
                print(f"step {step}, loss {loss.item():.2f}")
                print("[", sep='')
        
        # evaluation per epoch
        @torch.no_grad()
        # evaluation pass - disable gradients
        def eval_loss():
            idx, targets = dataset_tools.get_batch(test, context_length, batch_size, device)
            logits = model(idx)
            loss = F.cross_entropy(logits, targets)
            print(f"step {step}, eval loss {loss.item():.2f}")
            return loss
        
        if step % eval_interval == 0: eval_loss().item()
        torch.save(model, "models/liamgpt_1.5B_e"+e)



def write_out(data: list, file: str):
    # writes data to be saved as a csv format
    with open("analytics/"+file, "a") as f:
        write_str = ""
        i = 1
        for d in data:
            write_str += str(d)
            if i<len(data):
                write_str += ","
            else:
                write_str += "\n"
            i+=1
        f.write(write_str)



#output = model(input_tensor)
#print(output.size)

if __name__ == '__main__':
    device = detect_device()
    
    n_vocab, encode, decode = dataset_tools.byte_pair_encoding(device)
    
    model = init_model(device, n_vocab, 2048, 4, 0.1, 16)
    #torch.save(model, "models/liamgpt_e0")
    #train("input.txt", model, "cpu", 1, 100, 1e-4, 1024, 2, 100)
    input = torch.tensor([0,245,248,6854,3157,152,123]).to(device)
    print(input.size())
    output = model(input)
    print(output.size())