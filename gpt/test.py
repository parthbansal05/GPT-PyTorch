import torch
import torch.nn as nn
import math

# Define some hyperparameters
d_model = 512 # hidden size of the model
n_heads = 8 # number of attention heads
d_ff = 2048 # hidden size of the feed-forward layer
dropout = 0.1 # dropout rate

device = 'cuda' if torch.cuda.is_available() else 'cpu'



# Define a scaled dot-product attention function
def scaled_dot_product_attention(q, k, v, mask=None):
    # q: query tensor of shape (batch_size, n_heads, seq_len, d_head)
    # k: key tensor of shape (batch_size, n_heads, seq_len, d_head)
    # v: value tensor of shape (batch_size, n_heads, seq_len, d_head)
    # mask: optional mask tensor of shape (batch_size, 1, 1, seq_len)

    # Compute the attention scores
    scores = torch.matmul(q, k.transpose(-2,-1)) / math.sqrt(d_model) # shape: (batch_size,n_heads ,seq_len ,seq_len)

    # Apply the mask if given
    if mask is not None:
        scores = scores.masked_fill(mask == 0 , -float('inf'))

    # Compute the attention weights
    weights = torch.softmax(scores , dim=-1) # shape: (batch_size,n_heads ,seq_len ,seq_len)

    # Apply dropout if given
    if dropout is not None:
        weights = nn.Dropout(dropout)(weights)

    # Compute the output tensor by multiplying the weights with the values
    output = torch.matmul(weights , v) # shape: (batch_size,n_heads ,seq_len ,d_head)

    return output


# Define a multi-head self-attention layer
class MultiHeadSelfAttention(nn.Module):
    
    def __init__(self,d_model,n_heads):
        super().__init__()

        assert d_model % n_heads == 0
        
        self.d_model = d_model 
        self.n_heads = n_heads 
        self.d_head = d_model // n_heads 

        self.q_proj = nn.Linear(d_model,d_model) 
        self.k_proj = nn.Linear(d_model,d_model) 
        self.v_proj = nn.Linear(d_model,d_model) 

        self.out_proj = nn.Linear(d_model,d_model) 

    
    def forward(self,x,x_mask=None):
        
        batch_size , seq_len , _= x.shape 

        
        q= self.q_proj(x).view(batch_size,-1,self.n_heads,self.d_head).transpose(1,-3) 
       
        k= self.k_proj(x).view(batch_size,-1,self.n_heads,self.d_head).transpose(1,-3) 
        
        v= self.v_proj(x).view(batch_size,-1,self.n_heads,self.d_head).transpose(1,-3)

        # Apply scaled dot-product attention
        att = scaled_dot_product_attention(q,k,v,x_mask) # shape: (batch_size,n_heads ,seq_len ,d_head)

        # Concatenate and project the outputs
        att = att.transpose(1,-3).contiguous().view(batch_size,-1,self.d_model) # shape: (batch_size,seq_len,d_model)
        out = self.out_proj(att) # shape: (batch_size,seq_len,d_model)

        return out

# Define a feed-forward layer
class FeedForward(nn.Module):
    
    def __init__(self,d_model,d_ff):
        super().__init__()

        self.linear1 = nn.Linear(d_model,d_ff)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(d_ff,d_model)

    
    def forward(self,x):

        
        x= self.linear1(x) 
       
        x= self.relu(x) 
        
        x= self.linear2(x) 

        return x

# Define a residual connection with layer normalization
class ResidualNorm(nn.Module):
    
    def __init__(self,d_model):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    
    def forward(self,x,res):
        
        x= self.norm(x + res) 
       
        x= self.dropout(x) 

        return x

# Define an encoder block
class EncoderBlock(nn.Module):
    
    def __init__(self,d_model,n_heads,d_ff):
        super().__init__()

        self.att = MultiHeadSelfAttention(d_model,n_heads)
        self.ff = FeedForward(d_model,d_ff)
        self.resnorm1 = ResidualNorm(d_model)
        self.resnorm2 = ResidualNorm(d_model)

    
    def forward(self,x,x_mask=None):

        
        att_out = self.att(x,x_mask) 
        
        x= self.resnorm1(att_out,x) 
        
        ff_out = self.ff(x) 
        
        x= self.resnorm2(ff_out,x) 

        return x

# Define a decoder block
class DecoderBlock(nn.Module):
    
    def __init__(self,d_model,n_heads,d_ff):
        super().__init__()

        self.self_att = MultiHeadSelfAttention(d_model,n_heads)
        self.enc_att = MultiHeadSelfAttention(d_model,n_heads)
        self.ff = FeedForward(d_model,d_ff)
        self.resnorm1 = ResidualNorm(d_model)
        self.resnorm2 = ResidualNorm(d_model)
        self.resnorm3 = ResidualNorm(d_model)

    
    def forward(self,x,y,y_mask=None,x_mask=None):

        
        att_out1 = self.self_att(y,y_mask) 
        
        y= self.resnorm1(att_out1,y) 
        
        att_out2 = self.enc_att(x,y,x_mask) 
        
        y= self.resnorm2(att_out2,y) 
        
        ff_out = self.ff(y) 
        
        y= self.resnorm3(ff_out,y) 

        return y

# Define an encoder
class Encoder(nn.Module):
    
    def __init__(self,d_model,n_heads,d_ff,N):
        super().__init__()

        self.layers = nn.ModuleList([EncoderBlock(d_model,n_heads,d_ff) for _ in range(N)])
        self.norm = nn.LayerNorm(d_model)

    
    def forward(self,x,x_mask=None):

        
        for layer in self.layers:
            x= layer(x,x_mask) 
        
        x= self.norm(x) 

        return x

# Define a decoder
class Decoder(nn.Module):
    
    def __init__(self,d_model,n_heads,d_ff,N):
        super().__init__()

        self.layers = nn.ModuleList([DecoderBlock(d_model,n_heads,d_ff) for _ in range(N)])
        self.norm = nn.LayerNorm(d_model)

    
    def forward(self,x,y,x_mask=None,y_mask=None):

        
        for layer in self.layers:
            y= layer(x,y,y_mask,x_mask) 
        
        y= self.norm(y) 

        return y
    
class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model,device=device)
        position = torch.arange(0, max_len, dtype=torch.float, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, device=device).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
    

# Define a transformer
class Transformer(nn.Module):
    
    def __init__(self,src_vocab,trg_vocab,d_model,n_heads,d_ff,N):
        super().__init__()

        # self.encoder = Encoder(d_model,n_heads,d_ff,N)
        self.decoder = Decoder(d_model,n_heads,d_ff,N)
        self.src_embed = nn.Embedding(src_vocab,d_model)
        self.trg_embed = nn.Embedding(trg_vocab,d_model)
        self.pos_enc = PositionalEncoding(d_model)
        self.fc_out = nn.Linear(d_model,trg_vocab)

    
    def forward(self,x,y,x_mask=None,y_mask=None):

        
        x= self.src_embed(x) 
        x= self.pos_enc(x) 
        y= self.trg_embed(y) 
        y= self.pos_enc(y) 

        
        # x= self.encoder(x,x_mask) 
        y= self.decoder(x,y,x_mask,y_mask) 

        
        out= self.fc_out(y) 

        return out
    

# ===================================================================================================
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - 256, (64,), device=device)
    x = torch.stack([data[i:i+256] for i in ix])
    y = torch.stack([data[i+1:i+256+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(100)
        for k in range(100):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
src_vocab = len(chars)    
trg_vocab= src_vocab
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long, device=device)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]



N=6

model = Transformer(src_vocab,trg_vocab,d_model,n_heads,d_ff,N)
m= model.to(device)
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

for iter in range(2000):

    # every once in a while evaluate the loss on train and val sets
    if iter % 100 == 0 or iter == 2000 - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))