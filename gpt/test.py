import torch
import torch.nn as nn
import math

# Define some hyperparameters
d_model = 512 # hidden size of the model
n_heads = 8 # number of attention heads
d_ff = 2048 # hidden size of the feed-forward layer
dropout = 0.1 # dropout rate

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

class PositionalEncoding(T.nn.Module):
  def __init__(self, d_model, dropout=0.1, max_len=5000):
    super(PositionalEncoding, self).__init__()
    self.dropout = T.nn.Dropout(p=dropout)
    pe = T.zeros(max_len, d_model)
    position = T.arange(0, max_len, dtype=T.float).unsqueeze(1)
    div_term = T.exp(T.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = T.sin(position * div_term)
    pe[:, 1::2] = T.cos(position * div_term)
    pe = pe.unsqueeze(0).transpose(0, 1)
    self.register_buffer('pe', pe)

  def forward(self, x):
    x = x + self.pe[:x.size(0), :]
    return self.dropout(x)

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

# Define a transformer
class Transformer(nn.Module):
    
    def __init__(self,src_vocab,trg_vocab,d_model,n_heads,d_ff,N):
        super().__init__()

        self.encoder = Encoder(src_vocab,d_model,n_heads,d_ff,N)
        self.decoder = Decoder(trg_vocab,d_model,n_heads,d_ff,N)
        self.src_embed = nn.Embedding(src_vocab,d_model)
        self.trg_embed = nn.Embedding(trg_vocab,d_model)
        self.pos_enc = PositionalEncoding(d_model)
        self.fc_out = nn.Linear(d_model,trg_vocab)

    
    def forward(self,x,y,x_mask=None,y_mask=None):

        
        x= self.src_embed(x) 
        x= self.pos_enc(x) 
        y= self.trg_embed(y) 
        y= self.pos_enc(y) 

        
        x= self.encoder(x,x_mask) 
        y= self.decoder(x,y,x_mask,y_mask) 

        
        out= self.fc_out(y) 

        return out