import torch
import torch.nn as nn
from torch.nn import functional as F
from lib import MultiHeadAttention, FeedFoward, Encoder, DecoderBlock, estimate_loss, get_batch, data_split, train,EncoderBlock

# hyperparameters
batch_size = 64  # how many independent sequences will we process in parallel?
block_size = 256  # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2
# ------------
print(device)

torch.manual_seed(1337)

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
encoder = Encoder(text)
encoder_e = Encoder(text)
# Train and test splits
data = torch.tensor(encoder.encode(text), dtype=torch.long)

train_data, val_data = data_split(data, 0.9)


class GPTLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(encoder.vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.token_embedding_table_e = nn.Embedding(encoder_e.vocab_size, n_embd)
        self.position_embedding_table_e = nn.Embedding(block_size, n_embd)
        self.decoderblocks = nn.Sequential(
            *[DecoderBlock(n_embd, n_head=n_head, block_size=block_size, dropout=dropout) for _ in range(n_layer)])
        self.encoderblocks = nn.Sequential(
            *[EncoderBlock(n_embd, n_head=n_head, block_size=block_size, dropout=dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)  # final layer norm
        self.lm_head = nn.Linear(n_embd, encoder.vocab_size)

        # better init, not covered in the original GPT video, but important, will cover in followup video
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, edx, targets=None):
        B, T = idx.shape
        B1, T1 = edx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx)  # (B,T,C)
        tok_emb_e = self.token_embedding_table_e(edx)  # (B,T,C)
        pos_emb = self.position_embedding_table(
            torch.arange(T, device=device))  # (T,C)
        pos_emb_e = self.position_embedding_table_e(
            torch.arange(T1, device=device))  # (T,C)
        x = tok_emb + pos_emb  # (B,T,C)
        x1 = tok_emb_e+ pos_emb_e  # (B,T,C)
        y = self.encoderblocks(x1)
        x = self.decoderblocks(x,y,y)  # (B,T,C)
        x = self.ln_f(x)  # (B,T,C)
        logits = self.lm_head(x)  # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, edx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            edx_cond = edx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond,edx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx


model = GPTLanguageModel()
m = train(model, device, learning_rate, max_iters, eval_interval,
          eval_iters, batch_size, block_size, train_data, val_data)


# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(encoder.decode(m.generate(context, max_new_tokens=500)[0].tolist()))
#open('more.txt', 'w').write(encoder.decode(m.generate(context, max_new_tokens=10000)[0].tolist()))
