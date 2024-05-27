import torch
import torch.nn as nn
from torch.nn import functional as F
import math

class Head(nn.Module):
    """One head of self-attention"""

    def __init__(self, n_embd, block_size, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B, T, C = x.shape
        k = self.key(x)  # (B,T,hs)
        q = self.query(x)  # (B,T,hs)
        # compute attention scores ("affinities")
        wei = (
            q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
        )  # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))  # (B, T, T)
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x)  # (B,T,hs)
        out = wei @ v  # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out


class MultiHeadAttention(nn.Module):
    """Multiple heads of self-attention in parallel"""

    def __init__(self, n_embd, block_size, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList(
            [Head(n_embd, block_size, head_size) for _ in range(num_heads)]
        )
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedFoward(nn.Module):
    """Simple linear layer followed by a non-linearity"""

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(0.2),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Transformer block: communication followed by computation"""

    def __init__(self, n_embd, n_head, block_size):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_embd, block_size, n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x,):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class GPTLanguageModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        n_embd: int,
        n_head: int,
        n_layer: int,
        block_size: int,
        weights=None,
    ):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.block_size = block_size
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(
            *[
                Block(n_embd, n_head=n_head, block_size=block_size)
                for _ in range(n_layer)
            ]
        )
        self.ln_f = nn.LayerNorm(n_embd)  # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size) # CHANGE

        self.weights = weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx:torch.Tensor, metadata_embedding: torch.Tensor = None, pad_code:int=0):
        B, T = idx.shape
        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx)  # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))  # (T,C) # this is the same for any batch?
        x = tok_emb + pos_emb  # (B,T,C)

        #########################################
        # ADD METADATA TO THE _EMBEDDED_ TOKENS #
        #########################################
        x = torch.cat((metadata_embedding.unsqueeze(1), x), 1) if metadata_embedding is not None else x

        #######################################
        # PASS EVERYTHING THROUGH TRANSFORMER #
        #######################################
        x = self.blocks(x)  # (B,T,C)
        x = self.ln_f(x)  # (B,T,C)
        logits = self.lm_head(x)  # (B,T,vocab_size)
        return logits
    
class LanguageDurationModel(GPTLanguageModel):
    def __init__(
        self,
        vocab_size: int,
        duration_embd: int,
        n_embd: int,
        n_head: int,
        n_layer: int,
        block_size: int,
        pad_code:int,
        weights=None,
        duration_dim:int=1,
        hidden_dim:int=128, 
    ):
        super().__init__(vocab_size, n_embd, n_head, n_layer, block_size, weights)
        # each token directly reads off the logits for the next token from a lookup table
        self.block_size = block_size
        self.n_embd = n_embd
        self.pad_code = pad_code
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(
            *[
                Block(n_embd + duration_embd, n_head=n_head, block_size=block_size)
                for _ in range(n_layer)
            ]
        )
        self.duration_in = nn.Sequential(
            nn.Linear(duration_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, duration_embd)
        )
        L_out_shape = math.floor((n_embd + duration_embd - 5)/(3)  + 1)
        # self.duration_out = nn.Linear(n_embd + duration_embd, 1)
        self.duration_pool = nn.AvgPool1d(5, stride=3)
        self.duration_out = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(L_out_shape, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1),
        )
        self.ln_f = nn.LayerNorm(n_embd + duration_embd)  # final layer norm
        self.lm_head = nn.Linear(n_embd + duration_embd, vocab_size) # CHANGE

        self.weights = weights

        # better init, not covered in the original GPT video, but important, will cover in followup video
        self.apply(self._init_weights)

    def forward(self, idx:torch.Tensor, metadata_embedding: torch.Tensor = None, durations: torch.Tensor = None):
        # DURATIONS ARE GONNA HAVE TO ACCOUNT FOR METADATA. GIVE IT A DURATION OF 0
        B, T = idx.shape
        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx)  # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))  # (T,C) # this is the same for any batch
        
        #########################################
        # Embed DURATIONS
        #########################################
        if durations is not None:
            if len(durations.shape) <= 2:
                durations = durations.unsqueeze(-1)

        dur_emb = self.duration_in(durations)  # (B,T,C)
        x = tok_emb + pos_emb  # (B,T,C)
        #########################################
        # ADD METADATA TO THE _EMBEDDED_ TOKENS #
        #########################################
        x = torch.cat((metadata_embedding.unsqueeze(1), x), 1) if metadata_embedding is not None else x
        #######################################
        # ADD DURATIONS TO THE _EMBEDDED_ TOKENS #
        #######################################
        x = torch.cat((x, dur_emb), 2) if dur_emb is not None else x
        #######################################
        # PASS EVERYTHING THROUGH TRANSFORMER #
        #######################################
        x = self.blocks(x)  # (B,T,C)
        x = self.ln_f(x)  # (B,T,C)
        logits = self.lm_head(x)  # (B,T,vocab_size)
        
        duration_output = self.duration_out(self.duration_pool(x))
        # time_remaining_output = self.time_remaining_out(self.time_remaining_out_pool(x))
        # time_remaining_output = self.time_remaining_out(x)
        # duration_output = self.duration_out(x)

        # return logits, duration_output, 
        return logits, duration_output

class EventMetadataLanguageModel(LanguageDurationModel):
    def __init__(
            self,
            vocab_size: int,
            event_embd: int,
            n_embd: int,
            n_head: int,
            n_layer: int,
            block_size: int,
            weights=None,
    ):
        super().__init__(vocab_size, event_embd, n_embd, n_head, n_layer, block_size, weights)
        # each token directly reads off the logits for the next token from a lookup table
        hidden_dim = 2 * n_embd
        self.block_size = block_size
        self.n_embd = n_embd
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(
            *[
                Block(n_embd * 2, n_head=n_head, block_size=block_size)
                for _ in range(n_layer)
            ]
        )
        self.duration_out = nn.Linear(n_embd * 2, 1)
        L_out_shape = math.floor((n_embd * 2 - 5)/(3)  + 1)
        # self.duration_out = nn.Linear(n_embd + duration_embd, 1)
        # self.duration_pool = nn.AvgPool1d(5, stride=3)
        # self.duration_out = nn.Sequential(
        #     nn.Dropout(0.1),
        #     nn.Linear(L_out_shape, hidden_dim),
        #     nn.ReLU(),
        #     nn.Dropout(0.2),
        #     nn.Linear(hidden_dim, 1),
        # )
        self.time_remaining_out = nn.Linear(n_embd * 2, 1)
        # self.time_remaining_out_pool = nn.AvgPool1d(5, stride=3)
        # self.time_remaining_out = nn.Sequential(
        #     nn.Dropout(0.1),
        #     nn.Linear(L_out_shape, hidden_dim),
        #     nn.ReLU(),
        #     nn.Dropout(0.2),
        #     nn.Linear(hidden_dim, 1),
        # )

        self.ln_f = nn.LayerNorm(n_embd * 2)  # final layer norm
        self.lm_head = nn.Linear(n_embd * 2, vocab_size) # CHANGE

        self.weights = weights

        # better init, not covered in the original GPT video, but important, will cover in followup video
        self.apply(self._init_weights)

    def forward(self, idx:torch.Tensor, metadata_embedding: torch.Tensor, event_metadata_embedding: torch.Tensor):
        # idx has shape(B, T-1)
        # metadata embedding has shape (B, n_embd)
        # event_metadata_embedding has shape (B, T, n_embd)

        #########################################
        # embed tokens
        #########################################
        B, T = idx.shape
        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx)  # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))  # (T,C) # this is the same for any batch
        x = tok_emb + pos_emb  # (B,T,C)


        #########################################
        # ADD case level METADATA TO THE _EMBEDDED_ TOKENS #
        #########################################
        x = torch.cat((metadata_embedding.unsqueeze(1), x), 1) if metadata_embedding is not None else x
             
        #######################################
        # add event level metadata to the _embedded_ tokens #
        #######################################
        x = torch.cat((x, event_metadata_embedding), 2) if event_metadata_embedding is not None else x

        #######################################
        # PASS EVERYTHING THROUGH TRANSFORMER #
        #######################################
        x = self.blocks(x)  # (B,T,C)
        x = self.ln_f(x)  # (B,T,C)

        logits = self.lm_head(x)  # (B,T,vocab_size)
        duration_output = self.duration_out(x)
        time_remaining_output = self.time_remaining_out(x)

        return logits, duration_output, time_remaining_output
    
