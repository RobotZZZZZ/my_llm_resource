import torch
import torch.nn as nn

# 层归一化是沿着emb维度进行归一化
class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        # 避免分母为0
        self.eps = 1e-5
        # 每一个emb的维度都有独立的scale和shift
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        # 层归一化
        # 一般的输入shape是[batch_size, num_token, emb_dim]
        # 沿着emb维度进行归一化
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        # 缩放和平移
        return self.scale * norm_x + self.shift

# GELU激活
class GELU(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        # 近似函数
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) * 
            (x + 0.044715 * torch.pow(x, 3))
        ))

# 前馈层
class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)

# 利用权重拆分实现多头注意力
class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()

        # 确保d_out是否能被num_heads整除
        assert (d_out % num_heads == 0), "d_out must be divisible by num_heads"

        # 参数初始化
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

        # 增加线性层，不改变维度
        self.out_proj = nn.Linear(d_out, d_out)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))
    
    def forward(self, x):
        b, num_tokens, d_in = x.shape

        # 计算keys, queries, values
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        # 将keys, queries, values拆分成多个head
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
        # 转置，将num_heads移到前面，方便后续计算
        # shape = b, num_heads, num_tokens, head_dim
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # 计算attn weights
        # shape = b, num_heads, num_tokens, num_tokens
        attn_scores = queries @ keys.transpose(2, 3)
        # mask未来信息, 避免信息泄露，同时适配不同token长度
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        # 归一化
        attn_weights = torch.softmax(attn_scores / (keys.shape[-1] ** 0.5), dim=-1)

        # 使用dropout防止过拟合
        attn_weights = self.dropout(attn_weights)
        
        # 计算上下文向量
        # shape = b, num_tokens, num_heads, head_dim
        context_vec = (attn_weights @ values).transpose(1, 2)
        # 调整上下文形状
        # shape = b, num_tokens, d_out(=num_heads * head_dim)
        # 在进行view之前，需要先进行contiguous()，否则会报错
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        # 线性层，增加一次变换
        context_vec = self.out_proj(context_vec)

        return context_vec

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # 多头注意力层
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"],
        )
        # 前馈层
        self.ff = FeedForward(cfg)
        # 归一化层
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        # dropout
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])
    
    def forward(self, x):
        # attn+shortcut
        shortcut = x
        x = self.norm1(x)
        # attn里的dropout是对attn_weights
        x = self.att(x)
        # 这里的dropout是对context_vecs
        x = self.drop_shortcut(x)
        x = x + shortcut

        # ffn+shortcut
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        return x