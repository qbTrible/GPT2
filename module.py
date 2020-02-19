from header import *
import torch.nn as nn
import config as cfg


class Attention(nn.Module):

    def __init__(self, isMask=False):
        super().__init__()
        self.dk = (cfg.embed_dim // cfg.head_num) ** 0.5
        self.isMask = isMask

        self.c_attn = nn.Linear(cfg.embed_dim, cfg.embed_dim * 3)   # 扩充为3v（q，k，v）

        self.attn_drop = nn.Dropout(0.1)
        self.resi_drop = nn.Dropout(0.1)

        self.c_proj = nn.Linear(cfg.embed_dim, cfg.embed_dim)    # 线性层

        if self.isMask:
            self.register_buffer("mask", torch.tril(torch.ones(cfg.pos_num, cfg.pos_num)))

    def forward(self, x):    # x：NSV
        x = self.c_attn(x)   # NS(3V)
        x = x.reshape(*x.shape[:-1], cfg.head_num, -1)   # N，S，12，60*3（NSHV）
        x = x.transpose(-2, -3)  # NHSV
        q, k, v = x.chunk(3, dim=-1)
        w = (q @ k.transpose(-1, -2)) / self.dk #NHSS
        if self.isMask:
            mask = self.mask[0:w.size(-2), 0:w.size(-1)]
            w = w * mask - (1 - mask) * 1e5
        w = torch.softmax(w, dim=-1)
        w = self.attn_drop(w)

        a = w @ v   # NHSV（N，H，S，64）

        a = a.transpose(-2, -3)   # NSHV
        a = a.reshape(*a.shape[:-2], cfg.embed_dim)   # NSV

        h = self.c_proj(a)
        h = self.resi_drop(h)

        return h


class Block(nn.Module):

    def __init__(self, isMask=False):
        super().__init__()
        self.layer_normal_1 = nn.LayerNorm(cfg.embed_dim)
        self.attention = Attention(isMask)
        self.layer_normal_2 = nn.LayerNorm(cfg.embed_dim)
        self.proj = nn.Sequential(
            nn.Linear(cfg.embed_dim,  2 * cfg.embed_dim),
            nn.LeakyReLU(),
            nn.Linear(2 * cfg.embed_dim, cfg.embed_dim),
        )

        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        h = self.layer_normal_1(x)
        a = self.attention(h)
        a = a + x
        a = self.layer_normal_2(a)
        h = self.proj(a)
        h = self.dropout(h)
        y = h + a
        return y


class GPT2(nn.Module):

    def __init__(self):
        super().__init__()

        self.vocab_embed = nn.Embedding(cfg.vocab_num, cfg.embed_dim)
        self.pos_embed = nn.Embedding(cfg.pos_num, cfg.embed_dim)
        self.blocks = []
        for _ in range(cfg.block_num):
            self.blocks.append(Block(isMask=False))
        self.drop = nn.Dropout(0.1)
        self.sequential = nn.Sequential(*self.blocks)
        self.output_layer = nn.Linear(cfg.embed_dim, cfg.vocab_num, bias=False)

    def forward(self, x, p):
        e = self.vocab_embed(x)
        p = self.pos_embed(p)
        h = self.drop(e + p)
        h = self.sequential(h)
        return self.output_layer(h)

# class Attention(nn.Module):
#     def __init__(self, isMask=True):
#         super(Attention, self).__init__()
#
#         # dk = 词的维度/词的头数
#         self.dk = (cfg.embed_dim // cfg.head_num) ** 0.5
#
#         # 把一个词分解成Q，K，V
#         self.c_attn = nn.Linear(cfg.embed_dim, cfg.embed_dim * 3)
#
#         self.attn_drop = nn.Dropout(0.1)
#         self.resi_drop = nn.Dropout(0.1)
#
#         # 接一个线性曾提供参数，使得词向量可训练
#         self.c_proj = nn.Linear(cfg.embed_dim, cfg.embed_dim)
#
#         self.isMask = isMask
#         if self.isMask:
#             # 定义一个下三角掩码，写在这儿会当成权重保存，不用被训练，自动传入网络
#             self.register_buffer("mask", torch.tril(torch.ones(cfg.pos_num, cfg.pos_num)))
#         # 如果掩码这样写，需要手动传入数据
#         # self.mask = (数据).cuda()
#
#     def forward(self, x):
#         # x形状(N,S,V)，N代表多少个句子，S代表多少个词，V代表每个词的维度
#         x = self.c_attn(x)
#
#         # (N,S,V)——>(N,S,H,V)(N,S, H, V/H*3)
#         x = x.reshape(*x.shape[:-1], cfg.head_num, -1)
#
#         # (N,S,H,V)(N,S,H,V/H*3)——>(N,H,S,V)(N,H,S,V/H*3)
#         x = x.transpose(-2, -3)
#
#         # (N,H,S,V)(N,H,S,V/H*3) ——>(N,H,S,V)(N,H,S,V/H))
#         q, k, v = x.chunk(3, dim=-1)
#
#         # (N,H,S,(V/H))@(N,H,(V/H),S)=(N,H,S,S)
#         w = (q @ k.transpose(-1, -2)) / self.dk
#
#         # 掩码形状（S,S）
#         if self.isMask:
#             mask = self.mask[0:w.size(-2), 0:w.size(-1)]
#             # 将w的上三角全部变为负无穷小，有利于做softmax归一化
#             w = w * mask - (1 - mask) * 1e8
#
#         # 归一化得到权重
#         w = torch.softmax(w, dim=-1)
#
#         # dropout
#         w = self.attn_drop(w)
#
#         # (N,H,S,S)@(N,H,S,(V/H))-->(N,H,S,V)(N,H,S,(V/H))
#         a = w @ v
#
#         """和合并形状"""
#         # (N,H,S,(V/H))-->(N,S,H,(V/H))
#         a = a.transpose(-2, -3)
#         # (N,S,H,(V/H))-->(N,S,V)
#         a = a.reshape(*a.shape[:-2], cfg.embed_dim)
#
#         # 全连接层提供参数
#         h = self.c_proj(a)
#
#         # dropout
#         h = self.resi_drop(h)
#
#         return h
#
#
# class Block(nn.Module):
#
#     def __init__(self):
#         super(Block, self).__init__()
#
#         # 数据传进来归一化
#         self.layer_normal_1 = nn.LayerNorm(cfg.embed_dim)
#
#         # 注意力
#         self.attention = Attention()
#
#         # 值控制到0~1
#         self.layer_normal_2 = nn.LayerNorm(cfg.embed_dim)
#
#         # 全连接层，扩大参数量
#         self.proj = nn.Sequential(nn.Linear(cfg.embed_dim, cfg.multi * cfg.embed_dim),
#                                   nn.LeakyReLU(),
#
#                                   nn.Linear(cfg.multi * cfg.embed_dim, cfg.embed_dim)
#                                   )
#
#         # dropout
#         self.dropout = nn.Dropout(0.1)
#
#     def forward(self, x):
#         h = self.layer_normal_1(x)
#         a = self.attention.forward(h)
#
#         # 加一个残差
#         a = a + x
#
#         a = self.layer_normal_2(a)
#
#         h = self.proj(a)
#
#         h = self.dropout(h)
#
#         # 加一个残差
#         y = h + a
#
#         return y
#
#
# class GPT2(nn.Module):
#
#     def __init__(self):
#         super(GPT2, self).__init__()
#
#         # 定义一个字典
#         self.vocab_embed = nn.Embedding(cfg.vocab_num, cfg.embed_dim)
#
#         # 定义一个位置编码
#         self.pos_embed = nn.Embedding(cfg.pos_num, cfg.embed_dim)
#
#         # 定义一个类型编码
#         self.type_embed = nn.Embedding(cfg.type_num, cfg.embed_dim)
#
#         # 叠6层block
#         self.blocks = []
#         for _ in range(cfg.block_num):
#             self.blocks.append(Block())
#
#         # dropout
#         self.drop = nn.Dropout(0.1)
#
#         # 将叠的block形成一个网络
#         self.sequential = nn.Sequential(*self.blocks)
#
#         # 全连接输出层
#         self.output_layer = nn.Linear(cfg.embed_dim, cfg.vocab_num, bias=False)
#
#     def forward(self, x, p, ):
#         # 对输入进行词向量编码
#         e = self.vocab_embed(x)
#
#         # 对输入进行位置编码
#         p = self.pos_embed(p)
#
#         # # 对输入进行类型编码
#         # t = self.type_embed(t)
#         # h = self.drop(e + p + t)
#         h = self.drop(e + p)
#
#         h = self.sequential(h)
#
#         return self.output_layer(h)
