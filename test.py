from header import *
from module import *
from dataset import *
import traceback
import os

vocab_path = "./data/vocab.txt"
weight_file = r"./weight/weight.pkl"


# 网络
net = GPT2().to(torch.device(cfg.device))
if os.path.exists(weight_file):
    print("有网络参数文件")
    if os.path.getsize(weight_file) != 0:
        net.load_state_dict(torch.load(weight_file))
        print("加载保存的参数成功")
    else:
        print("网络文件里没有网络参数")
net.eval()


with open(vocab_path, "r+", encoding="utf-8") as f:
    tokens = f.read().split()

# 给定一个开始词
print("请输入开始词:\n")
x = input()
x_index = []
for m, j in enumerate(x):
    x_index.append(tokens.index(j))

os = []
# 给定一个开始词
x = torch.tensor([x_index]).to(torch.device(cfg.device))
# 起始位置
p = torch.tensor([[a for a in range(len(x_index))]]).to(torch.device(cfg.device))

with open(vocab_path, "r+", encoding="utf-8") as f:
    tokens = f.read().split()
for i in range(len(x_index)-1, 24):
    y = net(x, p)
    # print(y.shape)
    y = y[:, -1:]
    # print(y.shape)
    # 所有字中，选概率最大的8个
    v, y = torch.topk(y, 8, dim=int(-1))
    v, y = v.reshape(-1, 8), y.reshape(-1, 8)
    # 在8个候选字中通过概率去选一个， 增加文章的随机性
    v = torch.multinomial(torch.softmax(v, dim=-1), 1)
    # 通过索引拿到取的值
    y = torch.gather(y, -1, v)
    # 将生成的字拼接到前面去
    x = torch.cat([x, y], dim=1)
    # 加一个位置
    p = torch.tensor([range(i + 2)]).to(torch.device(cfg.device))

    # print("生成{0}个字如下：\n".format(i+1))
for m, j in enumerate(x.detach().cpu()[0]):
    print(tokens[j], end="")
print("\n")
