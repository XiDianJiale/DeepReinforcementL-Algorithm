#以下是我第一次编程时候的战场实录，我的体会主要在于对创新模型的编程方式和组件安排和利用
import torch
#import torch.fuction as F
import torch.nn.functional as F
import numpy
import matplotlib as plt
import torchvision

import torch.nn as nn



class Patch(img_tensor,batch_size):
    self.__init__()
    batch = img


class ViT(nn.Module):
    self.__init__
    def __init__(self,img_size,patch_size,depth,dim,embed_dim,mlp_ratio,num_class,num_head):
        super(ViT,self).__init__()
        self.patch_embed = Patch_embed(patch_size,img_size,embed_dim)
        self.pos_embed =
        self.transformer = Transformer(depth,embed_dim,num_head,mlp_ratio)
        self.head = nn.Linear(embed_dim,num_class)  #参数num_class指明了该ViT用于cv分类任务

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

    def forward(self,x):
        x = self.patch_embed(x)
        x = x + self.pos_embed(x)
        x = self.transformer(x)

#注意轮子和模型的区别，事实上，patchEmbed和Transformer的编程已经很成熟，这些方法在Vit类中已经使用，至此Vit的东西已经完全体现
#那么对上面的代码进行总结：上述代码描述了一个网络（为ViT结构），实现了输入x，x的位置嵌入，x的patchEmbed，然后经过网络（其中有注意力的计算和机制，最后输出分类的结果
#下面是对于Patchembed和Transformer的实现练习

class Patch_embed(nn.Module):
    def __init__(self,patch_size,img_size,embed_dim):
        self.patch_size = patch_size
        self.img_size = img_size
        self.num_patch =(img_size // patch_size)  ** 2

        self.proj = nn.Conv2d(3,embed_dim,kernel_size=patch_size,stride=patch_size)

    def forward(self,x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1,2)
        return x


class Attention(nn.Module):
    def __init__(self,embed_dim,num_heads):
        super(Attention,self).__init__()
        self.num_heads = num_heads
        head_dim = embed_dim // num_heads
        self.scale = head_dim **(-0.5)  #所谓缩放因子

        self.qkv = nn.Linear(embed_dim,embed_dim*3)  #一次计算把输入x映射到qkv三个层
        self.proj = nn.Linear( (embed_dim//num_heads*num_heads) ,embed_dim)#作用是强制让x经过注意力头后输出与输入维度一致

    def forward(self,x):
        B , N , C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q , k ,v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x

class MLP(nn.Module):
    def __init__(self,in_features,hidden_features,out_features,):
        super().__init__()

        self.fc1 = nn.Linear(in_features,hidden_features)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(hidden_features,out_features)
        self.dropout = nn.Dropout()  #认知增量：dropout如何在pytorch中使用？


    def forward(self,x):  #错误二：没有第一时间反应过来forward需要传入的参数和定义的方法
     #   self.
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.act(x)
        x = self.dropout(x)
        return x


#3/8编程练习：transformer组构
class TransformerBlock(nn.Module):
    def __init__(self,embed_dim,num_head,mlp_ratio = 4):
        super(TransformerBlock,self).__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.atten1 = Attention(embed_dim,num_head)
        self.norm2 = nn.LayerNorm(embed_dim)
    #    self.atten2 = Attention(embed_dim)
        self.mlp = MLP(embed_dim,int(embed_dim * mlp_ratio))

    def forward(self,x):
        x = self.atten1(self.norm1(x)) + x
 #      x = self.atten2(self.mlp(x)) +x
        x = self.mlp(self.norm2(x)) + x
        return x


class Transformer(nn.Module):  # 本质上是Transformer编码器
    def __init__(self,depth,embed_dim,num_head,mlp_ratio):   #这里定义了Block中需要传入的参数，有一种小block给大block进行call的感觉
        super(Transformer,self).__init__()
        self.layers = nn.ModuleList([TransformerBlock(embed_dim,num_head,mlp_ratio) for _ in range(depth)])

    def forward(self,x):
            for layers in self.layers:
                x = layers(x)
            return x

#上述代码中存在很多问题，以下是对我的纠正以及需要重点关注的地方(coding练习耗时漫长，这些反馈来之不易

1.混淆了模型定义与实现细节（如在ViT类中直接处理位置编码，而未将其抽象为独立模块）
2.应该保有的bigPictureViT
├── patch_embed
├── pos_embed (未实现)
├── transformer
│   └── TransformerBlock × depth
└── head
