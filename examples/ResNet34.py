import torch.nn as nn
from torch_combinators import residual, seq, lift, probe, repeat

def basic_block(n):
    return residual(
        nn.Conv2d(n, n, kernel_size=3, stride=1, padding=1, bias=False)
        |seq| nn.BatchNorm2d(n)
        |seq| nn.ReLU(inplace=True)
        |seq| nn.Conv2d(n, n, kernel_size=3, stride=1, padding=1, bias=False)
        |seq| nn.BatchNorm2d(n)
    ) |seq| nn.ReLU(inplace=True)

def reshaping_block(a,b): 
  f= nn.Conv2d(a, b,\
            kernel_size=3,\
            stride=2,\
            padding=1,\
            bias=False)\
     |seq| nn.BatchNorm2d(b)\
     |seq| nn.ReLU(inplace=True)\
     |seq| nn.Conv2d(b, b, \
                  kernel_size=3, \
                  stride=1,\
                  padding=1,\
                  bias=False)\
     |seq| nn.BatchNorm2d(b)
  downsample = nn.Conv2d(a, b,\
                      kernel_size=1,\
                      stride=2,\
                      bias=False)\
               |seq| nn.BatchNorm2d(b)
  recombine=lift(lambda x:x[1][0]+x[1][1])
  return probe(recombine,f,downsample)\
         |seq| nn.ReLU(inplace=True)

def my_resnet34(numClasses):
    entry_block=nn.Conv2d(3, 64, kernel_size=7, stride=2,padding=3,bias=False)\
                   |seq| nn.BatchNorm2d(64)\
                   |seq| nn.ReLU(inplace=True)\
                   |seq| nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    block1 = repeat(3,basic_block(64))
    block2 = reshaping_block(64,128) |seq| repeat(3,basic_block(128))
    block3 = reshaping_block(128,256) |seq| repeat(5,basic_block(256)) 
    block4 = reshaping_block(256,512) |seq| repeat(2,basic_block(512)) 
    exit_block = nn.AdaptiveAvgPool2d((1, 1)) |seq| nn.Flatten(1) |seq| nn.Linear(512, numClasses)

    return entry_block |seq| block1 |seq| block2 |seq| block3 |seq| block4 |seq| exit_block
           
           
print(my_resnet34(7))