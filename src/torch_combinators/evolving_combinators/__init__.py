"""This package is a set of evolving combinators, used as demonstration in the paper. For standard use we recommend the main library. """

import torch.nn as nn
import torch

class Infix:
    def __init__(self, func):
        self.func = func
    def __or__(self, other):
        return self.func(other)
    def __ror__(self, other):
        return Infix(lambda x: self.func(other, x))

@Infix
def seq(a, b):
    return nn.Sequential(a,b)


class residual1(nn.Module):
    def __init__(self, f):
        super().__init__()
        self.f=f
    
    def forward(self, x):
        return self.f(x)+x

class skip1(nn.Module):
  def __init__(self,r,f):
    super().__init__()
    self.f=f
    self.r=r 

  def forward(self,x):
    return self.r(self.f(x),x)

class probe(nn.Module):
  def __init__(self,r,f,g):
    super().__init__()
    self.f=f
    self.g=g
    self.r=r 

  def forward(self,x):
    return self.r((x,(self.f(x),self.g(x))))

def residual2(f):
  return skip1(lambda x:x[0]+x[1],f)

def skipC(f):
  return skip1(lambda x:torch.cat(x, 0),f)

class linear_recombine(nn.Module):
  def __init__(self,i_shape,o_shape):
    super().__init__()
    self.r=nn.Linear(i_shape,o_shape) |seq| nn.Sigmoid()

  def forward(self,x):
    U=self.r(x[0])
    return (x[1][1] * U) + (x[1][1]*(1-U))

def residual_gate(i_shape,o_shape,f):
  return probe(linear_recombine(i_shape,o_shape),
               f,nn.Identity())

def basic_gate1(i_shape,o_shape,f,g):
  return probe(linear_recombine(i_shape,o_shape), 
               f,g)



