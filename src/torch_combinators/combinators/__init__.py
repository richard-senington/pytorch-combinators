import torch.nn as nn
import torch
import copy

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

class lift(nn.Module):
  def __init__(self,f,innerNetworks=[]):
    super().__init__()
    self.f=f
    self.inner=nn.ModuleList(innerNetworks)

  def forward(self,x):
    return self.f(x)

class probe(nn.Module):
  def __init__(self,r,f,g):
    super().__init__()
    self.f=f
    self.g=g
    self.r=r 

  def forward(self,x):
    return self.r((x,(self.f(x),self.g(x))))

def repeat(n,f):
    x=copy.deepcopy(f)
    for _ in range(n-1):
      x=x |seq| copy.deepcopy(f)
    return x

class duplicate(nn.Module):
  def __init__(self,n,f):
    super().__init__()
    self.fs=nn.ModuleList([copy.deepcopy(f) for _ in range(n)])

  def forward(self,x):
    return [f(x) for f in self.fs]

def skip(r,f):
  r2=lift(lambda x:r((x[1][0],x[1][1])),
          innerNetworks=[r])
  return probe(r2,f,nn.Identity())

def residual(f):
  return skip(lift(lambda x:x[0]+x[1]),f)

class first(nn.Module):
  def __init__(self,f):
    super().__init__()
    self.f=f 

  def forward(self,x):
    return (self.f(x[0]),x[1])

def basic_gate(i_shape,o_shape,f,g):
  r = first(nn.Linear(i_shape,o_shape) |seq| nn.Sigmoid)\
      |seq|\
      lift(lambda x:x[0]*x[1][0]+(1-x[0])*x[1][1])

  return probe(r,f,g)

def auto_encoder(encode,decode,bottleneck=None):
  if bottleneck is None:
    return encode |seq| decode
  else:
    return encode |seq| bottleneck |seq| decode

class fan_l(nn.Module):
  def __init__(self,fs):
    super().__init__()
    self.fs=nn.ModuleList(fs)
  def forward(self,x):
    return (x,[f(x) for f in self.fs])

class fan_t(nn.Module):
  def __init__(self,*args):
    super().__init__()
    self.fs=nn.ModuleList(args)
  def forward(self,x):
    return (x,tuple([f(x) for f in self.fs]))

class recurrent(nn.Module):
  def __init__(self, initial_f,f , reset_manager=None):
    super().__init__()
    self.f = f
    self.memory=initial_f()
    if reset_manager is not None:
        reset_manager.manage_for_reset(self,initial_f)
    
  def forward(self, x):
    self.memory = self.f((x, self.memory))
    return self.memory

class ResetManager():
    """Only to manage resets in recurrent networks"""
    def __init__(self):
        self.managed=[]
    
    def manage_for_reset(self,n,f):
        self.managed.append((n,f))
    
    def reset(self):
        for (a,b) in self.managed:
            a.memory=b()


class convolution(nn.Module):
  def __init__(self,sampler,recognizer,reshape):
    super().__init__()
    self.sampler=sampler
    self.recognizer=recognizer
    self.reshape=reshape

  def forward(self,x):
    samples=self.sampler(x)
    y=[self.recognizer(s) for s in samples]
    return self.reshape(x,y)