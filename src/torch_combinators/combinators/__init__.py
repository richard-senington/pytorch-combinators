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
  """An infix operator for sequencing two networks. It is a thin wrapper of nn.Sequential, and for longer sequences it is probably best to use that directly.
     Example:
        >>> network1 |seq| network2
  """ 
  return nn.Sequential(a,b)

class lift(nn.Module):
  def __init__(self,f,innerNetworks=[]):
    super().__init__()
    self.f=f
    self.inner=nn.ModuleList(innerNetworks)

  def forward(self,x):
    return self.f(x)



def repeat(n,f):
  """Replicates a given network (f) several times and sequences them. This makes use of Python's deep copy, and 
  the weights will be duplicated with, so reset/rerandomize the network before use. For this to work, it is important
  that the parameter (f), the network, can be chained together, so the input Tensor Shape/Type must match the output.
  
  Example:
        >>> repeat(5, nn.Linear(5,5)) 

  """
  x=copy.deepcopy(f)
  for _ in range(n-1):
    x=x |seq| copy.deepcopy(f)
  return x

def skip(r,f):
  """Applies the network f to an input, and then recombines (r) the output with the original input. For this to work, the network, or lifted
  operation (r) must be able to take 2 parameters as a tuple and recombine them. """
  r2=lift(lambda x:r((x[1][0],x[1][1])),
          innerNetworks=[r])
  return probe(r2,f,nn.Identity())

def residual(f):
  """A specialisation of skip, where the recombination is pointwise addition. This constrains what is allowed in the network (f). It must give back a tensor 
  of the same shape as the input.
  
  Example:
        >>> residual(  nn.Linear(5,5) |seq| nn.ReLU()  )

  """
  return skip(lift(lambda x:x[0]+x[1]),f)

class first(nn.Module):
  def __init__(self,f):
    super().__init__()
    self.f=f 

  def forward(self,x):
    return (self.f(x[0]),x[1])

def basic_gate(i_shape,o_shape,f,g):
  """A standard gate will take the input, apply a normal fully connected layer to it with the Sigmoid activation, and the the 0-1 ranged output
     to integrate the outputs of 2 other networks. This combinator provides this by creating an appropriate recombination network for the probe
     combinator."""
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



def duplicate(n,f):
  
  return fan_l([copy.deepcopy(f) for _ in range(n)])

class fan_t(nn.Module):
  def __init__(self,*args):
    super().__init__()
    self.fs=nn.ModuleList(args)
  def forward(self,x):
    return (x,tuple([f(x) for f in self.fs]))

def probe(r,f,g):
    return fan_t(f,g) |seq| r

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
    """Only to manage resets in recurrent networks. Intended for use during training rather than in normal runtime."""
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





class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
 
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(token_ids)

  # NOTE THIS COULD JUST BE, EMBEDDING()
  # LOGIC IS, internally a tensor of vocab_size vs size of embeddings, then loop over an integer tensor and rebuild.





def sinusoidal_positional_encoding(d_model: int, max_seq_len: int = 512):
  # Compute sinusoidal positional encodings.
  positions = torch.arange(max_seq_len).unsqueeze(1)        # (max_seq_len, 1)
  dims      = torch.arange(0, d_model, 2)                   # (d_model/2,)
  divisor   = torch.pow(10000, dims / d_model)              # (d_model/2,)
 
  PE = torch.zeros(max_seq_len, d_model)
  PE[:, 0::2] = torch.sin(positions / divisor)
  PE[:, 1::2] = torch.cos(positions / divisor)

  return lift(lambda _:PE)

def position_encoding(position_encoder):
  """For use in attention, we need a way to add position encoding. This is usually done as the addition of position information to the input, which is a form of residual"""
  return residual(position_encoder)


def apply_first(f):
  return lift(lambda x:(f(x[0]),x[1] ),innerNetworks=[f])

def apply_second(f):
  return lift(lambda x:(x[0],f(x[1]) ),innerNetworks=[f])

def reset_weights(model):
  """A little helper for use when I have made use of 'repeat', which uses deep copy.
  This will resent/randomize all weights throughout a model"""
  
  for layer in model.modules():
    if hasattr(layer, 'reset_parameters'):
      layer.reset_parameters()     
