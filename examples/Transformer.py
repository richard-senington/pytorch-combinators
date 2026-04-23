import torch.nn as nn
from torch_combinators import residual, seq, lift, probe, repeat, TokenEmbedding, position_encoding, sinusoidal_positional_encoding, skip
import math

def transformerBlock(q,k,vproj,ffn,d_model):   # annoynig we need the d_model value for the normalization
  # Q and K must give same shaped outputs  

  dot = lift(lambda _,qk: qk[0] @ qk[1].transpose(-2, -1)  )     # Q @ K.transpose(-2, -1)
  scale = lift(lambda x:x/ math.sqrt(d_model))

  f =  probe(dot, q,k ) |seq| scale | seq| nn.Softmax(dim=-1)
   

  return nn.LayerNorm(d_model) |seq| probe( lift(lambda _,qs: qs[0] @ qs[1] ) ,f,vproj  )    |seq| ffn

def transformer(num_tokens,n_model,max_len,dim_k):
    start = TokenEmbedding(num_tokens,n_model) |seq| position_encoding(sinusoidal_positional_encoding(max_len))

    basicffn =  nn.Linear(n_model, dim_k) |seq| nn.GELU() |seq| nn.Linear(dim_k, n_model)
    q = nn.Linear(n_model , dim_k)  # explain each bit, for self
    k = nn.Linear(n_model,dim_k ) 
    v = nn.Linear(n_model,dim_k )   # in practice, seems hard to not just duplicate these

    # this ends with a lifted operation for taking last
    take_last =  lift(lambda x:x[-1])
    finalBlock = nn.LayerNorm(n_model) |seq| nn.Linear(n_model, num_tokens) |seq| take_last

    return start |seq| repeat(3,residual(transformerBlock(q,k,v,basicffn,n_model))) |seq| finalBlock

# computing Q - Transform from :: Tensor A B -> Tensor A C
# K is the same
# then [dot(A,B) for A from Q B from K]

# softmax and scaling

# matrix multiply






t=transformer(10,20,50,10)
print(t)
k=sum(p.numel() for p in t.parameters())
print(k)