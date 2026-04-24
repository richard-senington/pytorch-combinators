# SOME ATTEMPTS, NOT REALLY WORKING YET, ALL THE REUSE OF TENSORS ETC... ARE TRICKY


mask1=lift(lambda x:x*mask)

s = mask1 |seq| scale |seq| torch.exp()
t = mask1 |seq| translate 

op2=lift(lambda x:(1-mask)*x*(s(x)+t(x)),innerNetworks=[s,t])

fan_t(mask1,op2) |seq| apply_second(lift(lambda x:x[0]+x[1])) | seq| apply_first(inverseCalculation)




masked = lift(lambda x: x * mask)  # or nn.Identity() with mask baked in

st_net = masked |seq| probe(apply_second(lambda st: (torch.exp(st[0]), st[1]) )  , scale, translate)

coupling = skip(
    lift(lambda x, st: st[0] + (1-mask)*(x*st[1][0] + st[1][1])),
    st_net
)


coupling = lift(lambda x_ld, st: (
    x_ld[0]*mask + (1-mask)*(x_ld[0]*st[0] + st[1]),
    x_ld[1] + st[0].sum()
), ...)