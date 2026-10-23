import torch

def layer_norm(x, gamma, beta, eps=1e-5):
    # TODO: normalize over the last dim, then affine-transform with gamma and beta
    mu = torch.mean(x,dim=-1,keepdim=True)
    var = torch.var(x,dim=-1,keepdim=True,unbiased=False)
    normalized = (x-mu)/(torch.sqrt(var)+eps)
    return gamma*normalized+beta