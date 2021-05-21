import torch

class BatchNorm1D():
    def __init__(self):
        super().__init__()

    def forward(self,input:torch.tensor(),batch_size:torch.tensor():int,mean_batch:torch.tensor,var_batch:torch.tensor,gamma,beta):
        eps=1e-6
        self.input=input
        self.batch_size=batch_size
        self.mean_batch=mean_batch
        self.var_batch=var_batch
        self.gamma=gamma
        self.beta=beta
        self.eps=eps
        zb=1/batch_size*((input-mean_batch)/(var_batch+eps).sqrt())
        yb=gamma @ zb+beta
        return yb

    def backward(self,G):
        Gv=-0.5*(self.var_batch+self.eps).pow(-3/2)
        Gv=Gv.mul((G @ (self.input-self.mean_batch)).sum())
        Gm=-1/(self.var_batch+eps).sqrt()*G.sum()
        
        return G.mul(1/(self.var_batch+eps).sqrt())+ \
            2/self.batch_size *Gv.mul(self.input-self.mean_batch)+\
                1/self.batch_size*Gm
class ReLU(Module):
    
    def __init__(self):
        super(ReLU).__init__()



    def forward(self, input_):
    
        self.input_ = input_
        return input_.clamp(min=0)

    def backward(self, G):
        
        return G.mul((self.input_ > 0).float())