import torch.nn as nn
import torch


def normalize(mat):
    norm=torch.norm(mat,dim=1).view(-1,1)
    return torch.div(mat,norm)

class SimCLRLoss(nn.Module):
    """SimCLR Loss

    We compare two vector $Z_1$ and $Z_2$

    We want :
    - Maximize Z_{1,i},Z_{2,i} agreement
    - Minimize Z_{1,i},Z_{2,j} agreement if j != i

    Implementation based on :
    https://medium.com/analytics-vidhya/understanding-simclr-a-simple-framework-for-contrastive-learning-of-visual-representations-d544a9003f3c

    Args:
        nn ([type]): [description]
    """
    def __init__(self,temperature=1):
        super(SimCLRLoss, self).__init__()
        self.temperature=temperature
    def forward(self, Z1, Z2):
        """We have :
        Z1_i, Z2_j = positive pair example (j=i)

        Args:
            Z1 ([type]): [description]
            Z2 ([type]): [description]
        """
        # Normalization...
        Zn1=normalize(Z1)
        Zn2=normalize(Z2)
        # Concatenation...(to compute either X1=>X2 and X2=>X1 comparision)
        Zn12=torch.cat([Zn1,Zn2],dim=0)
        Zn21=torch.cat([Zn2,Zn1],dim=0) # And not [Zn1,Zn2] !!!
        # We compute the product (cosine similarity) between every samples
        # We use only Zn12 and it's transpose.
        sim=torch.mm(Zn12,torch.t(Zn12))/self.temperature
        # We get sum sim where k !=i (we sum and substract diagonal values)
        denominator=sim.exp().sum(dim=1)-torch.diag(sim.exp())
        # We get the numerator, this is directly the cosine similarity Z1,Z2.
        # We use here Zn12 and Zn21, because we want to compare 1<=>2 and 2<=>1
        # IT IS NOT THE DIAGONAL !!! We want sim(Z1_i,Z2_i) not sim(Z1_i,Z1_i)
        numerator=torch.exp(nn.CosineSimilarity(dim=1)(Zn12,Zn21)/self.temperature)
        # We compute the loss
        loss=-torch.log(torch.div(numerator,denominator))
        return loss.mean()

if __name__=="__main__":
    Input=torch.tensor([[0.5,0.4],[0.2,0.1],[0.3,0.3]])
    Target=torch.tensor([[0.5,0.4],[0.2,0.1],[0.7,0.1]])
    res=SimCLRLoss()(Input,Target)