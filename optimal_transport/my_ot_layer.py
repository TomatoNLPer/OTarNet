import torch
from torch import nn
from my_OTK.kernel import OTKernel


class OTLayer(nn.Module):
    def __init__(self, in_dim, out_size, heads=1, eps=0.1, max_iter=10,
                 position_encoding=None, position_sigma=0.1, out_dim=None,
                 dropout=0.4):
        super().__init__()
        self.out_size = out_size
        self.heads = heads
        if out_dim is None:
            out_dim = in_dim
        self.kernel = OTKernel(in_dim, out_size, heads, eps, max_iter, log_domain=False,
                     position_encoding=position_encoding, position_sigma=position_sigma)
        self.layer = nn.Sequential(
            nn.Linear(heads * in_dim, out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
            )
        nn.init.xavier_uniform_(self.layer[0].weight)
        # nn.init.xavier_uniform_(self.layer[1].weight)

    def forward(self, input,target):
        output = self.kernel(input,target)
        output = self.layer(output)

        return output

# ot1 = OTLayer(in_dim = 6, out_size = 2, heads=1, eps=0.1, max_iter=10,
#                  position_encoding=None, position_sigma=0.1, out_dim=None,
#                  dropout=0.4)
# ot2 = OTLayer(in_dim = 6, out_size = 3, heads=1, eps=0.1, max_iter=10,
#                  position_encoding=None, position_sigma=0.1, out_dim=None,
#                  dropout=0.4)
# a = torch.tensor([[1,1,1,2,2,2],[2,2,2,3,3,3],[3,3,3,4,4,4]],dtype=torch.float).unsqueeze(0)
# b = torch.tensor([[1,2,3,4,5,6],[3,4,5,6,7,8]],dtype=torch.float).unsqueeze(0)
#
# c = ot1(a,b)
# d = ot2(b,a)
#
# x = 1
