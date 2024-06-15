import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import math

class Relation(nn.Module):
    def __init__(self, in_features, ablation):
        super(Relation, self).__init__()
        
        self.gamma_1 = nn.Linear(in_features, in_features, bias=False)
        self.gamma_2 = nn.Linear(in_features, in_features, bias=False)

        self.beta_1 = nn.Linear(in_features, in_features, bias=False)
        self.beta_2 = nn.Linear(in_features, in_features, bias=False)

        self.r = Parameter(torch.FloatTensor(1, in_features))

        self.elu = nn.ELU()
        self.lrelu = nn.LeakyReLU(0.2)

        self.sigmoid = nn.Sigmoid()
        self.reset_parameter()
        self.ablation = ablation
        

    def reset_parameter(self):
        stdv = 1. / math.sqrt(self.r.size(1))
        self.r.data.uniform_(-stdv, stdv)
        

    def forward(self, ft, neighbor):

        if self.ablation == 3:
            self.m = ft + self.r - neighbor

        else:
            
            self.r_v  = self.gamma_1(ft) + self.gamma_2(neighbor)

            self.m = self.r_v
            # end
            
        return self.m
    

