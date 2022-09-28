import torch
import torch.nn as nn
import torch.nn.functional as F
from models.treeGAN.gcn import TreeGCN

class Generator(nn.Module):
    def __init__(self,features,degrees,support,args=None):
        self.layer_num = len(features)-1
        assert self.layer_num == len(degrees), "Number of features should be one more than number of degrees."
        self.pointcloud = None
        super(Generator, self).__init__()
        
        vertex_num = 1
        self.gcn = nn.Sequential()
        for inx in range(self.layer_num):
            # NOTE last layer activation False
            if inx == self.layer_num-1:
                self.gcn.add_module('TreeGCN_'+str(inx),
                                    TreeGCN(inx, features, degrees, 
                                            support=support, node=vertex_num, upsample=True, activation=False,args=args))
            else:
                self.gcn.add_module('TreeGCN_'+str(inx),
                                    TreeGCN(inx, features, degrees, 
                                            support=support, node=vertex_num, upsample=True, activation=True,args=args))
            vertex_num = int(vertex_num * degrees[inx])

    def forward(self, tree):
        feat = self.gcn(tree)
        
        self.pointcloud = feat[-1]
        
        return self.pointcloud

    def getPointcloud(self):
        return self.pointcloud[-1] # return a single point cloud (2048,3)

    def get_params(self,index):
        
        if index < 7:
            for param in self.gcn[index].parameters():
                yield param
        else:
            raise ValueError('Index out of range')
