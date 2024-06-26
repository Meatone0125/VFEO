"""
    Dynamic Edge-Conditioned Filters in Convolutional Neural Networks on Graphs
    https://github.com/mys007/ecc
    https://arxiv.org/abs/1704.02901
    2017 Martin Simonovsky
"""
from __future__ import division
from __future__ import print_function
from builtins import range

import igraph
import torch
from collections import defaultdict
import numpy as np
    
class GraphConvInfo(object):          
    """ Holds information about the structure of graph(s) in a vectorized form useful to `GraphConvModule`. 
    
    We assume that the node feature tensor (given to `GraphConvModule` as input) is ordered by igraph vertex id, e.g. the fifth row corresponds to vertex with id=4. Batch processing is realized by concatenating all graphs into a large graph of disconnected components (and all node feature tensors into a large tensor).

    The class requires problem-specific `edge_feat_func` function, which receives dict of edge attributes and returns Tensor of edge features and LongTensor of inverse indices if edge compaction was performed (less unique edge features than edges so some may be reused).
    """

    def __init__(self, *args, **kwargs):
        self._idxn = None           #indices into input tensor of convolution (node features)
        self._idxe = None           #indices into edge features tensor (or None if it would be linear, i.e. no compaction)
        self._degrees = None        #in-degrees of output nodes (slices _idxn and _idxe)
        self._degrees_gpu = None
        self._edgefeats = None  #edge features tensor (to be processed by feature-generating network)
        self._adj_matrix = None
        if len(args)>0 or len(kwargs)>0:
            self.set_batch(*args, **kwargs)
      
    def set_batch(self, graphs, edge_feat_func):
        """ Creates a representation of a given batch of graphs.
        
        Parameters:
        graphs: single graph or a list/tuple of graphs.
        edge_feat_func: see class description.
        """
        
        graphs = graphs if isinstance(graphs,(list,tuple)) else [graphs]
        p = 0
        adj_matrix = []
        idxn = []
        degrees = []
        edge_indexes = []
        edgeattrs = defaultdict(list)
                
        for G in graphs:
            # edge = list(G.get_adjacency())
            # adj.append(np.array(edge))
            # a = np.array(G.get_adjacency())
            # adj.append(a)
            E = np.array(G.get_edgelist())
            idx = E[:,1].argsort() # sort by target
            
            idxn.append(p + E[idx,0])
            edgeseq = G.es[idx.tolist()]
            for a in G.es.attributes():
                edgeattrs[a] += edgeseq.get_attribute_values(a)
            degrees += G.indegree(G.vs, loops=True)
            edge_indexes.append(np.asarray(p + E[idx]))
            p += G.vcount()
            adj_matrix.append(G.get_adjacency().data)

        max_dim = max(len(mat) for mat in adj_matrix)
        padded_adj_matrix = []
        for mat in adj_matrix:
            pad_width = max_dim - len(mat)
            padded_mat = np.pad(mat,((0,pad_width),(0,pad_width)),mode='constant')
            padded_adj_matrix.append(padded_mat)

        self._edgefeats, self._idxe = edge_feat_func(edgeattrs)
        
        self._idxn = torch.LongTensor(np.concatenate(idxn))
        if self._idxe is not None:
            assert self._idxe.numel() == self._idxn.numel()
            
        self._degrees = torch.LongTensor(degrees)
        self._degrees_gpu = None

        self._edge_indexes = torch.LongTensor(np.concatenate(edge_indexes).T)
        self._adj_matrix = torch.from_numpy(np.concatenate(padded_adj_matrix)).float()
        # self.adj = torch.from_numpy(np.concatenate(adj)).long()
    def cuda(self):
        self._idxn = self._idxn.cuda()
        if self._idxe is not None: self._idxe = self._idxe.cuda()
        self._degrees_gpu = self._degrees.cuda()
        self._edgefeats = self._edgefeats.cuda()
        self._edge_indexes = self._edge_indexes.cuda()
        self._adj_matrix = self._adj_matrix.cuda()
        
    def get_buffers(self):
        """ Provides data to `GraphConvModule`.
        """
        return self._idxn, self._idxe, self._degrees, self._degrees_gpu, self._edgefeats

    def get_pyg_buffers(self):
        """ Provides data to `GraphConvModule`.
        """
        return self._edge_indexes

    # def get_adj(self):
    #     return self.adj