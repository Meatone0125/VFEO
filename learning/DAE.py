
from __future__ import division
from __future__ import print_function
from builtins import range

import warnings
warnings.filterwarnings('ignore')
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import random
import numpy as np
import json
import os
import sys
import argparse
import ast
from tqdm import tqdm
import logging
import torch.nn.functional as nnf
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
import torchnet as tnt

DIR_PATH = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(DIR_PATH, '..'))
from evaluation import eva
from learning import spg
from learning import pointnet
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from torch.optim import Adam
device = 0

class DAEGC(nn.Module):
    def __init__(self, num_features, hidden_size, embedding_size, alpha, num_clusters, v=1):
        super(DAEGC, self).__init__()
        self.num_clusters = num_clusters
        self.v = v
        self.embedding_size = embedding_size
        self.gat = GAT(num_features, hidden_size, embedding_size, alpha)

        self.cluster_layer = Parameter(torch.Tensor(num_clusters, embedding_size))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)
    def set_num_clusters(self,num_clusters):
        self.num_clusters = num_clusters
        self.cluster_layer = Parameter(torch.Tensor(num_clusters, self.embedding_size))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)

    def forward(self, x, adj, M):
        A_pred, z = self.gat(x, adj, M)
        q = self.get_Q(z)

        return A_pred, z, q

    def get_Q(self, z):
        eps=1e-6
        q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_layer.unsqueeze(0), 2), 2) / self.v)
        q = q.pow((self.v + 1.0) / 2.0)
        q = (q.t() / (torch.sum(q, 1)+eps)).t()
        return q

class GAT(nn.Module):
    def __init__(self, num_features, hidden_size, embedding_size, alpha):
        super(GAT, self).__init__()
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.alpha = alpha
        self.conv1 = GATLayer(num_features, hidden_size, alpha)
        self.conv2 = GATLayer(hidden_size, embedding_size, alpha)


    def forward(self, x, adj, M):
        h = self.conv1(x, adj, M)
        h = self.conv2(h, adj, M)
        z = nnf.normalize(h, p=2, dim=1)
        A_pred = self.dot_product_decode(z)
        return A_pred, z

    def dot_product_decode(self, Z):
        A_pred = torch.sigmoid(torch.matmul(Z, Z.t()))
        return A_pred

class GATLayer(nn.Module):

    def __init__(self, in_features, out_features, alpha=0.2):
        super(GATLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a_self = nn.Parameter(torch.zeros(size=(out_features, 1)))
        nn.init.xavier_uniform_(self.a_self.data, gain=1.414)
        self.a_neighs = nn.Parameter(torch.zeros(size=(out_features, 1)))
        nn.init.xavier_uniform_(self.a_neighs.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj, M, concat=True):
        h = torch.mm(input, self.W)
        attn_for_self = torch.mm(h, self.a_self)  # (N,1)
        attn_for_neighs = torch.mm(h, self.a_neighs)  # (N,1)
        attn_dense = attn_for_self + torch.transpose(attn_for_neighs, 0, 1)
        attn_dense = torch.mul(attn_dense, M)
        attn_dense = self.leakyrelu(attn_dense)  # (N,N)
        zero_vec = -9e15 * torch.ones_like(adj)
        adj = adj.float()
        adj = torch.where(adj > 0, attn_dense, zero_vec)
        attention = nnf.softmax(adj, dim=1)
        h_prime = torch.matmul(attention, h)

        if concat:
            return nnf.elu(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return (
            self.__class__.__name__
            + " ("
            + str(self.in_features)
            + " -> "
            + str(self.out_features)
            + ")"
        )
def train():
    dae.train()
    loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=spg.eccpc_collate,num_workers=args.nworkers, shuffle=False, drop_last=True)

    with torch.autograd.set_detect_anomaly(True):
        loss_meter = tnt.meter.AverageValueMeter()
        ari_meter = tnt.meter.AverageValueMeter()
        acc_meter = tnt.meter.ClassErrorMeter(accuracy=True)
        if logging.getLogger().getEffectiveLevel() > logging.DEBUG: loader = tqdm(loader, ncols=65)

        for bidx, (targets, GIs, clouds_data,edge_index) in enumerate(loader):
            data = ptnCloudEmbedder.run(model, *clouds_data) #x

            label_mode_cpu, label_vec_cpu, segm_size_cpu = targets[:, 0], targets[:, 2:], targets[:, 1:].sum(1)
            if args.cuda:
                label_mode, label_vec, segm_size = label_mode_cpu.cuda(), label_vec_cpu.float().cuda(), segm_size_cpu.float().cuda()
            else:
                label_mode, label_vec, segm_size = label_mode_cpu, label_vec_cpu.float(), segm_size_cpu.float()
            y = label_mode
            n = torch.unique(y)
            new_num_clusters = len(n)
            dae.set_num_clusters(new_num_clusters)

            adj = torch.sparse_coo_tensor(edge_index, torch.ones(edge_index.shape[1]), torch.Size([data.shape[0], data.shape[0]])).to_dense()  # [3840,3840]
            adj_label = adj
            adj += torch.eye(adj.shape[0])
            adj = adj / adj.sum(dim=1, keepdim=True)  # [3840,3840]
            M = get_M(adj).cuda()
            adj = adj.clone().detach().cuda()

            with torch.no_grad():
                _, z = dae.gat(data, adj, M)
            kmeans = KMeans(n_clusters= new_num_clusters, n_init=20)  # n_cluster caution
            kmeans.fit_predict(z.data.cpu().numpy())
            y=y.cpu().numpy()
            dae.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(device)

            A_pred, z, Q = dae(data, adj, M)
            q = Q.detach().data.cpu().numpy().argmax(1)
            ari = eva(y, q, epoch)
            p = target_distribution(Q.detach())

            A_pred = A_pred.cuda()
            adj_label = adj_label.cuda()
            # kl_loss = F.kl_div(Q.log(), p, reduction='batchmean')
            re_loss = F.binary_cross_entropy(A_pred.view(-1), adj_label.view(-1))
            # loss = 0.1 * kl_loss + re_loss
            loss = re_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_meter.add(loss.item())
            ari_meter.add(ari.item())
            o_cpu, t_cpu, tvec_cpu = filter_valid(z.data.cpu().numpy(), y,label_vec_cpu.numpy())
            if t_cpu.size > 0:
                acc_meter.add(o_cpu, t_cpu)

            if epoch % 1 == 0 or epoch == args.epochs - 1:
                torch.save(dae.state_dict(), os.path.join(args.daeodir, f'dae_model_{epoch}.pth'))

        return meter_value(acc_meter), loss_meter.value()[0], ari_meter.value()[0]

def create_model(args, dbinfo):
    """ Creates model """

    if not 'use_pyg' in args:
        args.use_pyg = 0

    model = nn.Module()

    model.ptn = pointnet.PointNet(args.ptn_widths[0], args.ptn_widths[1], args.ptn_widths_stn[0], args.ptn_widths_stn[1], dbinfo['node_feats'], args.ptn_nfeat_stn, prelast_do=args.ptn_prelast_do)
    dae = DAEGC(num_features=args.input_dim, hidden_size=args.hidden_size, embedding_size=args.embedding_size,alpha=args.alpha, num_clusters=args.n_clusters).to(device)

    if args.cuda:
        model.cuda()
        dae.cuda()
    return model,dae

def create_optimizer(args, model):
    if args.optim=='sgd':
        return optim.SGD(dae.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd)
    elif args.optim=='adam':
        return optim.Adam(dae.parameters(), lr=args.lr, weight_decay=args.wd)

def set_seed(seed, cuda=True):
    """ Sets seeds in all frameworks"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)

def filter_valid(output, target, other=None):
    """ Removes predictions for nodes without ground truth """
    idx = target != -100
    if other is not None:
        return output[idx, :], target[idx], other[idx, ...]
    return output[idx, :], target[idx]

def meter_value(meter):
    return meter.value()[0] if meter.n > 0 else 0


def get_M(adj):
    adj_numpy = adj.cpu().numpy()
    # t_order
    t = 2
    tran_prob = normalize(adj_numpy, norm="l1", axis=0)
    M_numpy = sum([np.linalg.matrix_power(tran_prob, i) for i in range(1, t + 1)]) / t
    return torch.Tensor(M_numpy)

def target_distribution(q):
    weight = q ** 2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Large-scale Point Cloud Semantic Segmentation with Superpoint Graphs')

    # Optimization arguments
    parser.add_argument('--wd', default=0, type=float, help='Weight decay')
    parser.add_argument('--lr', default=1e-2, type=float, help='Initial learning rate')
    parser.add_argument('--lr_decay', default=0, type=float,
                        help='Multiplicative factor used on learning rate at `lr_steps`')
    parser.add_argument('--lr_steps', default='[]',
                        help='List of epochs where the learning rate is decreased by `lr_decay`')
    parser.add_argument('--momentum', default=0, type=float, help='Momentum')
    parser.add_argument('--epochs', default=10, type=int,
                        help='Number of epochs to train. If <=0, only testing will be done.')
    parser.add_argument('--batch_size', default=1, type=int, help='Batch size')
    parser.add_argument('--optim', default='adam', help='Optimizer: sgd|adam')
    parser.add_argument('--grad_clip', default=1, type=float,
                        help='Element-wise clipping of gradient. If 0, does not clip')
    parser.add_argument('--loss_weights', default='none',
                        help='[none, proportional, sqrt] how to weight the loss function')

    # Learning process arguments
    parser.add_argument('--cuda', default=1, type=int, help='Bool, use cuda')
    parser.add_argument('--nworkers', default=0, type=int,
                        help='Num subprocesses to use for data loading. 0 means that the data will be loaded in the main process')
    parser.add_argument('--test_nth_epoch', default=1, type=int, help='Test each n-th epoch during training')
    parser.add_argument('--save_nth_epoch', default=1, type=int, help='Save model each n-th epoch during training')
    parser.add_argument('--test_multisamp_n', default=10, type=int,
                        help='Average logits obtained over runs with different seeds')

    # Dataset
    parser.add_argument('--dataset', default='sema3d', help='Dataset name: sema3d|s3dis')
    parser.add_argument('--cvfold', default=0, type=int,
                        help='Fold left-out for testing in leave-one-out setting (S3DIS)')
    parser.add_argument('--odir', default='results', help='Directory to store results')
    parser.add_argument('--daeodir', default='dae_pre/s3dis', help='Directory to store dae_pre results')
    parser.add_argument('--resume', default='', help='Loads a previously saved model.')
    parser.add_argument('--db_train_name', default='train')
    parser.add_argument('--db_test_name', default='test')
    parser.add_argument('--use_val_set', type=int, default=0)
    parser.add_argument('--SEMA3D_PATH', default='datasets/semantic3d')
    parser.add_argument('--S3DIS_PATH', default='datasets/s3dis')
    parser.add_argument('--VKITTI_PATH', default='datasets/vkitti')
    parser.add_argument('--CUSTOM_SET_PATH', default='datasets/custom_set')
    parser.add_argument('--use_pyg', default=0, type=int, help='Wether to use Pytorch Geometric for graph convolutions')

    # Model
    parser.add_argument('--model_config', default='gru_10,f_8',
                        help='Defines the model as a sequence of layers, see graphnet.py for definitions of respective layers and acceptable arguments. In short: rectype_repeats_mv_layernorm_ingate_concat, with rectype the type of recurrent unit [gru/crf/lstm], repeats the number of message passing iterations, mv (default True) the use of matrix-vector (mv) instead vector-vector (vv) edge filters, layernorm (default True) the use of layernorms in the recurrent units, ingate (default True) the use of input gating, concat (default True) the use of state concatenation')
    parser.add_argument('--seed', default=0, type=int, help='Seed for random initialisation')
    parser.add_argument('--edge_attribs', default='delta_avg,delta_std,nlength/ld,surface/ld,volume/ld,size/ld,xyz/d',
                        help='Edge attribute definition, see spg_edge_features() in spg.py for definitions.')

    # Point cloud processing
    parser.add_argument('--pc_attribs', default='xyzrgbelpsvXYZ',
                        help='Point attributes fed to PointNets, if empty then all possible. xyz = coordinates, rgb = color, e = elevation, lpsv = geometric feature, d = distance to center')
    parser.add_argument('--pc_augm_scale', default=0, type=float,
                        help='Training augmentation: Uniformly random scaling in [1/scale, scale]')
    parser.add_argument('--pc_augm_rot', default=1, type=int,
                        help='Training augmentation: Bool, random rotation around z-axis')
    parser.add_argument('--pc_augm_mirror_prob', default=0, type=float,
                        help='Training augmentation: Probability of mirroring about x or y axes')
    parser.add_argument('--pc_augm_jitter', default=1, type=int,
                        help='Training augmentation: Bool, Gaussian jittering of all attributes')
    parser.add_argument('--pc_xyznormalize', default=1, type=int,
                        help='Bool, normalize xyz into unit ball, i.e. in [-0.5,0.5]')

    # Filter generating network
    parser.add_argument('--fnet_widths', default='[32,128,64]',
                        help='List of width of hidden filter gen net layers (excluding the input and output ones, they are automatic)')
    parser.add_argument('--fnet_llbias', default=0, type=int, help='Bool, use bias in the last layer in filter gen net')
    parser.add_argument('--fnet_orthoinit', default=1, type=int,
                        help='Bool, use orthogonal weight initialization for filter gen net.')
    parser.add_argument('--fnet_bnidx', default=2, type=int,
                        help='Layer index to insert batchnorm to. -1=do not insert.')
    parser.add_argument('--edge_mem_limit', default=30000, type=int,
                        help='Number of edges to process in parallel during computation, a low number can reduce memory peaks.')

    # Superpoint graph
    parser.add_argument('--spg_attribs01', default=1, type=int,
                        help='Bool, normalize edge features to 0 mean 1 deviation')
    parser.add_argument('--spg_augm_nneigh', default=100, type=int, help='Number of neighborhoods to sample in SPG')
    parser.add_argument('--spg_augm_order', default=3, type=int, help='Order of neighborhoods to sample in SPG')
    parser.add_argument('--spg_augm_hardcutoff', default=512, type=int,
                        help='Maximum number of superpoints larger than args.ptn_minpts to sample in SPG')
    parser.add_argument('--spg_superedge_cutoff', default=-1, type=float,
                        help='Artificially constrained maximum length of superedge, -1=do not constrain')

    # Point net
    parser.add_argument('--ptn_minpts', default=40, type=int,
                        help='Minimum number of points in a superpoint for computing its embedding.')
    parser.add_argument('--ptn_npts', default=128, type=int, help='Number of input points for PointNet.')
    parser.add_argument('--ptn_widths', default='[[64,64,128,128,256], [256,64,32]]', help='PointNet widths')
    parser.add_argument('--ptn_widths_stn', default='[[64,64,128], [128,64]]', help='PointNet\'s Transformer widths')
    parser.add_argument('--ptn_nfeat_stn', default=11, type=int,
                        help='PointNet\'s Transformer number of input features')
    parser.add_argument('--ptn_prelast_do', default=0, type=float)
    parser.add_argument('--ptn_mem_monger', default=1, type=int,
                        help='Bool, save GPU memory by recomputing PointNets in back propagation.')

    # Decoder
    parser.add_argument('--sp_decoder_config', default="[]", type=str,
                        help='Size of the decoder : sp_embedding -> sp_class. First layer of size sp_embed (* (1+n_ecc_iteration) if concatenation) and last layer is n_classes')

    # pretrain
    parser.add_argument("--name", type=str, default="Citeseer")
    parser.add_argument("--max_epoch", type=int, default=1)
    parser.add_argument("--n_clusters", default=1, type=int)
    parser.add_argument("--input_dim", default=1, type=int)
    parser.add_argument("--hidden_size", default=1, type=int)
    parser.add_argument("--embedding_size", default=1, type=int)
    parser.add_argument("--weight_decay", type=int, default=5e-3)
    parser.add_argument("--alpha", type=float, default=1, help="Alpha for the leaky_relu.")
    parser.add_argument("--pre_bool", type=bool, default=1)

    args = parser.parse_args()
    args.start_epoch = 0
    args.lr_steps = ast.literal_eval(args.lr_steps)
    args.fnet_widths = ast.literal_eval(args.fnet_widths)
    args.ptn_widths = ast.literal_eval(args.ptn_widths)
    args.sp_decoder_config = ast.literal_eval(args.sp_decoder_config)
    args.ptn_widths_stn = ast.literal_eval(args.ptn_widths_stn)

    if not os.path.exists(args.daeodir):
        os.makedirs(args.daeodir)

    set_seed(args.seed, args.cuda)
    logging.getLogger().setLevel(logging.INFO)  # set to logging.DEBUG to allow for more prints
    if (args.dataset == 'sema3d' and args.db_test_name.startswith('test')) or (
            args.dataset.startswith('s3dis_02') and args.cvfold == 2):
        # needed in pytorch 0.2 for super-large graphs with batchnorm in fnet  (https://github.com/pytorch/pytorch/pull/2919)
        torch.backends.cudnn.enabled = False

    if args.use_pyg:
        torch.backends.cudnn.enabled = False

    # Decide on the dataset
    if args.dataset == 'sema3d':
        import sema3d_dataset

        dbinfo = sema3d_dataset.get_info(args)
        create_dataset = sema3d_dataset.get_datasets
    elif args.dataset == 's3dis':
        import s3dis_dataset

        dbinfo = s3dis_dataset.get_info(args)
        create_dataset = s3dis_dataset.get_datasets
    elif args.dataset == 'vkitti':
        import vkitti_dataset

        dbinfo = vkitti_dataset.get_info(args)
        create_dataset = vkitti_dataset.get_datasets
    elif args.dataset == 'custom_dataset':
        import custom_dataset  # <- to write!

        dbinfo = custom_dataset.get_info(args)
        create_dataset = custom_dataset.get_datasets
    else:
        raise NotImplementedError('Unknown dataset ' + args.dataset)


    model,dae = create_model(args, dbinfo)
    optimizer = create_optimizer(args, model)
    stats = []
    train_dataset, test_dataset, valid_dataset, scaler = create_dataset(args)

    print('Train dataset: %i elements - Test dataset: %i elements - Validation dataset: %i elements' % (len(train_dataset), len(test_dataset), len(valid_dataset)))
    ptnCloudEmbedder = pointnet.CloudEmbedder(args)
    scheduler = MultiStepLR(optimizer, milestones=args.lr_steps, gamma=args.lr_decay, last_epoch=args.start_epoch - 1)

    for epoch in range(args.epochs):
        print('Epoch {}/{} ({}):'.format(epoch, args.epochs, args.daeodir))
        acc, loss, ari = train()
        scheduler.step()

        stats.append({'epoch': epoch, 'acc': acc, 'loss': loss, 'ari': ari})
        if len(stats)>0:
            with open(os.path.join(args.daeodir, 'trainlog.json'), 'w') as outfile:
                json.dump(stats, outfile, indent=4)