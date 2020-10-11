from matching import *
from torch_geometric.nn import knn
from tools import *
import matplotlib.pyplot as plt
import math


class OTSmoothShells(SmoothShells):
    def __init__(self, param=DiffParam()):
        super().__init__(param)
        self.p = None
        self.p_adj = None

    def sinkhorn(self, d, sigma=None, num_sink=None):
        if sigma is None:
            sigma = self.param.sigma_sink
        if num_sink is None:
            num_sink = self.param.num_sink

        d = d / d.mean()

        log_p = -d / (2*sigma**2)

        for it in range(num_sink):
            log_p = log_p - torch.logsumexp(log_p, dim=1, keepdim=True)
            log_p = log_p - torch.logsumexp(log_p, dim=0, keepdim=True)
        log_p = log_p - torch.logsumexp(log_p, dim=1, keepdim=True)
        self.p = torch.exp(log_p)
        log_p = log_p - torch.logsumexp(log_p, dim=0, keepdim=True)
        self.p_adj = torch.exp(log_p).transpose(0, 1)

    def compute_mean_error(self, shape_x, shape_y):
        vert_x = self.vert_x_star[shape_x.samples, :]
        vert_y = self.vert_y_smooth[shape_y.samples, :]

        return ((vert_x - torch.mm(self.p, vert_y))**2).sum() + ((torch.mm(self.p_adj, vert_x) - vert_y) ** 2).sum()

    def compute_deformation(self, shape_x: Shape, shape_y: Shape):
        k = self.k

        spec_x = shape_x.evecs[shape_x.samples, :k]
        spec_y = shape_y.evecs[shape_y.samples, :k]

        spec_x = torch.cat((spec_x, torch.mm(self.p_adj, spec_x)), 0)
        spec_y = torch.cat((torch.mm(self.p, spec_y), spec_y), 0)
		
        vert_x = self.vert_x_smooth[shape_x.samples, :]
        vert_y = self.vert_y_smooth[shape_y.samples, :]
		
        res = torch.cat((torch.mm(self.p, vert_y) - vert_x, vert_y - torch.mm(self.p_adj, vert_x)), 0)

        self.compute_fm(spec_x, spec_y)
        self.compute_tau(spec_x, res)

    def feat_correspondences(self, shape_x, shape_y, emb_x, emb_y):
        d = dist_mat(emb_x, emb_y, False)
        self.sinkhorn(d)

    def compute_correspondences(self, shape_x: Shape, shape_y: Shape):
        emb_x, emb_y = self.product_embedding(shape_x, shape_y)

        self.feat_correspondences(shape_x, shape_y, emb_x, emb_y)

    def plot_correspondences(self, ax):
        ax.imshow(self.p[0::10, 0::10].detach().cpu().numpy())
        plt.title("k = " + str(self.k))

