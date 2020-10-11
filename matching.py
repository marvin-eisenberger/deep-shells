from shape_utils import *
from param import *
from torch_geometric.nn import knn
from tools import *
import matplotlib.pyplot as plt
import math

class SmoothShells:
    def __init__(self, param=Param()):
        self.k = None

        self.fm = None
        self.tau = None

        self.vert_x_smooth = None
        self.vert_y_smooth = None
        self.vert_x_star = None

        self.param = param

    def compute_mean_error(self, shape_x, shape_y):
        pass

    def compute_deformation(self, shape_x: Shape, shape_y: Shape):
        pass

    def feat_correspondences(self, shape_x, shape_y, emb_x, emb_y):
        pass

    def compute_correspondences(self, shape_x: Shape, shape_y: Shape):
        pass

    def update_vert_x_star(self, shape_x):
        self.vert_x_star = self.vert_x_smooth + torch.mm(shape_x.evecs[:, :self.k], self.tau)

    def product_embedding(self, shape_x: Shape, shape_y: Shape):
        k = self.k

        self.update_vert_x_star(shape_x)
        spec_x_star = torch.mm(shape_x.evecs[shape_x.samples, :k], self.fm)
        norm_x_star = compute_outer_normal(self.vert_x_star, shape_x.triv, shape_x.samples)

        vert_y = self.vert_y_smooth
        vert_y_norm = vert_y.norm()
        norm_y = compute_outer_normal(vert_y, shape_y.triv, shape_y.samples)

        fac_spec = self.param.fac_spec * vert_y_norm / shape_y.evecs[:, :k].norm()
        fac_norm = self.param.fac_norm * vert_y_norm / norm_y.norm()

        emb_x = torch.cat((self.vert_x_star[shape_x.samples, :], fac_spec * spec_x_star, fac_norm * norm_x_star), 1)
        emb_y = torch.cat((vert_y[shape_y.samples, :], fac_spec * shape_y.evecs[shape_y.samples, :k], fac_norm * norm_y), 1)

        return emb_x, emb_y

    def hierarchical_matching(self, shape_x: Shape, shape_y: Shape):
        tot_error = 0

        for k in self.param.k_array:
            self.k = k
            self.smooth_shape(shape_x, shape_y)

            self.compute_deformation(shape_x, shape_y)
            self.compute_correspondences(shape_x, shape_y)

            curr_error = self.compute_mean_error(shape_x, shape_y)
            tot_error += curr_error / self.param.k_array_len

            if self.param.status_log:
                self.print_status(shape_x, shape_y, curr_error)
        return tot_error

    def print_status(self, shape_x: Shape, shape_y: Shape, curr_error):
        k = self.k

        print("k: \t", k, ", mean error: \t", '{:f}'.format(curr_error))

        self.plot_smooth_shells_iteration(shape_x, shape_y, "err: {:f}".format(curr_error))

    def smooth_shape(self, shape_x: Shape, shape_y: Shape):
        k_x = self.k
        self.vert_x_smooth = torch.mm(shape_x.evecs[:, :k_x], shape_x.xi[:k_x, :])

        weights, k_y = self._smooth_shell_weights(k_x)
        self.vert_y_smooth = torch.mm(shape_y.evecs[:, :k_y], weights * shape_y.xi[:k_y, :])

    def _smooth_shell_weights(self, k, smooth_thresh=10, trunc_thresh=1e-2):
        if k == 1:
            weights = [1]
        elif k >= smooth_thresh:
            t = -1 / (smooth_thresh - 1) * math.log(1 / (1 - trunc_thresh) - 1)
            weights = list(range(-(k - 1), smooth_thresh - 1))
            weights = [1. / (1 + math.exp(t * w)) for w in weights]
        else:
            t = -1 / (k - 1) * math.log(1 / (1 - trunc_thresh) - 1)
            weights = list(range(-(k - 1), k-1))
            weights = [1 / (1 + math.exp(t * w)) for w in weights]

        weights = my_tensor(weights).unsqueeze(1)
        k = weights.shape[0]

        return weights, k

    def shot_correspondences(self, shape_x, shape_y):
        emb_x = shape_x.SHOT[shape_x.samples, :]
        emb_y = shape_y.SHOT[shape_y.samples, :]

        self.feat_correspondences(shape_x, shape_y, emb_x, emb_y)

    def compute_fm(self, spec_x, spec_y):
        self.fm, _ = torch.solve(torch.mm(spec_x.transpose(0, 1), spec_y), torch.mm(spec_x.transpose(0, 1), spec_x))

        if self.param.area_preservation:
            u, _, v = torch.svd(self.fm)
            self.fm = torch.mm(u, v.transpose(0, 1))

    def compute_tau(self, spec_x, res):
        self.tau, _ = torch.solve(torch.mm(spec_x.transpose(0, 1), res), torch.mm(spec_x.transpose(0, 1), spec_x))

    def plot_correspondences(self, ax):
        pass

    def plot_smooth_shells_iteration(self, shape_x, shape_y, title_curr=None):
        fig = plt.figure()

        ax = fig.add_subplot(221)
        ax.imshow(self.fm.detach().cpu().numpy())
        plt.title("k = " + str(self.k))

        ax = fig.add_subplot(222)
        self.plot_correspondences(ax)
        if title_curr is not None:
            plt.title(str(title_curr))

        ax = fig.add_subplot(223, projection='3d')
        ax.plot_trisurf(self.vert_x_star[:, 0].detach().cpu().numpy(),
                        self.vert_x_star[:, 1].detach().cpu().numpy(),
                        self.vert_x_star[:, 2].detach().cpu().numpy(),
                        triangles=shape_x.get_triv_np(), cmap='viridis', linewidths=0.2)

        plt.title("X*")

        ax = fig.add_subplot(224, projection='3d')
        ax.plot_trisurf(self.vert_y_smooth[:, 0].detach().cpu().numpy(),
                        self.vert_y_smooth[:, 1].detach().cpu().numpy(),
                        self.vert_y_smooth[:, 2].detach().cpu().numpy(),
                        triangles=shape_y.get_triv_np(), cmap='viridis', linewidths=0.2)
        plt.title("Y")

        plt.show()


class AssSmoothShells(SmoothShells):
    def __init__(self, param=Param()):
        super().__init__(param)
        self.samples_x = None
        self.samples_y = None

    def compute_mean_error(self, shape_x, shape_y):
        return (self.vert_x_star[self.samples_x, :] - self.vert_y_smooth[self.samples_y, :]).abs().mean()

    def compute_deformation(self, shape_x: Shape, shape_y: Shape):
        k = self.k

        samples_x = self.samples_x
        samples_y = self.samples_y

        spec_x = shape_x.evecs[:, :k]
        spec_y = shape_y.evecs[:, :k]
        spec_x = spec_x[samples_x, :]
        spec_y = spec_y[samples_y, :]

        res = self.vert_y_smooth[samples_y, :] - self.vert_x_smooth[samples_x, :]

        self.compute_tau(spec_x, res)
        self.compute_fm(spec_x, spec_y)

    def feat_correspondences(self, shape_x, shape_y, emb_x, emb_y):
        ass_x = nn_search(emb_y, emb_x)
        ass_y = nn_search(emb_x, emb_y)

        self.samples_x = torch.cat((shape_x.samples, shape_x.samples[ass_y]), 0)
        self.samples_y = torch.cat((shape_y.samples[ass_x], shape_y.samples), 0)

    def compute_correspondences(self, shape_x: Shape, shape_y: Shape):
        emb_x, emb_y = self.product_embedding(shape_x, shape_y)
        self.embedding_correspondences(shape_x, shape_y, emb_x, emb_y)
    
    def embedding_correspondences(self, shape_x: Shape, shape_y: Shape, emb_x, emb_y):
        emb_x = emb_x.to(device_cpu)
        emb_y = emb_y.to(device_cpu)

        ass_x = knn(emb_y, emb_x, k=1).to(device)
        ass_y = knn(emb_x, emb_y, k=1).to(device)

        ass_y = torch.index_select(ass_y, 0, my_long_tensor([1, 0]))

        ass_x = ass_x.transpose(0, 1)
        ass_y = ass_y.transpose(0, 1)

        self.ass_to_samples(ass_x, ass_y, shape_x, shape_y)

    def ass_to_samples(self, ass_x, ass_y, shape_x, shape_y):
        samples = torch.cat((ass_x, ass_y), 0)

        self.samples_x = samples[:, 0]
        self.samples_y = samples[:, 1]

        self.samples_x = shape_x.samples[self.samples_x]
        self.samples_y = shape_y.samples[self.samples_y]

    def plot_correspondences(self, ax):
        ax.plot(self.samples_x.detach().cpu().numpy())
        ax.plot(self.samples_y.detach().cpu().numpy())
