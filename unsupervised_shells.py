import os
import torch
import torch.nn
from diff_matching import *
from shape_utils import *
from tools import *
from param import *
from data import *


class SpecConvModule(torch.nn.Module):
    def __init__(self, feat_channels, num_filters, num_basis_fcts=16, num_evecs=200):
        super().__init__()

        self.feat_channels = feat_channels
        self.num_filters = num_filters
        self.num_evecs = num_evecs
        self.num_basis_fcts = num_basis_fcts
        self.k_arr = my_range(0, num_basis_fcts).unsqueeze(0)  # [1, num_basis_fcts]

        param = Param()
        self.filter_scale = param.filter_scale

        self.a = torch.nn.Sequential(torch.nn.Linear(num_basis_fcts, feat_channels * num_filters))  # [num_basis_fcts, feat_channels x num_filters]

    def forward(self, emb, shape_x: Shape):
        emb_spec = torch.mm(shape_x.evecs[:, :self.num_evecs].transpose(0, 1),
                            emb * shape_x.A.unsqueeze(1)).unsqueeze(2)  # [num_evecs, feat_channels, 1]

        lambd = shape_x.evals[:self.num_evecs].unsqueeze(1)  # [num_evecs, 1]
        L = self.get_filter(lambd)

        emb_spec = emb_spec * L  # [num_evecs, feat_channels, num_filters]
        emb_spec = emb_spec.sum(dim=1)  # [num_evecs, num_filters]

        emb = torch.mm(shape_x.evecs[:, :self.num_evecs], emb_spec)  # [n, num_filters]
        return emb

    def get_filter(self, lambd):
        B = torch.cos(self.k_arr * lambd * math.pi / self.filter_scale)  # [num_evecs, num_basis_fcts]
        L = self.a(B)  # [num_evecs, feat_channels x num_filters]
        L = L.view((self.num_evecs, self.feat_channels, self.num_filters))  # [num_evecs, feat_channels, num_filters]
        return L


class FeatModuleSpecconv(torch.nn.Module):
    def __init__(self, dim_f, feat_channels=32):
        super().__init__()

        self.conv = SpecConvModule(dim_f, feat_channels)

    def forward(self, shape_x: Shape):
        emb = self.conv(shape_x.SHOT, shape_x)  # [n, num_filters]

        return emb[shape_x.samples, :]

    def get_all_features(self, shape_x: Shape, shape_y: Shape):
        emb_x_0 = shape_x.SHOT
        emb_x_1 = self.conv(emb_x_0, shape_x)  # [n, num_filters]
        L_x = self.conv.get_filter(shape_x.evals[:self.conv.num_evecs].unsqueeze(1))

        emb_y_0 = shape_y.SHOT
        emb_y_1 = self.conv(emb_y_0, shape_y)  # [n, num_filters]
        L_y = self.conv.get_filter(shape_y.evals[:self.conv.num_evecs].unsqueeze(1))

        ans = {'X': [emb_x_0.detach().cpu().numpy(), emb_x_1.detach().cpu().numpy()],
               'Y': [emb_y_0.detach().cpu().numpy(), emb_y_1.detach().cpu().numpy()],
               'L_x': L_x.detach().cpu().numpy(),
               'L_y': L_y.detach().cpu().numpy()
               }

        return ans


class UnsupervisedShells:
    def __init__(self, dataset=None, save_path=None, param=None):
        self.save_path = save_path
        self.dataset = dataset
        if dataset is not None:
            self.train_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

        if param is None:
            self.param = DeepParam()
        else:
            self.param = param

        self.feat_mod = FeatModuleSpecconv(352, self.param.num_hidden_layers).to(device)

        self.shells = OTSmoothShells(self.param)
        self.i_epoch = 0

    def save_self(self):
        folder_path = self.save_path

        if not os.path.isdir(folder_path):
            os.mkdir(folder_path)

        ckpt = {'i_epoch': self.i_epoch,
                'feat_mod': self.feat_mod.state_dict(),
                'par': self.param.__dict__}

        ckpt_name = 'ckpt_ep{}.pth'.format(self.i_epoch)
        ckpt_path = os.path.join(folder_path, ckpt_name)

        ckpt_last_name = 'ckpt_last.pth'
        ckpt_last_path = os.path.join(folder_path, ckpt_last_name)

        torch.save(ckpt, ckpt_path)
        torch.save(ckpt, ckpt_last_path)

    def load_self(self, folder_path, num_epoch=None):
        if num_epoch is None:
            ckpt_name = 'ckpt_last.pth'
            ckpt_path = os.path.join(folder_path, ckpt_name)
        else:
            ckpt_name = 'ckpt_ep{}.pth'.format(num_epoch)
            ckpt_path = os.path.join(folder_path, ckpt_name)
        ckpt = torch.load(ckpt_path, map_location=device)

        self.i_epoch = ckpt['i_epoch']
        self.feat_mod.load_state_dict(ckpt['feat_mod'])

        if 'par' in ckpt:
            self.param.from_dict(ckpt['par'])
            self.param.print_self()

        self.feat_mod.train()

        if num_epoch is None:
            print("Loaded model from ", folder_path, " with the latest weights")
        else:
            print("Loaded model from ", folder_path, " with the weights from epoch ", num_epoch)

    def train(self, num_epochs=int(1e5)):
        self.param.print_self()
        print("start training ...")

        optimizer = torch.optim.Adam(self.feat_mod.parameters(), lr=self.param.lr)
        self.feat_mod.train()

        while self.i_epoch < num_epochs:
            tot_loss = 0
            i_tot = 0
            for i, data in enumerate(self.train_loader):
                i_tot += 1
                shape_x = batch_to_shape(data["X"])
                shape_y = batch_to_shape(data["Y"])

                if self.param.subsample_num is not None:
                    shape_x.subsample_random(self.param.subsample_num)
                    shape_y.subsample_random(self.param.subsample_num)

                loss = self.match_pair(shape_x, shape_y)

                loss.backward()

                if i_tot % self.param.batch_size == 0:
                    optimizer.step()
                    optimizer.zero_grad()

                tot_loss += loss.detach() / self.dataset.__len__()

            optimizer.step()
            optimizer.zero_grad()

            print("epoch {:d}".format(self.i_epoch), ", loss = {:f}".format(tot_loss))
            if self.i_epoch % self.param.log_freq == 0 and self.save_path is not None:
                self.save_self()

            self.i_epoch += 1

    def feat_embedding(self, shape_x, shape_y):
        emb_x = self.feat_mod(shape_x)
        emb_y = self.feat_mod(shape_y)
        return emb_x, emb_y

    def feat_corr_pair(self, shape_x, shape_y):
        emb_x, emb_y = self.feat_embedding(shape_x, shape_y)

        self.shells.feat_correspondences(shape_x, shape_y, emb_x, emb_y)

    def match_pair(self, shape_x, shape_y):
        # compute learned correspondences
        self.feat_corr_pair(shape_x, shape_y)

        # match pair
        loss = self.shells.hierarchical_matching(shape_x, shape_y)

        # plot matched pair
        if self.i_epoch % self.param.log_freq == 0 and self.save_path is None:
            plot_shape_pair(shape_x, shape_y, self.shells.vert_x_star, self.shells.vert_y_smooth,
                            tit="epoch #" + str(self.i_epoch))

        return loss

    def test_model(self, shape_x, shape_y, plot_result=False, max_vert=6000):
        nx = shape_x.vert.shape[0]
        ny = shape_y.vert.shape[0]

        if nx > max_vert or ny > max_vert:
            n = min(nx - 1, ny - 1, max_vert)
            shape_x.subsample_fps(n)
            shape_y.subsample_fps(n)

        # compute feature correspondences
        self.feat_corr_pair(shape_x, shape_y)

        # compute matchings (with the differentiable pipeline)
        self.shells.param.status_log = False
        self.shells.param.k_array_len = 1
        self.shells.param.k_max = self.shells.param.k_min
        self.shells.param.compute_karray()
        self.shells.p = self.shells.p.detach()
        self.shells.p_adj = self.shells.p_adj.detach()
        self.shells.hierarchical_matching(shape_x, shape_y)

        # now set to the full resolution
        shape_x.reset_sampling()
        shape_y.reset_sampling()

        # fine alignment
        ass_shells = AssSmoothShells()
        ass_shells.param.status_log = False

        # transfer the correspondences from the surrogate and execute the full run
        emb_x, emb_y = self.shells.product_embedding(shape_x, shape_y)
        ass_shells.embedding_correspondences(shape_x, shape_y, emb_x, emb_y)
        ass_shells.hierarchical_matching(shape_x, shape_y)

        # (optional) plot final overlap
        if plot_result:
            plot_shape_triplet(shape_x, shape_y, ass_shells.vert_x_star)

        # extract and return final correspondences
        assignment = ass_shells.samples_y[:shape_x.vert.shape[0]]
        assignmentinv = ass_shells.samples_x[shape_x.vert.shape[0]:]

        return assignment, assignmentinv
