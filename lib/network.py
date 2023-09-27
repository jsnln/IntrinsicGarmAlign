import numpy as np
import torch
import torch.nn as nn


class PosEnc(nn.Module):

    def __init__(self, inp_features, n_freq, cat_inp=True):
        super().__init__()
        self.inp_feat = inp_features
        self.n_freq = n_freq
        self.cat_inp = cat_inp
        self.out_dim = 2 * self.n_freq * self.inp_feat
        if self.cat_inp:
            self.out_dim += self.inp_feat

    def forward(self, x):
        """
        :param x: (bs, npoints, inp_features)
        :return: (bs, npoints, 2 * out_features + inp_features)
        """
        assert len(x.size()) == 3
        bs, npts = x.size(0), x.size(1)
        const = (2 ** torch.arange(self.n_freq) * np.pi).view(1, 1, 1, -1)
        const = const.to(x)

        # Out shape : (bs, npoints, out_feat)
        cos_feat = torch.cos(const * x.unsqueeze(-1)).view(
            bs, npts, self.inp_feat, -1)
        sin_feat = torch.sin(const * x.unsqueeze(-1)).view(
            bs, npts, self.inp_feat, -1)
        out = torch.cat(
            [sin_feat, cos_feat], dim=-1).view(
            bs, npts, 2 * self.inp_feat * self.n_freq)
        # const_norm = torch.cat(
        #     [const, const], dim=-1).view(
        #     1, 1, 1, self.n_freq * 2).expand(
        #     -1, -1, self.inp_feat, -1).reshape(
        #     1, 1, 2 * self.inp_feat * self.n_freq)

        if self.cat_inp:
            out = torch.cat([out, x], dim=-1)
            # const_norm = torch.cat(
            #     [const_norm, torch.ones(1, 1, self.inp_feat).to(x)], dim=-1)

            return out 
        else:

            return out

class DeformationField(nn.Module):
    def __init__(self,
                 dim=3,
                 out_dim=3,
                 hidden_size=128,
                 pos_enc_freq=0
                ):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.hidden_size = hidden_size

        first_layer = nn.Identity() if pos_enc_freq == 0 else PosEnc(self.dim, pos_enc_freq, True)
        self.nonlinear = nn.Sequential(
            first_layer,
            nn.Linear(self.dim * (2 * pos_enc_freq + 1), self.hidden_size, True),
            nn.ELU(),
            nn.Linear(self.hidden_size, self.hidden_size, True),
            nn.ELU(),
            nn.Linear(self.hidden_size, self.hidden_size, True),
            nn.ELU(),
            nn.Linear(self.hidden_size, self.hidden_size, True),
            nn.ELU(),
            nn.Linear(self.hidden_size, self.out_dim, True),
        )


    def forward(self, x):
        """
        :param x: (bs, npoints, self.dim) Input coordinate (xyz)

        :return: (bs, npoints, self.dim) Gradient (self.dim dimension)
        """
        return self.nonlinear(x)
    