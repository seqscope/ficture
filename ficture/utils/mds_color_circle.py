import torch
import pymde
import numpy as np
import matplotlib.colors
import matplotlib.pyplot as plt
import re, os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# A constraint class that implements the unit circle constraint
class circle(pymde.constraints.Constraint):
    def name(self):
        return "circle"

    def initialization(self, n_items, embedding_dim, device=None):
        assert embedding_dim == 2, "Embedding dimension must be 2"
        angles = torch.rand(n_items, device=device) * 2 * torch.tensor([3.141592653589793])
        x_coords = torch.cos(angles)
        y_coords = torch.sin(angles)
        return torch.stack((x_coords, y_coords), dim=1)

    def project_onto_tangent_space(self, X, Z, inplace=True):
        assert Z.shape == X.shape, "Z and X must have the same shape"
        if not torch.all(torch.isclose(torch.norm(X, dim=1), \
                         torch.tensor(1.0), atol=1e-6)):
            X = self.project_onto_constraint(X, inplace=True)
        assert torch.all(torch.isclose(torch.norm(X, dim=1), \
                         torch.tensor(1.0), atol=1e-6)), "X must lie on the unit circle"
        # Compute the dot product of Z and X
        dot_product = torch.sum(Z * X, dim=1, keepdim=True)
        # Subtract the result from Z to get the orthogonal component
        if inplace:
            Z.sub_(dot_product * X)
            return Z
        else:
            return Z - dot_product * X

    def project_onto_constraint(self, Z, inplace=True):
        if inplace:
            Z.div_(torch.norm(Z, dim=1).unsqueeze(-1))
            return Z
        else:
            return torch.div(Z, torch.norm(Z, dim=1).unsqueeze(-1))

# Map pairwise factor proximity to an unit circle embedding then to RGB colors
def assign_color_mds_circle(mtx, cmap_name, weight=None, top_color=None, seed=None):
    if seed is not None:
        pymde.seed(seed)
    # mtx is a K by K similarity/proximity matrix
    assert mtx.shape[0] == mtx.shape[1], "mtx must be square"
    K = mtx.shape[0]
    # weight is a K vector of factor abundance
    if weight is None:
        weight = np.ones(K)
    weight /= weight.sum()

    # The color of the top factor (the one with the largest weight)
    if top_color is None:
        top_color = "#fcd217"
    else:
        match = re.search(r'^#(?:[0-9a-fA-F]{3}){1,2}$', top_color)
        if match is None:
            top_color = "#fcd217"

    # Construct cost function (want to match large values in mtx to large embedding distances)
    c_dist = []
    c_edge = []
    for i in range(K-1):
        for j in range(i+1,K):
            c_dist.append(-mtx[i,j] * (weight[i]+weight[j]) )
            c_edge.append([i,j])
    f=pymde.penalties.Log(torch.tensor(c_dist, dtype=torch.float32) )

    mde = pymde.MDE(
        n_items=K,
        embedding_dim=2,
        edges=torch.tensor(c_edge, dtype=torch.int64),
        distortion_function=f,
        constraint=circle())
    mde_res = mde.embed().detach().numpy()

    # Recover angles from the embedding
    angle = np.arctan2(mde_res[:, 1], mde_res[:,0]) + np.pi # (0, 2pi)
    anchor_k = np.argmax(weight)
    # Find the offset to map the top factor to the desired color
    cgrid = 200
    cmtx=plt.get_cmap(cmap_name)(np.arange(cgrid)/cgrid)

    top_color_rgb = matplotlib.colors.to_rgb(top_color)
    d = np.abs(cmtx[:, :3] - np.array(top_color_rgb).reshape((1, -1)) ).sum(axis = 1)
    anchor_pos = d.argmin() / cgrid
    anchor_angle = anchor_pos * 2 * np.pi
    angle_shift = angle + (anchor_angle - angle[anchor_k])
    if angle_shift.max() > 2*np.pi:
        angle_shift -= np.pi * 2
    angle_shift[angle_shift < 0] = 2 * np.pi + angle_shift[angle_shift < 0]
    c_pos = angle_shift / np.pi / 2
    return c_pos
