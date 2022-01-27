""" Differential geometry properties. Implements the computation of eigen values of 3D points

Author: Rohith Jayakumara Sarma, rohith.jayakumarasarma@epfl.ch
"""

# 3rd party
import torch
import math
import torch.nn as nn
import torch.nn.functional as F

# Project files.
from tcsr.models.common import Device


class DiffGeomPropsApprox(nn.Module, Device):
    """ Computes the differential geometry properties including normals,
    mean curvature, gaussian curvature, first fundamental form (fff). Works
    for 2 types of mappings, 2D -> 2D or 2D -> 3D. In the first case, only
    fff is available.

    Args:
        normals (bool): Whether to compute normals (2D -> 3D only).
        curv_mean (bool): Whether to compute mean curvature (2D -> 3D only).
        curv_gauss (bool): Whether to compute gauss. curvature (2D -> 3D only).
        fff (bool): Whether to compute first fundamental form.
        gpu (bool): Whether to use GPU.
    """
    def __init__(self, neigh=16, fff=True, gpu=True):
        nn.Module.__init__(self)
        Device.__init__(self, gpu=gpu)

        self._neigh = neigh
        self._comp_fff = fff

    def forward(self, X, uv):
        """ Computes first 3 eigen values.

        Args:
            X (torch.Tensor): 2D or 3D points in output space (B, M, 2 or 3).
            uv (torch.Tensor): 2D points, parameter space, shape (B, M, 2).

        Returns:
            dict: Depending on `normals`, `curv_mean`, `curv_gauss`, `fff`
                includes normals, mean curvature, gauss. curvature and first
                fundamental form as torch.Tensor.
        """

        # Return values.
        ret = {}

        if not (self._comp_fff):
            return ret

        # print(self._neigh)
        neigh_3d = self._knns_cal(self._neigh, X, uv) # B, N, K, 3
        cov_mat_3d = self._cov_mat(neigh_3d)

        eig_values = torch.linalg.eigvals(cov_mat_3d).real.sort(descending=True)[0]
        
        assert torch.isnan(eig_values).sum() == 0
        assert torch.isinf(eig_values).sum() == 0
        ret['fff'] = eig_values  # (B, M, 3)

        return ret


    # def _eig_solve(self, covariance_matrix):
    #     """ Computes computes eigen values from covariance matrix

    #     Args:
    #         covariance_matrix (torch.Tensor), shape (B, M, 3, 3)

    #     Returns:
    #         Eigen value (torch.Tensor), shape (B, M, 3).
    #     """
    #     a = -1
    #     b = covariance_matrix[:,:,0,0] + covariance_matrix[:,:,1,1] + covariance_matrix[:,:,2,2]
    #     c = -(covariance_matrix[:,:,0,0]*covariance_matrix[:,:,2,2] + covariance_matrix[:,:,0,0]*covariance_matrix[:,:,1,1] + covariance_matrix[:,:,1,1]*covariance_matrix[:,:,2,2] - covariance_matrix[:,:,1,2]**2 - covariance_matrix[:,:,0,1]**2 - covariance_matrix[:,:,0,2]**2)
    #     d = -covariance_matrix[:,:,0,0]*covariance_matrix[:,:,1,2]**2 -covariance_matrix[:,:,2,2]*covariance_matrix[:,:,0,1]**2 -covariance_matrix[:,:,1,1]*covariance_matrix[:,:,0,2]**2 + 2*covariance_matrix[:,:,1,2]*covariance_matrix[:,:,0,1]*covariance_matrix[:,:,0,2] + covariance_matrix[:,:,0,0]*covariance_matrix[:,:,1,1]*covariance_matrix[:,:,2,2]

    #     f_v = ((3.0*c/a)-((b**2.0)/(a**2.0)))/3.0                          # Helper Temporary Variable
    #     g_v = (((2.0*(b**3.0))/(a**3.0))-((9.0*b*c)/(a**2.0))+(27.0*d/a))/27.0                       # Helper Temporary Variable
    #     h_v = ((g_v**2.0)/4.0+(f_v ** 3.0)/27.0)                             # Helper Temporary Variable

    #     i_v = torch.clamp((((g_v ** 2.0) / 4.0) - h_v), min=0).sqrt()     # Helper Temporary Variable (is i_v is -ve)
    #     j_v = i_v ** (1 / 3.0)                      # Helper Temporary Variable
    #     k_v = torch.clamp((-(g_v / (2 * i_v + self.epsil))), min = -1, max= 1).acos()             # Helper Temporary Variable (is this >= 1)

    #     L_v = j_v * -1                              # Helper Temporary Variable
    #     M_v = (k_v / 3.0).cos()                     # Helper Temporary Variable
    #     N_v = 1.73205 * (k_v / 3.0).sin()      # Helper Temporary Variable 
    #     P_v = (b / (3.0 * a)) * -1                # Helper Temporary Variable

    #     x1 = 2 * j_v * (k_v / 3.0).cos() - (b / (3.0 * a))
    #     x2 = L_v * (M_v + N_v) + P_v
    #     x3 = L_v * (M_v - N_v) + P_v

    #     return torch.stack([x1, x2, x3], dim=2)          # Returning Real Roots as numpy array.
    
    def _cov_mat(self, neigh_3d):
        """ Computes covariance matrix from neighbourhood

        Args:
            points_3d (torch.Tensor), shape (B, M, K, 3)

        Returns:
            Covariance matrix (torch.Tensor), shape (B, M, 3, 3).
        """
        B,M,K,D = neigh_3d.shape
        points_mean = neigh_3d - neigh_3d.mean(dim=2, keepdim=True)
        return torch.bmm(points_mean.transpose(2, 3).reshape(B*M, D, K),points_mean.reshape(B*M, K, D)).reshape(B,M,D,D)
    
    def _knns_cal(self, k, X, uv):
        B, N, D = uv.shape

        # Distance matrix.
        dm = ((uv[:, None] - uv[:, :, None]) ** 2).sum(dim=3)  # (B, N, N)

        k_inds = torch.topk(dm, k, dim=2, largest=False, sorted=False)[1]  # (B, N, k)

        assert torch.isnan(X).sum() == 0
        assert torch.isinf(X).sum() == 0
        # Collect the points.
        knns = torch.gather(
            X[:, None].expand(-1, N, -1, -1), 2,
            k_inds[..., None].expand(-1, -1, -1, 3))  # (B, N, k, 3)

        return knns
