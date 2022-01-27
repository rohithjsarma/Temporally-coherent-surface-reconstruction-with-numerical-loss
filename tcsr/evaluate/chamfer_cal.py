import torch


def _chamfer_distance(pc_pred, pc_gt):
        """ Loss functions computing Chamfer distance.

        Args:
            pc_gt (torch.Tensor): GT pcloud, shape (B, N, 3).
            pc_pred (torch.Tensor): Predicted pcloud, shape (B, M, 3).

        Returns:
            torch.Tensor: Scalar loss.
        """
        # Get registrations, get loss.
        inds_p2gt, inds_gt2p = _register_pts(pc_gt, pc_pred) # indices to the closest point 
        return _cd(pc_gt, pc_pred, inds_p2gt, inds_gt2p)

def _register_pts(pc_gt, pc_p):
        """

        Args:
            pc_gt:
            pc_p:

        Returns:

        """
        distm = _distance_matrix(pc_gt, pc_p)  # (B, M, N)
        inds_p2gt = distm.argmin(dim=2)  # (B, M)
        inds_gt2p = distm.argmin(dim=1)  # (B, N)
        return inds_p2gt, inds_gt2p

def _cd(pc_gt, pc_p, inds_p2gt, inds_gt2p):
        """ Extended Chamfer distance.

        Args:
            pc_gt: (B, N, 3)
            pc_p: (B, M, 3)
            inds_p2gt: (B, M)
            inds_gt2p: (B, N)

        Returns:

        """
        # Reshape inds.
        inds_p2gt = inds_p2gt.unsqueeze(2).expand(-1, -1, 3)
        inds_gt2p = inds_gt2p.unsqueeze(2).expand(-1, -1, 3)

        # Get registered points.
        pc_gt_reg = pc_gt.gather(1, inds_p2gt)  # (B, M, 3)
        pc_p_reg = pc_p.gather(1, inds_gt2p)  # (B, N, 3)

        # Compute per-point-pair differences.
        d_p2gt = torch.pow((pc_p - pc_gt_reg), 2).sum(dim=2)  # (B, M)
        d_gt2p = torch.pow((pc_gt - pc_p_reg), 2).sum(dim=2)  # (B, N)

        # Compute scalar loss.
        return d_p2gt.mean() + d_gt2p.mean()

def _distance_matrix(pc_N, pc_M):
        """ Computes a distance matrix between two pclouds.

        Args:
            pc_N (torch.Tensor): GT pcloud, shape (B, N, 3)
            pc_M (torch.Tensor): Predicted pcloud, shape (B, M, 3)

        Returns:
            Distance matrix, shape (B, M, N).
        """
        B, M, D = pc_M.shape
        B2, N, D2 = pc_N.shape
        assert B == B2 and D == D2 and D == 3

        x = pc_M.reshape((B, M, 1, D))
        y = pc_N.reshape((B, 1, N, D))
        print(x.shape, y.shape)
        return (x - y).pow(2).sum(dim=3).sqrt()  # (B, M, N, 3) -> (B, M, N)