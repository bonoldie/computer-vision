import numpy as np
from scipy.optimize import linear_sum_assignment


def skew(v):
    """Return the skew-symmetric matrix of a vector v (3,)."""
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])

def vee(mat):
    """Inverse of skew (from 3x3 skew to 3x1 vector)."""
    return np.array([mat[2,1], mat[0,2], mat[1,0]])

def log_so3(R):
    """Compute logarithm of a rotation matrix R."""
    cos_theta = (np.trace(R) - 1) / 2.0
    cos_theta = np.clip(cos_theta, -1.0, 1.0)  # numerical stability
    theta = np.arccos(cos_theta)
    
    if np.isclose(theta, 0):
        return np.zeros(3)
    else:
        return vee(theta / (2*np.sin(theta)) * (R - R.T))

def log_se3(T):
    """Compute logarithm map of SE(3) transformation T.
       Returns 6-vector xi = [omega, v] (rotation, translation)."""
    R = T[:3, :3]
    t = T[:3, 3]
    
    omega = log_so3(R)
    theta = np.linalg.norm(omega)
    
    if np.isclose(theta, 0):
        V_inv = np.eye(3)  # when rotation is very small
    else:
        omega_hat = skew(omega / theta)
        V_inv = (np.eye(3) - 0.5 * omega_hat +
                 (1/theta - 0.5/np.tan(theta/2)) * omega_hat @ omega_hat)
    
    v = V_inv @ t
    xi = np.concatenate([omega, v])
    return xi

def chamfer_distance(A, B, squared=True):
    """
    A: (Na,2), B: (Nb,2)
    Returns symmetric Chamfer distance.
    If squared=True, uses squared Euclidean (common in vision); else uses Euclidean.
    """
    # pairwise squared distances
    d2 = np.sum((A[:,None,:] - B[None,:,:])**2, axis=2)  # (Na,Nb)
    # NN distances both directions
    a2b = d2.min(axis=1)  # (Na,)
    b2a = d2.min(axis=0)  # (Nb,)
    if squared:
        return a2b.mean() + b2a.mean()
    else:
        return np.sqrt(a2b).mean() + np.sqrt(b2a).mean()

def hausdorff_distance(A, B):
    d2 = np.sum((A[:,None,:] - B[None,:,:])**2, axis=2)
    a2b = np.sqrt(d2.min(axis=1)).max()
    b2a = np.sqrt(d2.min(axis=0)).max()
    return max(a2b, b2a)

def partial_assignment_rmse(A, B):
    """
    Match each point in the smaller set to a unique point in the larger set
    via Hungarian algorithm, then compute RMSE over those matches.
    Returns rmse, indicesA, indicesB for the matched pairs.
    """
    Na, Nb = len(A), len(B)
    # cost matrix (Euclidean)
    C = np.sqrt(((A[:,None,:] - B[None,:,:])**2).sum(axis=2))  # (Na,Nb)

    if Na <= Nb:
        row_ind, col_ind = linear_sum_assignment(C)  # size Na
        rmse = np.sqrt(np.mean(C[row_ind, col_ind]**2))
        return rmse, row_ind, col_ind
    else:
        # transpose to always match smaller->larger
        col_ind, row_ind = linear_sum_assignment(C.T)  # size Nb
        rmse = np.sqrt(np.mean(C[row_ind, col_ind]**2))
        return rmse, row_ind, col_ind
    
def symmetric_partial_rmse(A, B):
    rmse1, _, _ = partial_assignment_rmse(A, B)
    rmse2, _, _ = partial_assignment_rmse(B, A)
    return 0.5 * (rmse1 + rmse2)