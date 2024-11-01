import numpy as np


def time_delay_embed(ts, tau, emd, L=None):
    """ Time-delay embedding of a univariate time series.
    Args:
        ts   time series (univariate)
        tau  time delay
        emd  embedding dimension
        L    max length/time length, if None then full length
        
    Returns:
        An numpy array. Shape: [L-(emd-1)*tau, emd].

        

        Each row is one sample that represents a point (a certain time index t) on the embeddng;
        the elements of each sample are the values at time indices: 
            {t:[t, t-tau, t-2*tau ... t-(E-1)*tau]} = Shadow attractor manifold
        
        Array dimensions: [number of samples, embedding dimension].
        Number of samples equals L-(emd-1)*tau.
    """

    if L is None:
        L = len(ts)
    resultArr=np.zeros((L-(emd-1)*tau,emd))
    for t in range((emd - 1) * tau, L):
        for i in range(emd):
            resultArr[t - (emd - 1) * tau][i] = ts[t - i * tau]
    return resultArr



def partial_corr(x, y, cond):
    """ Partial correlation between x and y conditionned on cond.
    ref: https://github.com/PengTao-HUST/crossmapy/blob/master/crossmapy/utils.py

    Parameters
    ----------
    x: 2D array [num_samples, dim]
        First variable.
    y: 2D array [num_samples, dim]
        Second variable.
    cond: 2D array [num_samples, dim]
        Conditioning variable.
    """  
    z = cond

    partial_corr = np.zeros(x.shape[1])

    for i in range(z.shape[1]):
        r_xy = np.corrcoef(x[:, i], y[:, i])[0, 1]
        r_xz = np.corrcoef(x[:, i], z[:, i])[0, 1]
        r_yz = np.corrcoef(y[:, i], z[:, i])[0, 1]

        partial_corr[i] = (r_xy - r_xz * r_yz) / np.sqrt((1 - r_xz ** 2) * (1 - r_yz ** 2))

    return partial_corr


def corr(x, y):
    """ Pearson's correlation between x and y.
     2D array [num_samples(timestamps), dim]
    """

    corr = np.zeros(x.shape[1])

    for i in range(x.shape[1]):
        corr[i] = np.corrcoef(x[:, i], y[:, i])[0, 1]

    return corr



# Structural Hamming Distance (from the Causal Discovery Toolbox)
# Ref: https://fentechsolutions.github.io/CausalDiscoveryToolbox/html/_modules/cdt/metrics.html#SHD

def SHD(true_mat, pred_mat):
    """ Structural Hamming Distance (SHD) between two DAGs.
    Args:
        true_mat: 2D array [num_nodes, num_nodes]
            True adjacency matrix (entries are 0 or 1).
        pred_mat: 2D array [num_nodes, num_nodes]
            Predicted adjacency matrix (entries are 0 or 1).
    Returns:
        SHD score.
    """
    diff=np.abs(true_mat-pred_mat)
    diff=diff+diff.transpose()
    diff[diff > 1] = 1  # Ignoring the double edges.
    return np.sum(diff)
                
