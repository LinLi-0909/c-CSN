import numpy as np
from scipy.stats import norm
import pandas as pd


def condition_g(adjmc, kk):
    """
    Select genes based on adjacency matrix and kk value.

    Parameters:
    - adjmc: Adjacency matrix (numpy array)
    - kk: Integer (0 or 1). Determines the number of genes to select.

    Returns:
    - id_genes: Indices of selected genes
    """
    degrees = np.sum(adjmc, axis=1)
    id_genes = np.argsort(degrees)[::-1][:kk]
    return id_genes


def network_flow_entropy(data, alpha=0.01, boxsize=1.5, kk=0):
    """
    Construction of conditional cell-specific network (CCSN) and conditional network degree matrix.
    This function transforms a gene expression matrix into a conditional network degree matrix (cndm).

    Parameters:
    - data: Gene expression matrix (numpy array), rows = genes, columns = cells
    - alpha: Significance level (e.g., 0.001, 0.01, 0.05 ...). Larger alpha leads to more edges. Default = 0.01
    - boxsize: Size of neighborhood; a value between 1 to 2 is recommended. Default = 1.5
    - kk: Integer (0 or 1). Determines the depth of conditional network construction.

    Returns:
    - NFE: Network Flow Entropy vector for each cell (numpy array of length equal to the number of cells)
    """

    n1, n2 = data.shape  # n1: number of genes, n2: number of cells
    upbound = np.zeros((n1, n2))
    lowbound = np.zeros((n1, n2))

    # Define the neighborhood of each gene expression profile
    for i in range(n1):
        s2 = np.argsort(data[i, :])         # Indices that would sort data[i, :]
        s1 = data[i, s2]                    # Sorted data for gene i
        num_positive = np.sum(s1 > 0)
        n3 = n2 - num_positive
        h = int(round(boxsize * np.sqrt(num_positive)))
        k = 0
        while k < n2:
            s = 0
            while k + s + 1 < n2 and s1[k + s + 1] == s1[k]:
                s += 1
            indices = s2[k:k + s + 1]
            if s >= h:
                upbound[i, indices] = data[i, s2[k]]
                lowbound[i, indices] = data[i, s2[k]]
            else:
                up_idx = min(n2 - 1, k + s + h)
                upbound[i, indices] = data[i, s2[up_idx]]
                if n3 > h:
                    low_idx = max(int(n3 * (n3 > h)), k - h)
                else:
                    low_idx = max(0, k - h)
                lowbound[i, indices] = data[i, s2[low_idx]]
            k += s + 1

    # Construction of CSN and network degree matrix
    B = np.zeros((n1, n2), dtype=bool)
    NFE = np.zeros(n2)
    p = -norm.ppf(alpha)  # Critical value from the normal distribution

    for k in range(n2):
        # Build the binary matrix B for cell k
        for j in range(n2):
            B[:, j] = (data[:, j] <= upbound[:, k]) & (data[:, j] >= lowbound[:, k])

        a = np.sum(B, axis=1)                # Sum over columns for each gene
        c = np.dot(B, B.T)                   # Co-occurrence matrix
        numerator = c * n2 - np.outer(a, a)
        denominator = np.sqrt(np.outer(a, a) * np.outer(n2 - a, n2 - a) / (n2 - 1) + np.finfo(float).eps)
        adjmc = numerator / denominator
        adjmc = (adjmc > p)

        id_genes = condition_g(adjmc, kk)    # Select genes based on adjmc and kk

        adjmc_total = np.ones((n1, n1), dtype=bool)
        for m in range(kk):
            idx = id_genes[m]
            B_z = B[idx, :] & B
            idc = np.nonzero(B[idx, :])[0]
            B_z = B_z[:, idc]
            r = B_z.shape[1]
            a_z = np.sum(B_z, axis=1)
            c_z = np.dot(B_z, B_z.T)
            numerator = c_z * r - np.outer(a_z, a_z)
            denominator = np.sqrt(np.outer(a_z, a_z) * np.outer(r - a_z, r - a_z) / (r - 1) + np.finfo(float).eps)
            adjmc1 = numerator / denominator
            adjmc1 = (adjmc1 > p)
            adjmc_total &= adjmc1

        # Calculate Network Flow Entropy (NFE)
        P = np.outer(data[:, k], data[:, k]) * adjmc_total
        id_nonzero = np.nonzero(np.sum(P, axis=1))[0]
        x = data[id_nonzero, k]
        x_n = x / np.sum(x)
        P1 = P[id_nonzero][:, id_nonzero]
        sum_P1 = np.sum(P1, axis=1)
        sum_P1[sum_P1 == 0] = np.finfo(float).eps  # Avoid division by zero
        P_n = P1 / sum_P1[:, np.newaxis]
        x_p = P_n * x_n[:, np.newaxis]
        x_p[x_p == 0] = 1  # To handle log(0), set zeros to 1 (since log(1) = 0)
        NFE[k] = -np.sum(x_p * np.log(x_p))
        print(f"Processed cell {k + 1}/{n2}")

    return NFE


if __name__ == '__main__':
    exp = pd.read_csv('exp1.csv', index_col=0)
    mat = exp.values
    nfe = network_flow_entropy(mat)
    nfe = pd.DataFrame(nfe, index=exp.columns, columns=['entropy'])
    nfe.to_csv('nfe1.csv')