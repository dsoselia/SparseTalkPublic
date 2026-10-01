import torch
try:
    import cupy as cp
except ImportError:
    cp = None



def kmeans_cupy(features, num_clusters=10, max_iters=100, tol=1e-4):
    """
    Perform K-Means clustering on 256-dimensional features using CuPy (GPU-accelerated).
    """
    if cp is None:
        raise ImportError("cupy is required for kmeans_cupy")
    
    N, D = features.shape


    features = cp.asarray(features.detach().cpu().numpy()) 
    features = cp.asarray(features).astype(cp.float32)

    # Initialize cluster centers randomly from dataset
    indices = cp.random.permutation(N)[:num_clusters]
    cluster_centers = features[indices]

    prev_cluster_centers = cp.zeros_like(cluster_centers)
    
    for i in range(max_iters):
        # Compute squared Euclidean distances between each feature and cluster centers
        distances = cp.linalg.norm(features[:, None, :] - cluster_centers[None, :, :], axis=2)  # Shape: (N, num_clusters)

        # Assign each point to the nearest cluster
        cluster_assignments = cp.argmin(distances, axis=1)

        # Compute new cluster centers
        new_cluster_centers = cp.zeros_like(cluster_centers)
        counts = cp.zeros(num_clusters, dtype=cp.int32)

        for cluster_idx in range(num_clusters):
            mask = cluster_assignments == cluster_idx
            if cp.any(mask):
                new_cluster_centers[cluster_idx] = cp.mean(features[mask], axis=0)
                counts[cluster_idx] = cp.sum(mask)

        # Avoid division by zero
        valid_clusters = counts > 0
        cluster_centers[valid_clusters] = new_cluster_centers[valid_clusters]

        # Check convergence
        centroid_shift = cp.linalg.norm(cluster_centers - prev_cluster_centers, axis=1).mean()
        prev_cluster_centers[:] = cluster_centers

        if centroid_shift < tol:
            print(f"Converged in {i+1} iterations.")
            break

    # cluster_assignments = torch.utils.dlpack.from_dlpack(cp.asarray(cluster_assignments))
    cluster_assignments = torch.tensor(cluster_assignments, device="cpu")


    return cluster_assignments, cluster_centers


def compute_feature_entropy(features):
    """
    Compute entropy of Gaussian features.
    
    Args:
        features (torch.Tensor): Tensor of shape (N, D) where 
                                 N = number of Gaussians,
                                 D = feature dimension.
    
    Returns:
        entropy (torch.Tensor): Tensor of shape (N,) containing entropy values.
    """
    # Normalize features to probabilities (softmax over feature dimension)
    probs = torch.softmax(features, dim=1)

    # Compute entropy: H(X) = -sum(p * log(p)) across feature dimension
    entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)

    return entropy

def select_high_entropy_gaussians(features, num_samples, device='cuda'):
    """
    Select the top-K Gaussians with the highest entropy.
    """
    # Compute entropy for each Gaussian
    entropy = compute_feature_entropy(features.to(device))

    # Get the indices of the top-K highest entropy Gaussians
    topk_indices = torch.topk(entropy, num_samples, largest=True).indices

    # Select the corresponding features
    # selected_features = features[topk_indices]

    return topk_indices.to('cpu')


def compute_knn_density_cupy_blockwise(points, k=10, block_size=8192):
    """
    Compute density of each point using k-NN distances with CuPy

    """
    if cp is None:
        raise ImportError("cupy is required for compute_knn_density_cupy_blockwise")
    points_cp = cp.asarray(points.cpu().numpy())  # Convert to CuPy
    N = points_cp.shape[0]
    density_values = cp.zeros(N)

    for i in range(0, N, block_size):
        end_i = min(i + block_size, N)

        # Compute pairwise distances in chunks
        chunk = points_cp[i:end_i]
        distances = cp.linalg.norm(chunk[:, None, :] - points_cp[None, :, :], axis=2)

        # Sort distances & select k nearest neighbors (excluding self)
        sorted_dists = cp.sort(distances, axis=1)[:, 1:k+1]  # Exclude self at index 0

        # Compute density
        density_values[i:end_i] = 1.0 / (cp.mean(sorted_dists, axis=1) + 1e-6)

    # Convert back to PyTorch tensor
    density = torch.tensor(cp.asnumpy(density_values), device=points.device)
    
    return density

def sample_points_adaptive_cupy(points, num_samples, k=10, replace=False, device="cuda"):
    """
    Sample 3D points adaptively based on local density using CuPy.

    """
    points = points.to(device)

    # Compute density in a memory-efficient manner
    density = compute_knn_density_cupy_blockwise(points, k=k)

    # Normalize density for sampling
    probs = density / density.sum()

    # Sample indices based on density probabilities
    sampled_indices = torch.multinomial(probs, num_samples, replacement=replace)

    # Retrieve sampled points
    # sampled_points = points[sampled_indices]

    return sampled_indices

def multivariate_gaussian_pdf(x, mean, cov):
    """
    Computes the probability density function (PDF) of a multivariate Gaussian at point x.
    """
    D = mean.shape[-1]  # Dimensionality
    diff = x - mean  # Difference (N, D)
    
    # Compute determinant and inverse of covariance matrix
    cov_inv = torch.inverse(cov)  # (N, D, D)
    cov_det = torch.det(cov)  # (N,)
    
    # Mahalanobis distance
    exponent = -0.5 * torch.einsum("bi,bij,bj->b", diff, cov_inv, diff)  # (N,)

    # Normalization constant
    norm_const = torch.sqrt((2 * torch.pi) ** D * cov_det)
    
    return torch.exp(exponent) / norm_const

def retrieve_language_feature(gaussians, language_features, device='cuda'):
    """
    Retrieves the aggregated language feature considering Gaussian parameters.
    """
    mean = gaussians['points'].to(device)  # (N, D)
    cov = gaussians['covariances'].to(device)  # (N, D, D)
    opacity = gaussians['opacities'].to(device)  # (N,)
    language_features = language_features.to(device)  # (N, F)

    # import pdb; pdb.set_trace()

    final_feature = opacity[:, None] * language_features  

    return final_feature
