import torch
import torch.nn.functional as F

def cp_defense_lucia(features, compression_ratio=32):
    """
    Defense mechanism using average pooling to reduce feature size, normalizing features and computing L1 distance in parallel.
    
    Args:
    - features: Tensor of shape (n, C, W, H), where n is the number of agents, C is the number of channels, 
                and W and H are the spatial dimensions.
    - threshold: A float value used to filter out agents based on L1 distance scores.
    - top_k: The number of top dissimilar agents to return.
    - compression_ratio: The factor by which to reduce the spatial dimensions using average pooling.
    
    Returns:
    - mask: A binary mask of shape (n,) where the top k most dissimilar agents are marked with 1, others with 0.
    """
    features = features.detach()#.cpu()
    n, C, W, H = features.shape

    # Step 1: Normalize the features (L2 normalization across spatial dimensions and channels)
    pooled_features = F.avg_pool2d(features, kernel_size=compression_ratio)  # Reduce spatial dimensions by compression_ratio

    # Step 2: Normalize the pooled features (L2 normalization across spatial dimensions and channels)
    norms = torch.norm(pooled_features, p=2, dim=(1, 2, 3), keepdim=True)  # Compute L2 norm for each agent
    normalized_pooled_features = pooled_features / norms  # Normalize the pooled features
    l1_scores = torch.zeros(n, device=features.device)  # To store the sum of L1 distances for each agent
    
    if n == 2:
        l1_scores[1] = torch.sum(torch.abs(normalized_pooled_features[1] - normalized_pooled_features[0]))
        l1_scores[0] = 0.

    else:
        # Compute L1 distance incrementally
        for i in range(n):
            for j in range(i + 1, n):
                # Compute L1 distance between agent i and agent j on the pooled feature map
                l1_distance = torch.sum(torch.abs(normalized_pooled_features[i] - normalized_pooled_features[j]))
                l1_scores[i] += l1_distance
                l1_scores[j] += l1_distance

    l1_scores[0] = torch.min(l1_scores)

    l1_scores = torch.nn.functional.softmax(l1_scores, dim=0)
    # Print L1 distance scores for inspection
    print("Trustworthiness Score:", 1 - l1_scores)
    
    return 1 - l1_scores


