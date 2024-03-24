import torch
from generate_samples import TENSOR_PROP

def inverse_transform_sampling(pdf_tensor: torch.Tensor, N: int) -> torch.Tensor:
    """ Inverse transform sampling

    Args:
        pdf_tensor (torch.Tensor): PDF of different choices
        N (int): number of samples

    Returns:
        torch.Tensor : sampled indices
    """
    cdf = torch.cumsum(pdf_tensor, dim = 0)
    uniform_samples = torch.rand(N, **TENSOR_PROP)
    indices = torch.searchsorted(cdf, uniform_samples, right=True)
    return indices

if __name__ == "__main__":
    pdf_tensor = torch.tensor([0.2, 0.3, 0.4, 0.1])
    N = 10000

    samples = inverse_transform_sampling(pdf_tensor, N)

    print("Sampling results:")
    for i in range(4):
        print(f"{i} ratio:", (samples == i).sum() / N)