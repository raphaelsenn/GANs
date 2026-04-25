import torch


def interpolate_noise(
        z0: torch.Tensor, 
        z1: torch.Tensor, 
        n_steps: int
) -> torch.Tensor:
    """Simple manifold interpolation.""" 
    alpha = torch.linspace(0, 1, n_steps)
    zs = []
    for i in range(n_steps):
        z = (1 - alpha[i]) * z0 + alpha[i] * z1     # [N, nz]
        zs.append(z)
    return torch.stack(zs, dim=1)                   # [N, n_steps, nz]