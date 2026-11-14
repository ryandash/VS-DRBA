import torch
from .softsplat import softsplat as warp
from .utils import distance_calculator

_bounds_cache: dict[tuple[str, tuple[int, ...], torch.dtype], tuple[torch.Tensor, torch.Tensor]] = {}


def _get_cached_bounds(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    key = (str(tensor.device), tuple(tensor.shape), tensor.dtype)
    cached = _bounds_cache.get(key)
    if cached is None:
        zeros = torch.zeros(tensor.shape, device=tensor.device, dtype=tensor.dtype)
        ones = torch.ones(tensor.shape, device=tensor.device, dtype=tensor.dtype)
        cached = (zeros, ones)
        _bounds_cache[key] = cached
    return cached


def get_drm_t(drm: torch.Tensor, t: float, precision: float = 1e-3) -> torch.Tensor:
    """
    DRM is a tensor with dimensions b, 1, h, w, where for any value x (0 < x < 1).
    We define the timestep of the entire DRM tensor as 0.5, want the entire DRM approach t,
    but during this process, all values in the tensor must maintain their original proportions.
    """
    dtype = drm.dtype

    _x, b = 0.5, 0.5
    l, r = 0.0, 1.0

    d = drm.float()
    x_drm = d.clone()
    l_drm, r_drm = _get_cached_bounds(d)

    while abs(_x - t) > precision:
        if _x > t:
            r = _x
            _x = _x - (_x - l) * b

            r_drm = x_drm
            x_drm = x_drm - (x_drm - l_drm) * d
        else:
            l = _x
            _x = _x + (r - _x) * b

            l_drm = x_drm
            x_drm = x_drm + (r_drm - x_drm) * d

    return x_drm.to(dtype)


def calc_drm_gmfss(t, flow10, flow12, metric10, metric12, linear=False):
    d10 = distance_calculator(flow10)
    d12 = distance_calculator(flow12)

    drm10 = d10 / (d10 + d12)
    drm12 = d12 / (d10 + d12)

    warp_method = 'soft' if (metric10 is not None and metric12 is not None) else 'avg'

    if linear:
        drm1t_t01 = drm12 * t * 2
        drm1t_t12 = drm10 * t * 2
        drm0t_t01_unaligned = 1 - drm1t_t01
        drm2t_t12_unaligned = 1 - drm1t_t12
    else:
        drm1t_t01 = get_drm_t(drm12, t)
        drm1t_t12 = get_drm_t(drm10, t)
        drm0t_t01_unaligned = 1 - drm1t_t01
        drm2t_t12_unaligned = 1 - drm1t_t12

    _, ones_mask = _get_cached_bounds(drm0t_t01_unaligned)
    warped0 = warp(
        torch.cat((drm0t_t01_unaligned, ones_mask), dim=1),
        flow10, metric10, warp_method,
    )
    drm0t_t01, warped_ones_mask0t_t01 = warped0.split(1, dim=1)
    warped2 = warp(
        torch.cat((drm2t_t12_unaligned, ones_mask), dim=1),
        flow12, metric12, warp_method,
    )
    drm2t_t12, warped_ones_mask2t_t12 = warped2.split(1, dim=1)

    gap_0t_t01 = warped_ones_mask0t_t01 < 0.999
    gap_2t_t12 = warped_ones_mask2t_t12 < 0.999

    drm0t_t01 = torch.where(gap_0t_t01, drm0t_t01_unaligned, drm0t_t01)
    drm2t_t12 = torch.where(gap_2t_t12, drm2t_t12_unaligned, drm2t_t12)

    return {
        "drm0t_t01": drm0t_t01,
        "drm1t_t01": drm1t_t01,
        "drm1t_t12": drm1t_t12,
        "drm2t_t12": drm2t_t12
    }


def calc_drm_rife_auxiliary(t, flow10, flow12, metric10, metric12, linear=False):
    d10 = distance_calculator(flow10) + 1e-4
    d12 = distance_calculator(flow12) + 1e-4

    drm10 = d10 / (d10 + d12)
    drm12 = d12 / (d10 + d12)

    if linear:
        drm_t0_unaligned = drm10 * t * 2
        drm_t1_unaligned = drm12 * t * 2
    else:
        drm_t0_unaligned = get_drm_t(drm10, t)
        drm_t1_unaligned = get_drm_t(drm12, t)

    warp_method = 'soft' if (metric10 is not None and metric12 is not None) else 'avg'

    _, ones_mask = _get_cached_bounds(drm10)
    warped_t01 = warp(
        torch.cat((drm_t1_unaligned, ones_mask), dim=1),
        flow10 * drm_t1_unaligned, metric10, warp_method,
    )
    drm_t1_t01, mask_t1_t01 = warped_t01.split(1, dim=1)
    warped_t12 = warp(
        torch.cat((drm_t0_unaligned, ones_mask), dim=1),
        flow12 * drm_t0_unaligned, metric12, warp_method,
    )
    drm_t1_t12, mask_t1_t12 = warped_t12.split(1, dim=1)

    gap_t1_t01 = mask_t1_t01 < 0.999
    gap_t1_t12 = mask_t1_t12 < 0.999

    drm_t1_t01 = torch.where(gap_t1_t01, drm_t1_unaligned, drm_t1_t01)
    drm_t1_t12 = torch.where(gap_t1_t12, drm_t0_unaligned, drm_t1_t12)

    return {
        "drm_t1_t01": drm_t1_t01,
        "drm_t1_t12": drm_t1_t12
    }
