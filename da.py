"""
    Diffusion approximation solution
    @author: Qianyue He
    @date: 2023-5-23
"""

import numpy as np
SOL = 299792458

def exp_power(x, t, tau, eps, u_a, D, c = SOL):
    dt = t - tau
    return np.exp(
        -((x - eps) ** 2) / (4 * c * D * dt) -  u_a * c * dt
    )

def half_diffusion_neumann(x, t, tau, eps, u_a, D, c = SOL):
    """ - x: one-dim position
        - t: temporal coordinates
        - tau: the time point of which the emitter starts to emit pulse light
        - eps: the spatial position of the emitter
        - c: speed of light (can be unitless)
        - u_a: absorption coefficient
        - D: 1 / 3 * (ua + (1 - g) * us)
    """
    dt = np.maximum(t - tau, 0)
    coeff = c * dt / np.sqrt(4 * np.pi * c * D * dt)
    result = coeff * (exp_power(x, t, tau, -eps, u_a, D, c) + exp_power(x, t, tau, eps, u_a, D, c))
    return np.where(np.isnan(result), 0, result)