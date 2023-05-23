"""
    Diffusion approximation solution
    - [x] half infinity space | Neumann
    - [x] half infinity space | Dirichlet
    - [x] half infinity space | Neumann
    - [x] full infinity space | Dirichlet
    - [x] Taichi implementation for 2D visualization
    @author: Qianyue He
    @date: 2023-5-23
"""

import numpy as np
import taichi as ti
import taichi.math as tm
SOL = 299792458

@ti.func
def exp_power_ti(x: float, t: float, tau: float, eps: float, u_a: float, D: float, c: float = SOL):
    dt = t - tau
    return ti.exp(
        -((x - eps) ** 2) / (4. * c * D * dt) -  u_a * c * dt
    )

def exp_power(x, t, tau, eps, u_a, D, c = SOL):
    dt = t - tau
    return np.exp(
        -((x - eps) ** 2) / (4 * c * D * dt) -  u_a * c * dt
    )

@ti.experimental.real_func
def half_diffusion_ti(
    x: float, t: float, tau: float, eps: float, 
    u_a: float, D: float, c: float = SOL, sub: int = 0
) -> float :
    dt = ti.max(t - tau, 0.)
    coeff = c * dt / ti.sqrt(4 * tm.pi * c * D * dt)
    if tm.isnan(coeff):         # only when dt = 0, can coeff and result both be NaN
        return 0
    else:
        result = coeff * exp_power_ti(x, t, tau, eps, u_a, D, c)
        sec_part = coeff * exp_power_ti(x, t, tau, -eps, u_a, D, c)
        result += ti.select(sub == 0, sec_part, -sec_part)
        return result
    
@ti.experimental.real_func
def full_diffusion_ti(
    x: float, t: float, tau: float, eps: float, 
    u_a: float, D: float, c: float = SOL
) -> float :
    dt = np.maximum(t - tau, 0)
    coeff = c * dt / ti.sqrt(4 * np.pi * c * D * dt)
    if tm.isnan(coeff):
        return 0
    return coeff * exp_power_ti(x, t, tau, eps, u_a, D, c)

def half_diffusion(x, t, tau, eps, u_a, D, c = SOL, sub = False):
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
    result = coeff * exp_power(x, t, tau, eps, u_a, D, c)
    if sub:
        result -= coeff * exp_power(x, t, tau, -eps, u_a, D, c)
    else:
        result += coeff * exp_power(x, t, tau, -eps, u_a, D, c)
    return np.where(np.isnan(result), 0, result)

def full_diffusion(x, t, tau, eps, u_a, D, c = SOL):
    dt = np.maximum(t - tau, 0)
    coeff = c * dt / np.sqrt(4 * np.pi * c * D * dt)
    result = coeff * exp_power(x, t, tau, eps, u_a, D, c)
    return np.where(np.isnan(result), 0, result)
