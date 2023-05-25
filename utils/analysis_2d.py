"""
    2D analysis for computed diffusion approximation
"""

import numpy as np
import taichi as ti

from taichi.math import vec2
from da import full_diffusion_2d, half_diffusion_2d, tr_2d, get_diffusion_length_ti
from visualize_1d import normalized_integral, inverse_sampling, transmittance_sampling

def numpy_max(pixels: ti.template):
    part_value: np.ndarray = pixels.to_numpy()[..., 0]
    return np.unravel_index(part_value.argmax(), part_value.shape)

@ti.data_oriented
class AnalysisTool:
    MAX_RAY_MARCH_NUM = 512
    def __init__(self, sol, diff_mode: int = 0, max_ray_march_num: int = 400, sample_num = 256) -> None:
        self._sample_num        = ti.field(int, ())
        self._max_ray_march_num = ti.field(int, ())
        self._sample_num[None]  = sample_num
        self._max_ray_march_num[None] = max_ray_march_num
        if max_ray_march_num > AnalysisTool.MAX_RAY_MARCH_NUM:
            print(f"Warning: ray marching sample num is too big, setting to {self.MAX_RAY_MARCH_NUM}")

        self._values = ti.field(float, self.MAX_RAY_MARCH_NUM)
        self._trs    = ti.field(float, self.MAX_RAY_MARCH_NUM)
        self.diff_mode = diff_mode
        self.sol = sol
        self.ray_marched = False

    def setter(self, configs: dict):
        self.sample_num = configs['samp_num']
        self.max_ray_march_num = configs['rm_num']

    @ti.kernel
    def ray_marching(self, 
        vx: float, vy: float, ex: float, ey: float, 
        ua: float, us: float, target: vec2, use_tr: int, target_time: float
    ) -> float:
        """ Direction is not normalized 
            Actually, if we find a way to calculate diffusion faster and
            making use of SIMD, we can have fast ray marching (64/128 samples)
        """
        start_v = vec2([vx, vy])
        lit_pos = vec2([ex, ey])
        D = get_diffusion_length_ti(ua, us)
        direction = target - start_v
        ray_march_num = self._max_ray_march_num[None]
        for i in self._values:
            position = start_v + float(i + 1) / float(ray_march_num) * direction
            val = 0.
            t = 0.
            if target_time > 0.:      # using remaining time
                t = target_time - (position - start_v).norm()
            else:                # using light propagation time
                t = (position - lit_pos).norm()
            if ti.static(self.diff_mode == 0):      # Full infinite sapce
                val = full_diffusion_2d(position[0], position[1], t, 0, ex, ey, ua, D, self.sol)
            else:                                   # Half infinite sapce
                val = half_diffusion_2d(position[0], position[1], t, 0, ex, ey, ua, D, self.sol)
            transmittance = tr_2d(position[0], position[1], vx, vy, ua, us)
            self._trs[i] = transmittance
            if use_tr and val > 0.:
                val *= transmittance
            self._values[i] = val
        return direction.norm()

    def inverse_sampling(self, length: float, ua: float, us: float):
        """ Inverse sampling the ray marched result """
        # get positions first
        xs = np.linspace(0, length, self.max_ray_march_num + 1)[1:]
        da_integral = normalized_integral(self.da_solution)
        da_samples = inverse_sampling(da_integral, xs, self.sample_num)
        tr_samples = transmittance_sampling(self.sample_num, ua, us)
        return da_samples, tr_samples
    
    @staticmethod
    def get_histograms(samples: np.ndarray, bin_num: int):
        """ Get histogram analysis """
        max_pos = samples.max()
        bins = np.linspace(0, max_pos, bin_num)
        hist, _ = np.histogram(samples, bins)
        return bins[:-1], hist.astype(bins.dtype)
    
    @property
    def transmittance(self):
        return self._trs.to_numpy()[:self.max_ray_march_num]
    
    @property
    def da_solution(self):
        return self._values.to_numpy()[:self.max_ray_march_num]
    
    @property
    def max_ray_march_num(self):
        return self._max_ray_march_num[None]

    @max_ray_march_num.setter
    def max_ray_march_num(self, val):
        self._max_ray_march_num[None] = max(32, min(val, self.MAX_RAY_MARCH_NUM))

    @property
    def sample_num(self):
        return self._sample_num[None]

    @sample_num.setter
    def sample_num(self, val):
        self._sample_num[None] = max(1, val)
