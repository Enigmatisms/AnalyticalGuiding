"""
    Base class for diffusion visualization, this class hides some
    implementation details and utility functions from the main logic
    @author: Qianyue He
    @date: 2023-5-23
"""

import taichi as ti
from da import *

@ti.data_oriented
class DiffusionVizBase:
    FULL_DIFFUSE = 0
    HALF_DIFFUSE = 1
    def __init__(self, prop: dict) -> None:
        self.w, self.h   = prop['width'], prop['height']
        self.diffuse_mode = prop['diffuse_mode']

        self.coeffs       = ti.field(float, shape = 3)
        self._max_time    = ti.field(float, shape = ())
        self._emitter_pos = ti.field(float, shape = ())
        self._time        = ti.field(float, shape = ())
        # position (canvas) scaling
        self._scale       = ti.field(float, shape = ())
        # image value scaling
        self._v_scale     = ti.field(float, shape = ())

        self.setter(prop)

    def setter(self, configs: dict):
        self._v_scale[None] = configs['v_scale']
        self.max_time       = configs['max_time']
        self.emitter_pos    = configs['emitter_pos']
        self.time           = configs['time']
        self.scale          = configs['scale']
        self.coeffs[1]      = configs['ua']
        self.us             = configs['us']

    @property
    def us(self):
        return self.coeffs[0]
    
    @us.setter
    def us(self, val):
        if val < 0.0:
            raise ValueError("Scattering coefficient should be non-negative")
        self.coeffs[0] = val
        self.coeffs[2] = val + self.coeffs[1]

    @property
    def ua(self):
        return self.coeffs[1]

    @ua.setter
    def ua(self, val):
        if val < 0.0:
            raise ValueError("Absorption coefficient should be non-negative")
        self.coeffs[1] = val
        self.coeffs[2] = val + self.coeffs[0]

    @property
    def ut(self):
        return self.coeffs[2]
    
    @property
    def max_time(self):
        return self._max_time[None]
    
    @max_time.setter
    def max_time(self, val):
        if val <= 0.0:
            raise ValueError("Max time should be positive")
        self._max_time[None] = val

    @property
    def emitter_pos(self):
        return self._emitter_pos[None]
    
    @emitter_pos.setter
    def emitter_pos(self, val):
        self._emitter_pos[None] = val

    @property
    def time(self):
        return self._time[None]
    
    @time.setter
    def time(self, val):
        if val < 0.0 or val > self.max_time:
            raise ValueError(f"Time should be in range [0, {self.max_time}]")
        self._time[None] = val

    @property
    def scale(self):
        return self._scale[None]
    
    @scale.setter
    def scale(self, val):
        if val <= 0.0:
            raise ValueError("Scale should be positive")
        self._scale[None] = val

    @property
    def v_scale(self):
        return self._v_scale[None]
    
    @v_scale.setter
    def v_scale(self, val):
        if val <= 0:
            raise ValueError(f"Inappropriate image scaling value")
        self._v_scale[None] = val
