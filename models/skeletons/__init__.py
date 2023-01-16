'''
* Copyright (c) AVELab, KAIST. All rights reserved.
* author: Donghee Paek & Kevin Tirta Wijaya, AVELab, KAIST
* e-mail: donghee.paek@kaist.ac.kr, kevin.tirta@kaist.ac.kr
'''

from .rdr_base import RadarBase
from .ldr_base import LidarBase

def build_skeleton(cfg):
    return __all__[cfg.MODEL.SKELETON](cfg)

__all__ = {
    'RadarBase': RadarBase,
    'LidarBase': LidarBase,
}
