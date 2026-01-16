# Ultralytics Multimodal Module
# Universal RGB+X multimodal support for YOLO and RTDETR

from .router import MultiModalRouter
from .parser import MultiModalConfigParser

__all__ = ['MultiModalRouter', 'MultiModalConfigParser']