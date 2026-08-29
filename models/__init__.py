"""
action_agent_vllm 识别器模块

提供四种不同的动作识别策略，封装在面向对象的类中。
"""

from .base_recognizer import BaseRecognizer
from .main_recognizer import MainRecognizer
from .base1_recognizer import Base1Recognizer
from .base2_recognizer import Base2Recognizer
from .base3_recognizer import Base3Recognizer

__all__ = [
    'BaseRecognizer',
    'MainRecognizer',
    'Base1Recognizer',
    'Base2Recognizer',
    'Base3Recognizer'
]
