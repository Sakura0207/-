"""检测器基类接口"""
from abc import ABC, abstractmethod
from ..models import RuleResult


class BaseDetector(ABC):
    """所有规则检测器的抽象基类"""

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def detect(self, reply: str, knowledge_base: str) -> RuleResult:
        ...
