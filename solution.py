from abc import ABC, abstractmethod
import numpy as np

class Solution(ABC):
    @abstractmethod
    def __init__(self, env):
        pass
    
    @abstractmethod
    def solve(self):
        pass