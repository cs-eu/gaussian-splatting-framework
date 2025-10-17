from abc import ABC, abstractmethod


class EnvironmentManager(ABC):
    def __init__(self, env_name: str):
        self.env_name = env_name

    @abstractmethod
    def create(self):
        """Create the environment."""
        pass

    @abstractmethod
    def exists(self):
        """Check if the environment exists."""
        pass
