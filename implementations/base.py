from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


class Base(ABC):
    """Base class for implementations."""

    @abstractmethod
    def check_env_exists(self) -> bool:
        """Check if the environment exists."""
        pass

    @abstractmethod
    def create_env(self):
        """Create the environment."""
        pass

    @abstractmethod
    def check_repo_exists(self) -> bool:
        """Check if the repository exists."""
        pass

    @abstractmethod
    def create_repo(self):
        """Clone or create the repository."""
        pass

    @abstractmethod
    def run(self, *args, **kwargs):
        """Run the implementation."""
        pass

    def setup(self):
        """Setup the implementation by checking and creating the environment and repository."""
        # check if the repository exists, if not, clone it
        if not self.check_repo_exists():
            logger.info("Creating repository...")
            self.create_repo()

        # check if the environment exists, if not, create it
        if not self.check_env_exists():
            logger.info("Creating environment...")
            self.create_env()

    def execute(self, *args, **kwargs):
        """Execute the implementation."""
        self.setup()

        # run the implementation
        self.run(*args, **kwargs)
