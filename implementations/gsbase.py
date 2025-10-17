from abc import abstractmethod
import os
import threading
import time
import logging
from implementations.base import Base
from implementations.viewer import Viewer

logger = logging.getLogger(__name__)


class GSBase(Base):
    """Base class for Gaussian Splatting implementations."""

    def __init__(self, use_viewer: bool = False):
        super().__init__()
        self.use_viewer = use_viewer
        
        # Setup viewer if needed
        if self.use_viewer:
            logger.info("Setting up viewer")
            self.viewer = Viewer()
            self.viewer.setup()
        self.viewer_thread = None

    def start_viewer(self, path_to_ply: str):
        logger.info("Starting viewer for Gaussian Splatting...")

        def viewer_thread_func():
            while not os.path.exists(path_to_ply):
                time.sleep(1)
            self.viewer.run(path_to_ply, live_update=True)

        self.viewer_thread = threading.Thread(target=viewer_thread_func)
        self.viewer_thread.start()

    def is_viewer_running(self):
        """Check if the viewer thread is running."""
        return self.viewer_thread is not None and self.viewer_thread.is_alive()

    @abstractmethod
    def train(self, scene_path: str, output_path: str, *args, **kwargs) -> str:
        """Train the Gaussian Splatting model on a given scene.

        Returns:
            str: The path to the training result.
        """
        pass
