import os
import subprocess
import logging

logger = logging.getLogger(__name__)


class RepoManager:
    def __init__(self, repo_dir: str = None, repo_url: str = None, commit_hash: str = None):
        self.repo_dir = repo_dir
        self.repo_url = repo_url
        self.commit_hash = commit_hash

    def check_repo_exists(self) -> bool:
        """Check if the repository exists at the specified path."""
        return (
            self.repo_dir is not None
            and os.path.exists(self.repo_dir)
            and os.path.isdir(self.repo_dir)
            and os.path.exists(os.path.join(self.repo_dir, ".git"))
        )

    def clone_repo(self):
        target_dir = self.repo_dir if self.repo_dir is not None else self.repo_dir
        if not target_dir or not self.repo_url:
            raise ValueError("Both repo_url and repo_dir (or repo_path) must be set.")
        self.repo_dir = target_dir  # Update the instance's repo_path
        if not self.check_repo_exists():
            logger.debug(f"Cloning repository from {self.repo_url} to {target_dir}...")
            try:
                subprocess.check_call(
                    ["git", "clone", self.repo_url, target_dir, "--recursive"]
                )
                if self.commit_hash is not None:
                    subprocess.run(["git", "checkout", self.commit_hash], cwd=target_dir, check=True)
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to clone repository: {e}")
                raise
        else:
            logger.debug(f"Repository already exists at {target_dir}. Skipping clone.")
