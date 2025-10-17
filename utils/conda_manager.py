import logging
from utils.base_environment_manager import EnvironmentManager
from utils.command_runner import run_command

logger = logging.getLogger(__name__)


class CondaManager(EnvironmentManager):
    def run_conda_command(self, command: str, **kwargs):
        """Run a conda command, sourcing conda if necessary. Returns the completed process."""
        final_command = "source /opt/conda/bin/activate && " + command
        return run_command(final_command, **kwargs)

    def create(self, env_file: str = None, check: bool = True, silent: bool = True):
        """Create a conda environment from the specified environment file."""
        if env_file is None:
            raise ValueError("env_file must be specified to create an environment.")
        self.run_conda_command(
            f"conda env create --file {env_file}", check=check, silent=silent
        )
        logger.debug(f"Conda environment '{self.env_name}' created.")

    def create_empty(self, python_version: str, check: bool = True):
        """Create an empty conda environment with the specified Python version."""
        if not python_version:
            raise ValueError("python_version must be specified to create an empty environment.")
        self.run_conda_command(f"conda create -y -n {self.env_name} python={python_version}", check=check)
        logger.debug(f"Empty conda environment '{self.env_name}' with Python {python_version} created.")

    def exists(self):
        """Check if a conda environment exists."""
        result = self.run_conda_command("conda env list")
        return self.env_name in result.stdout

    def install_requirements(self, requirements_file: str = None):
        """Install Python packages from a requirements.txt file using pip."""
        if requirements_file is None:
            raise ValueError(
                "requirements_file must be specified to install requirements."
            )
        self.run_conda_command(f"pip install -r {requirements_file}")
        logger.debug(f"Installed packages from '{requirements_file}'.")

    def run_command(self, command: str, **kwargs):
        """
        Run a shell command with the conda environment activated.
        """
        activate_cmd = f"source /opt/conda/bin/activate && conda activate {self.env_name} && {command}"
        return run_command(activate_cmd, **kwargs)
