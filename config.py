# Configuration settings for the Gaussian Splatting framework
from typing import List, Tuple
import os


MIPNERF360_DATASET = "/mnt/projects/mlmi/gaussian_splatting/datasets/mip-nerf-360"
DB_DATASET = "/mnt/projects/mlmi/gaussian_splatting/datasets/deep-blending"
TANDT_DATASET = "/mnt/projects/mlmi/gaussian_splatting/datasets/tanks-and-temples"
DYNAMIC_GAUSSIAN_DATASET = "/tmp/dynamic-gaussian/data"
ENDOGAUSSIAN_DATASET = "/mnt/projects/mlmi/gaussian_splatting/datasets/endonerf"

# BASE_DIR = "/mnt/projects/mlmi/gaussian_splatting"
BASE_DIR = os.path.expanduser("~")
FRAMEWORK_BASE = f"{BASE_DIR}/gs_framework"
GS_DIR = f"{BASE_DIR}/orig_gs"
LIGHT_GS_DIR = f"{BASE_DIR}/light_gaussian_splatting"
LANGSPLAT_DIR = f"{BASE_DIR}/lang_splat"
VIEWER_DIR = f"{BASE_DIR}/viewer"
DYNAMIC_GS_DIR = f"{BASE_DIR}/dynamic_gaussians"
ENDO_GAUSSIAN_DIR = f"{BASE_DIR}/endo_gaussian"
DEPTH_MAPS_DIR = f"{BASE_DIR}/depth_anything_repo"

def configure_project():
    """
    Sets environment variables for the framework's required directory paths.

    This function assigns the following environment variables using their corresponding
    global variables:
        - FRAMEWORK_BASE: Base directory for the framework.
        - GS_DIR: Path to the GS directory.
        - LIGHT_GS_DIR: Path to the Light GS directory.
        - LANGSPLAT_DIR: Path to the LangSplat directory.
        - VIEWER_DIR: Path to the Viewer directory.
        - DYNAMIC_GS_DIR: Path to the Dynamic GS directory.
        - DEPTH_MAPS_DIR: Path to the Depth Maps directory.
        - MIPNERF360_DATASET: Path to the MipNeRF360 dataset.
        - DB_DATASET: Path to the deep blending dataset.
        - TANDT_DATASET: Path to the Tanks and Temples dataset.

    These environment variables are used throughout the project to locate necessary resources.
    """
    os.environ["FRAMEWORK_BASE"] = FRAMEWORK_BASE
    os.environ["GS_DIR"] = GS_DIR
    os.environ["LIGHT_GS_DIR"] = LIGHT_GS_DIR
    os.environ["LANGSPLAT_DIR"] = LANGSPLAT_DIR
    os.environ["VIEWER_DIR"] = VIEWER_DIR
    os.environ["DYNAMIC_GS_DIR"] = DYNAMIC_GS_DIR
    os.environ["ENDO_GAUSSIAN_DIR"] = ENDO_GAUSSIAN_DIR
    os.environ["DEPTH_MAPS_DIR"] = DEPTH_MAPS_DIR
    os.environ["MIPNERF360_DATASET"] = MIPNERF360_DATASET
    os.environ["DB_DATASET"] = DB_DATASET
    os.environ["TANDT_DATASET"] = TANDT_DATASET

def datasets_mounted() -> bool:
    """
    Check if the datasets are mounted by verifying the existence of dataset directories.
    Returns True if all datasets are mounted, False otherwise.
    """
    return all(os.path.exists(dataset) for dataset in [MIPNERF360_DATASET, DB_DATASET, TANDT_DATASET])

def get_all_dataset_paths(scenario: str = None) -> List[Tuple[str, str]]:
    """
    Iterate over all dataset directories and return a list of all dataset names with dataset paths.
    If scenario is not None, return path and scenario name of that folder.
    """
    if not datasets_mounted():
        raise FileNotFoundError("Datasets are not mounted. Please mount the datasets before proceeding.")
    
    paths = []
    for dataset in [MIPNERF360_DATASET, DB_DATASET, TANDT_DATASET]:
        for root, dirs, files in os.walk(dataset):
            for dir_name in dirs:
                subfolder_path = os.path.join(root, dir_name)
                paths.append((dir_name, subfolder_path))

    if scenario:
        # Filter the paths to only return the specified scenario
        paths = [(name, path) for name, path in paths if name == scenario]
        return paths
    else:
        return paths
