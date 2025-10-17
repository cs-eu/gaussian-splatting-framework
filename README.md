<table>
  <tr>
    <td style="vertical-align: middle; width: 100px;">
      <img src="logo.png" alt="Gaussian Splatting Framework Logo" width="90">
    </td>
    <td style="vertical-align: middle;">
      <h1 style="margin: 0;">Gaussian Splatting Framework</h1>
      <em>A Unified Framework for State-of-the-Art Gaussian Splatting Methods</em>
    </td>
  </tr>
</table>

---

This repository collects scripts, code, and utilities to run and evaluate various implementations of Gaussian Splatting for 3D scene reconstruction and rendering.

## Project Structure

- **implementations/**: Wrappers and scripts for running different Gaussian Splatting implementations.
  - `base.py`: Base class for implementations.
  - `original_gaussian_splatting.py`: Integration for the original 3D Gaussian Splatting method.
  - `create_depth_maps.py`: Depth map creation utilities.
  - `dynamic_gaussian.py`: Dynamic Gaussian Splatting implementation.
  - `langsplat.py`: LangSplat method integration.
  - `light_gaussian.py`: Lightweight Gaussian Splatting implementation.
  - `viewer.py`: Viewer utilities for visualizing results.
- **utils/**: Utilities for environment management and running implementations.
  - `base_environment_manager.py`, `conda_manager.py`, `repo_manager.py`, `command_runner.py`, `pointcloud_watcher.py`: Tools for managing environments, repositories, and scripts.
- **config.py**: Project-level configuration.
- **main.py**: Entry point for running and evaluating Gaussian Splatting implementations.
- **Dockerfile**: Container setup for reproducible environments.
- **logo.png**: Project logo.

## Usage

1. **Setup the project**

   ```sh
   git clone https://gitlab.lrz.de/mlmiss25/gscaf/gs_framework.git
   cd gs_framework
   ```

   Use `mount_zip.sh` to mount zipped datasets (e.g., for the Dynamic 3D Gaussian dataset on the TUM CAMP cluster, use `./mount_zip.sh /mnt/datasets/dynamic-gaussians/data.zip /tmp/dynamic-gaussian`) and make sure the dataset paths in `config.py` are correct.

2. **Print Usage**

   ```sh
   python main.py --help
   ```

3. **Example: Run the Original Implementation**

   ```sh
   python main.py original --use_viewer
   ```
   
   Replace the flags as needed. See `python main.py original --help` for all options.

## Usage of the Webviewer

1. 
  ```sh
   git clone https://gitlab.lrz.de/mlmiss25/gscaf/gs_framework.git
   cd gs_framework
   python main.py viewer
   ```

2. Use the install buttons to install and train the different implementations

3. Once the training is done, find the trained Gaussian splatting models in the output folder, and import the .ply, .pth, or .npz files via the import button of the viewer