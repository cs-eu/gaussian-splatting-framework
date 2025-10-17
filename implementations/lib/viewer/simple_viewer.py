import argparse
import logging
import math
import os
import threading
import time

import numpy as np
import torch
import torch.nn.functional as F
import viser
from pathlib import Path
from gsplat.distributed import cli
from gsplat.rendering import rasterization
from plyfile import PlyData
from numpy import nextafter, inf
import threading

from gsplat import export_splats

from nerfview import CameraState, RenderTabState, apply_float_colormap
from gsplat_viewer import GsplatViewer, GsplatRenderTabState

from tqdm import tqdm

logger = logging.getLogger(__name__)

render_state = {}
render_state_lock = threading.Lock()


def load_ply_file(ply_path, device):
    plydata = PlyData.read(ply_path)
    vertices = plydata['vertex']

    # Means (x, y, z)
    means_np = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    
    # Quaternions (rot_0, rot_1, rot_2, rot_3)
    quats_np = np.vstack([vertices['rot_0'], vertices['rot_1'], vertices['rot_2'], vertices['rot_3']]).T
    
    # Scales (scale_0, scale_1, scale_2)
    scales_np = np.vstack([vertices['scale_0'], vertices['scale_1'], vertices['scale_2']]).T
    
    # Opacities
    opacities_np = vertices['opacity']

    # SH0 (f_dc_0, f_dc_1, f_dc_2)
    sh0_np = np.vstack([vertices['f_dc_0'], vertices['f_dc_1'], vertices['f_dc_2']]).T # Shape (N, 3)

    # SHN (f_rest_0, ..., f_rest_K*3-1)
    shN_prop_names = sorted(
        [p.name for p in vertices.properties if p.name.startswith("f_rest_")],
        key=lambda name: int(name.split('_')[-1]) # Sort as f_rest_0, f_rest_1, ...
    )
    
    if shN_prop_names:
        shN_stacked_cols = [vertices[prop_name] for prop_name in shN_prop_names]
        shN_flat_np = np.vstack(shN_stacked_cols).T # Shape (N, K*3)
    else:
        # If no f_rest_ properties, means sh_degree is 0 (only DC component)
        shN_flat_np = np.empty((means_np.shape[0], 0), dtype=np.float32)

    # Convert to torch tensors
    means_t = torch.tensor(means_np, dtype=torch.float32, device=device)
    quats_t = torch.tensor(quats_np, dtype=torch.float32, device=device)
    # Scales are stored directly in PLY (not log-scales)
    scales_t = torch.exp(torch.tensor(scales_np, dtype=torch.float32, device=device)) #torch.tensor(scales_np, dtype=torch.float32, device=device)
    # Opacities are stored directly in PLY (not logits)
    opacities_t = torch.sigmoid(torch.tensor(opacities_np, dtype=torch.float32, device=device)) #torch.tensor(opacities_np, dtype=torch.float32, device=device)

    # Reshape sh0 to (N, 1, 3) as expected by the viewer's concatenation logic
    sh0_t = torch.tensor(sh0_np, dtype=torch.float32, device=device).unsqueeze(1)
    
    if shN_flat_np.shape[1] > 0:
        num_shN_coeffs_flat = shN_flat_np.shape[1]
        if num_shN_coeffs_flat % 3 != 0:
            raise ValueError(
                f"Number of f_rest_ features ({num_shN_coeffs_flat}) in {ply_path} "
                "is not divisible by 3. Each SH coefficient should have 3 color channels."
            )
        K = num_shN_coeffs_flat // 3 # K is the number of higher-order SH bands
        shN_t = torch.tensor(shN_flat_np, dtype=torch.float32, device=device).reshape(means_t.shape[0], 3, K).permute(0, 2, 1)
    else:
        # No higher-order SH coefficients
        shN_t = torch.empty((means_t.shape[0], 0, 3), dtype=torch.float32, device=device)

    # Normalize quaternions (important for rendering stability and consistency)
    quats_t = F.normalize(quats_t, p=2, dim=-1)

    return means_t, quats_t, scales_t, opacities_t, sh0_t, shN_t

def load_npz_file(npz_path, device, timestep=0):
    data = np.load(npz_path, allow_pickle=True)


    num_timesteps = data["means3D"].shape[0]
    t = timestep % num_timesteps

    means_np = data["means3D"][t]               # (N, 3)
    quats_np = data["unnorm_rotations"][t]      # (N, 4)
    scales_np = data["log_scales"]              # (N, 3)
    opacities_np = data["logit_opacities"].squeeze()  # (N,)  
    colors_np = data["rgb_colors"][t]              # (N, 3)

    print("colors_np shape:", colors_np.shape)
    print("First 5 colors:\n", colors_np[:5])

    means_t = torch.tensor(means_np, dtype=torch.float32, device=device)
    quats_t = F.normalize(torch.tensor(quats_np, dtype=torch.float32, device=device), p=2, dim=-1)
    scales_t = torch.exp(torch.tensor(scales_np, dtype=torch.float32, device=device))
    opacities_t = torch.sigmoid(torch.tensor(opacities_np, dtype=torch.float32, device=device))
    colors_t = torch.tensor(colors_np, dtype=torch.float32, device=device).unsqueeze(0)  # [1, N, 3]


    return means_t, quats_t, scales_t, opacities_t, colors_t

def load_gaussians_from_path(path: str, device: torch.device, timestep=0, video_cache=None):
    """Loads gaussians from a .pt or .ply file."""

    if not os.path.isfile(path):
        print(f"File not found: {path}")
        return None

    file_ext = os.path.splitext(path)[1].lower()

    if file_ext == ".pt":
        try:
            ckpt = torch.load(path, map_location=device)["splats"]
            means = ckpt["means"]
            quats = F.normalize(ckpt["quats"], p=2, dim=-1)
            scales = torch.exp(ckpt["scales"])
            opacities = torch.sigmoid(ckpt["opacities"])
            sh0 = ckpt["sh0"]
            shN = ckpt["shN"]
            colors = torch.cat([sh0, shN], dim=-2)
            sh_degree = int(math.sqrt(colors.shape[-2]) - 1)
            return {
                "means": means,
                "quats": quats,
                "scales": scales,
                "opacities": opacities,
                "colors": colors,
                "sh_degree": sh_degree
            }
        except Exception as e:
            print(f"Error loading .pt file: {e}")
            return None
    elif file_ext == ".pth":
        try:
            ckpt = torch.load(path, map_location=device, weights_only=False)
            # Special case for LangSplat checkpoints
            if isinstance(ckpt, tuple) and isinstance(ckpt[0], tuple) and len(ckpt[0]) == 13:
                logger.info("LangSplat checkpoint detected.")
                (
                    active_sh_degree,
                    xyz,
                    features_dc,
                    features_rest,
                    scaling,
                    rotation,
                    opacity,
                    language_features,
                    max_radii2D,
                    xyz_grad_accum,
                    denom,
                    optimizer_state,
                    spatial_lr_scale,
                ) = ckpt[0]

                means = xyz
                quats = F.normalize(rotation, p=2, dim=-1)
                scales = torch.exp(scaling)
                opacities = torch.sigmoid(opacity).squeeze()
                sh0 = features_dc
                shN = features_rest
                colors = torch.cat([sh0, shN], dim=-2)
                sh_degree = int(math.sqrt(colors.shape[-2]) - 1)
                return {
                    "means": means,
                    "quats": quats,
                    "scales": scales,
                    "opacities": opacities,
                    "colors": colors,
                    "sh_degree": sh_degree,
                    "language_features": language_features
                }
        except Exception as e:
            print(f"Error loading .pt/.pth file: {e}")
            return None
    elif file_ext == ".ply":
        try:
            means, quats, scales, opacities, sh0, shN = load_ply_file(path, device)
            colors = torch.cat([sh0, shN], dim=-2)
            sh_degree = int(math.sqrt(colors.shape[-2]) - 1) if colors.shape[-2] > 0 else 0
            return {
                "means": means,
                "quats": quats,
                "scales": scales,
                "opacities": opacities,
                "colors": colors,
                "sh_degree": sh_degree
            }
        except Exception as e:
            print(f"Error loading .ply file: {e}")
            return None
    elif file_ext == ".npz":
        try:
            if video_cache:
                print(f"Using cached video data for {path} at timestep {timestep}")
                means = video_cache["means"][timestep]
                quats = video_cache["quats"][timestep]
                scales = video_cache["scales"][timestep]
                opacities = video_cache["opacities"][timestep]
                colors = video_cache["colors"][timestep]
            else:
                # means, quats, scales, opacities, colors = load_npz_file(path, device, timestep=timestep)

                result = load_npz_file(path, device, timestep=timestep)
                print(f"Loaded result from .npz: type={type(result)}, len={len(result) if hasattr(result, '__len__') else 'N/A'}")
                means, quats, scales, opacities, colors = result

            return {
                "means": means,
                "quats": quats,
                "scales": scales,
                "opacities": opacities,
                "colors": colors
            }
        except Exception as e:
            print(f"Error loading .npz file: {e}")
            return None
    else:
        print(f"Unsupported file format: {path}. Must be .pt, .ply or .pth")
        return None



def main(local_rank: int, world_rank, world_size: int, args):
    torch.manual_seed(42)
    device = torch.device("cuda", local_rank)
    model_state = {}
    model_state2 = {}


    @torch.no_grad()
    def update_model_state(render_tab_state: RenderTabState):
        global render_state
        with render_state_lock:
            render_state.clear()
            do_comparing = render_tab_state.compare and "means" in model_state and "means" in model_state2 and len(model_state["means"]) > 0 and len(model_state2["means"]) > 0
            if do_comparing:
                n_gaussians = min(len(model_state["means"]), len(model_state2["means"]))
                m1, m2 = model_state["means"][:n_gaussians], model_state2["means"][:n_gaussians]
                q1, q2 = model_state["quats"][:n_gaussians], model_state2["quats"][:n_gaussians]
                s1, s2 = model_state["scales"][:n_gaussians], model_state2["scales"][:n_gaussians]
                o1, o2 = model_state["opacities"][:n_gaussians], model_state2["opacities"][:n_gaussians]
                c1, c2 = model_state["colors"][:n_gaussians].flatten(1), model_state2["colors"][:n_gaussians].flatten(1)

                diffs_raw = {
                    "means": torch.norm(m1 - m2, dim=1),
                    "quats": 1.0 - torch.abs(torch.einsum('ij,ij->i', q1, q2)),
                    "scales": torch.norm(s1 - s2, dim=1),
                    "opacities": torch.abs(o1 - o2),
                    "colors": torch.norm(c1 - c2, dim=1),
                }

                diffs_normalized = {}
                epsilon = 1e-9
                for key, raw_diff_tensor in diffs_raw.items():
                    max_diff = torch.max(raw_diff_tensor)
                    if max_diff > epsilon:
                        diffs_normalized[key] = raw_diff_tensor / max_diff
                    else:
                        diffs_normalized[key] = torch.zeros_like(raw_diff_tensor)

                is_different = {
                    "means": diffs_normalized["means"] > render_tab_state.means_threshold,
                    "quats": diffs_normalized["quats"] > render_tab_state.quats_threshold,
                    "scales": diffs_normalized["scales"] > render_tab_state.scales_threshold,
                    "opacities": diffs_normalized["opacities"] > render_tab_state.opacities_threshold,
                    "colors": diffs_normalized["colors"] > render_tab_state.colors_threshold,
                }
                difference_mask = is_different["means"] | is_different["quats"] | is_different["scales"] | is_different["opacities"] | is_different["colors"]

                for key in model_state:
                    if isinstance(model_state2[key], torch.Tensor):
                        render_state[key] = model_state[key][:n_gaussians][difference_mask]
                    else:
                        render_state[key] = model_state[key]

            else:
                render_state = model_state.copy()

            if "scales" in render_state and len(render_state["scales"]) > 0:
                override_tensor = torch.full_like(render_state["scales"], render_tab_state.max_scale)
                render_state["scales"] = torch.minimum(render_state["scales"], override_tensor)


    def load_new_model(path: str, render_tab_state: RenderTabState = None, timestep=0, video_cache=None):
        global render_state
        print(f"Attempting to load model from: {path}")
        logger.info(f"Attempting to load model from: {path}")

        if os.path.isdir(path):
            print("Path is a directory, loading up to 2 .pt or .ply files...")
            try:
                files = sorted([f for f in os.listdir(path) if f.lower().endswith(('.pt', '.ply'))])
                if len(files) == 0:
                    print("No .pt or .ply files found in the directory.")
                if len(files) >= 1:
                    model_state.clear()
                    model_state2.clear()
                    filepath1 = os.path.join(path, files[0])
                    print(f"Loading first model: {files[0]}")
                    loaded_data = load_gaussians_from_path(filepath1, device)
                    if loaded_data:
                        for key, value in loaded_data.items():
                            model_state[key] = value
                        if "sh_degree" not in model_state:
                            model_state["sh_degree"] = int(math.sqrt(model_state["colors"].shape[-2]) - 1) if model_state["colors"].shape[-2] > 0 else 0
                    else:
                        print("Model loading failed.")
                if len(files) >= 2:
                    filepath2 = os.path.join(path, files[1])
                    print(f"Loading second model: {files[1]}")
                    loaded_data2 = load_gaussians_from_path(filepath2, device)
                    if loaded_data2:
                        for key, value in loaded_data2.items():
                            model_state2[key] = value
                        if "sh_degree" not in model_state2:
                            model_state2["sh_degree"] = int(math.sqrt(model_state2["colors"].shape[-2]) - 1) if model_state2["colors"].shape[-2] > 0 else 0
                    else:
                        print("Model2 loading failed.")
            except Exception as e:
                print(f"Error while loading from directory: {e}")

        elif os.path.isfile(path):
            print("Path is a file, loading single model...")
            loaded_data = load_gaussians_from_path(path, device, timestep, video_cache)
            if loaded_data:
                model_state.clear()
                model_state2.clear()
                for key, value in loaded_data.items():
                    model_state[key] = value
                if ".npz" in path:
                    model_state["sh_degree"] = None
                else:
                    if "sh_degree" not in model_state:
                        model_state["sh_degree"] = int(math.sqrt(model_state["colors"].shape[-2]) - 1) if model_state["colors"].shape[-2] > 0 else 0
            else:
                print("Model loading failed. Keeping current model.")

        else:
            print(f"Path not found: {path}")

        if "means" not in model_state or len(model_state["means"]) == 0:
            print("Loading failed or no model provided. Initializing empty scene.")
            model_state["means"] = torch.empty((0, 3), dtype=torch.float32, device=device)
            model_state["quats"] = torch.empty((0, 4), dtype=torch.float32, device=device)
            model_state["scales"] = torch.empty((0, 3), dtype=torch.float32, device=device)
            model_state["opacities"] = torch.empty((0,), dtype=torch.float32, device=device)
            model_state["colors"] = torch.empty((0, 1, 3), dtype=torch.float32, device=device)
            model_state["sh_degree"] = 0
        
        render_state = model_state.copy()
        if render_tab_state:
            update_model_state(render_tab_state)
        print(f"Successfully loaded model with {len(model_state['means'])} Gaussians.")


    def export_model(path: str):
        print(f"Attempting to export model to: {path}")
        dir_name = os.path.dirname(path)
        if not os.path.exists(dir_name):
            print(f"Invalid path: Directory '{dir_name}' does not exist. Please create it first.")
            return

        means_to_save = model_state["means"]
        quats_to_save = model_state["quats"]
        scales_to_save = torch.log(torch.clamp(model_state["scales"], min=nextafter(0, inf)))
        opacities_to_save = torch.clamp(torch.logit(torch.clamp(model_state["opacities"], min=nextafter(0, inf))), max = nextafter(37, 0))
        num_sh_features = model_state["colors"].shape[-2]
        sh0_to_save = model_state["colors"][:, :1, :]
        shN_to_save = model_state["colors"][:, 1:, :]

        file_ext = os.path.splitext(path)[1].lower()
        if file_ext == ".pt":
            try:
                splats_dict = {
                    "means": means_to_save,
                    "quats": quats_to_save,
                    "scales": scales_to_save,
                    "opacities": opacities_to_save,
                    "sh0": sh0_to_save,
                    "shN": shN_to_save
                }
                data_to_save = {"step": -1, "splats": splats_dict}
                torch.save(data_to_save, path)
                print(f"Successfully exported model to {path}")
            except Exception as e:
                print(f"Error exporting .pt file: {e}")
        elif file_ext == ".ply":
            try:
                export_splats(
                    means=means_to_save,
                    scales=scales_to_save,
                    quats=quats_to_save,
                    opacities=opacities_to_save,
                    sh0=sh0_to_save,
                    shN=shN_to_save,
                    format="ply",
                    save_to=path,
                )
                print(f"Successfully exported model to {path}")
            except Exception as e:
                print(f"Error exporting .ply file: {e}")
        else:
            print(f"Unsupported export format: {path}. Must be .pt or .ply")

    def load_video(npz_path):
        data = np.load(npz_path)
        num_timesteps = data["means3D"].shape[0]
        data.close()

        means_list, quats_list, scales_list, opacities_list, colors_list = [], [], [], [], []

        for t in tqdm(range(num_timesteps), desc="Loading timesteps"):
            means_t, quats_t, scales_t, opacities_t, colors_t = load_npz_file(npz_path, device, timestep=t)

            means_list.append(means_t)
            quats_list.append(quats_t)
            scales_list.append(scales_t)
            opacities_list.append(opacities_t)
            colors_list.append(colors_t)

        means = torch.stack(means_list, dim=0)
        quats = torch.stack(quats_list, dim=0)
        scales = torch.stack(scales_list, dim=0)
        opacities = torch.stack(opacities_list, dim=0)
        colors = torch.stack(colors_list, dim=0)

        video_cache = {
            "means": means.to(device),
            "quats": quats.to(device),
            "scales": scales.to(device),
            "opacities": opacities.to(device),
            "colors": colors.to(device)
        }

        return video_cache

    if args.ckpt is None:
        load_new_model("")
    else:
        initial_path = args.ckpt[0]
        load_new_model(initial_path)

    if "language_features" in model_state:
        import sys
        module_path = os.environ["LANGSPLAT_DIR"]
        if module_path not in sys.path:
            sys.path.append(module_path)
        from eval.openclip_encoder import OpenCLIPNetwork 
        from autoencoder.model import Autoencoder 

        logger.info("Loading language features")
        language_features = model_state["language_features"]
        # Load CLIP and Autoencoder once
        encoder_hidden_dims = [256, 128, 64, 32, 3]
        decoder_hidden_dims = [16, 32, 64, 128, 256, 256, 512]
        ae_ckpt_path = args.ae_ckpt
        clip_model = OpenCLIPNetwork(device)
        checkpoint = torch.load(ae_ckpt_path, map_location=device)
        model = Autoencoder(encoder_hidden_dims, decoder_hidden_dims).to(device)
        model.load_state_dict(checkpoint)
        model.eval()
        # If language_features are compressed, decode them to CLIP space
        restored_feat = model.decode(language_features)  # [N, 512]

    # register and open viewer
    @torch.no_grad()
    def viewer_render_fn(camera_state: CameraState, render_tab_state: RenderTabState):
        global render_state
        assert isinstance(render_tab_state, GsplatRenderTabState)
        if render_tab_state.preview_render:
            width = render_tab_state.render_width
            height = render_tab_state.render_height
        else:
            width = render_tab_state.viewer_width
            height = render_tab_state.viewer_height
        with render_state_lock:
            if "means" not in render_state or len(render_state["means"]) == 0:
                render_tab_state.total_gs_count = 0
                render_tab_state.rendered_gs_count = 0
                return np.zeros((height, width, 3), dtype=np.uint8)
            means_to_render = render_state["means"]
            quats_to_render = render_state["quats"]
            scales_to_render = render_state["scales"]
            opacities_to_render = render_state["opacities"]
            colors_to_render = render_state["colors"]
            if "sh_degree" not in render_state:
                sh_degree_to_render = int(math.sqrt(colors_to_render.shape[-2]) - 1)
            else:
                sh_degree_to_render = render_state["sh_degree"]
        c2w = camera_state.c2w
        K = camera_state.get_K((width, height))
        c2w = torch.from_numpy(c2w).float().to(device)
        K = torch.from_numpy(K).float().to(device)
        viewmat = c2w.inverse()

        RENDER_MODE_MAP = {
            "rgb": "RGB",
            "depth(accumulated)": "D",
            "depth(expected)": "ED",
            "alpha": "RGB",
            "number of gaussians": "NG",
            "average opacity": "AO",
            "average scale": "AS",
            "language similarity": "RGB",
            "3D language similarity": "RGB",
            "3D language features": "RGB",
        }

        if render_tab_state.render_mode == "language similarity":
            # Use the text prompt from the UI
            text_prompt = render_tab_state.langsplat_text

            # Update the positive phrase for CLIP
            clip_model.set_positives([text_prompt])

            with torch.no_grad():
                # Compute relevancy (similarity) to the text prompt
                relevancy = clip_model.get_relevancy(restored_feat, positive_id=0)[:, 0]  # [N]
                # Normalize relevancy to [0, 1] for heatmap coloring
                normalized_relevancy = (relevancy - relevancy.min()) / (relevancy.max() - relevancy.min() + 1e-8)
                # Expand to shape [N, 1, 3] for RGB heatmap (e.g., red = high similarity)
                colors_to_render = normalized_relevancy[:, None, None].expand(-1, 1, 3)
                sh_degree_to_render = 0  # No SH
        elif render_tab_state.render_mode == "3D language similarity":
            text_prompt = render_tab_state.langsplat_text
            encoded_text = clip_model.get_text_embedding([text_prompt])
            encoded_text = encoded_text.to(dtype=next(model.parameters()).dtype)
            text_features = model.encode(encoded_text)
            # Compute cosine similarity between text_features and language_features
            # text_features: [1, 3], language_features: [N, 3]
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            lang_features = language_features / (language_features.norm(dim=-1, keepdim=True) + 1e-8)
            similarity = torch.matmul(lang_features, text_features.t()).squeeze(-1)  # [N]
            # Normalize similarity to [0, 1] for heatmap coloring
            normalized_similarity = (similarity - similarity.min()) / (similarity.max() - similarity.min() + 1e-8)
            # Expand to shape [N, 1, 3] for RGB heatmap (e.g., red = high similarity)
            colors_to_render = normalized_similarity[:, None, None].expand(-1, 1, 3)
            sh_degree_to_render = 0  # No SH
        elif render_tab_state.render_mode == "3D language features":
            colors_to_render = language_features[:, None, :].expand(-1, 1, 3)
            sh_degree_to_render = 0  # No SH

        render_colors, render_alphas, info = rasterization(
            means_to_render,  # [N, 3]
            quats_to_render,  # [N, 4]
            scales_to_render,  # [N, 3]
            opacities_to_render,  # [N]
            colors_to_render,  # [N, S, 3]
            viewmat[None],  # [1, 4, 4]
            K[None],  # [1, 3, 3]
            width,
            height,
            sh_degree=(
                min(render_tab_state.max_sh_degree, sh_degree_to_render)
                if sh_degree_to_render is not None
                else None
            ),
            near_plane=render_tab_state.near_plane,
            far_plane=render_tab_state.far_plane,
            radius_clip=render_tab_state.radius_clip,
            eps2d=render_tab_state.eps2d,
            backgrounds=torch.tensor([render_tab_state.backgrounds], device=device)
            / 255.0,
            render_mode=RENDER_MODE_MAP[render_tab_state.render_mode],
            rasterize_mode=render_tab_state.rasterize_mode,
            camera_model=render_tab_state.camera_model,
            packed=False,
            with_ut=args.with_ut,
            with_eval3d=args.with_eval3d,
        )
        render_tab_state.total_gs_count = len(means_to_render)
        render_tab_state.rendered_gs_count = (info["radii"] > 0).all(-1).sum().item()

        if render_tab_state.render_mode in ["rgb", "3D language features"]:
            # colors represented with sh are not guranteed to be in [0, 1]
            render_colors = render_colors[0, ..., 0:3].clamp(0, 1)
            renders = render_colors.cpu().numpy()
        elif render_tab_state.render_mode in ["depth(accumulated)", "depth(expected)"]:
            # normalize depth to [0, 1]
            depth = render_colors[0, ..., 0:1]
            if render_tab_state.normalize_nearfar:
                near_plane = render_tab_state.near_plane
                far_plane = render_tab_state.far_plane
            else:
                near_plane = depth.min()
                far_plane = depth.max()
            depth_norm = (depth - near_plane) / (far_plane - near_plane + 1e-10)
            depth_norm = torch.clip(depth_norm, 0, 1)
            if render_tab_state.inverse:
                depth_norm = 1 - depth_norm
            renders = (
                apply_float_colormap(depth_norm, render_tab_state.colormap)
                .cpu()
                .numpy()
            )
        elif render_tab_state.render_mode == "alpha":
            alpha = render_alphas[0, ..., 0:1]
            renders = (
                apply_float_colormap(alpha, render_tab_state.colormap).cpu().numpy()
            )
        elif render_tab_state.render_mode in ["number of gaussians", "average opacity", "average scale"]:
            accumulated_values = render_colors[0, ..., 0:1]
            if render_tab_state.render_mode in ["average opacity", "average scale"]:
                accumulated_weights = render_alphas[0, ..., 0:1]
                heatmap_values = accumulated_values / accumulated_weights.clamp(min=1e-10)
            else:
                heatmap_values = accumulated_values

            if render_tab_state.normalize_nearfar:
                min_val = heatmap_values.min()
                max_val = heatmap_values.max()
                if (max_val - min_val) > 1e-9:
                    normalized_values = (heatmap_values - min_val) / (max_val - min_val)
                else:
                    normalized_values = torch.zeros_like(heatmap_values)
            else:
                heatmap_min = render_tab_state.heatmap_min
                heatmap_max = render_tab_state.heatmap_max
                if (heatmap_max - heatmap_min) < 1e-9:
                    normalized_values = torch.zeros_like(heatmap_values)
                else:
                    normalized_values = (heatmap_values - heatmap_min) / (heatmap_max - heatmap_min)

            normalized_values = torch.clip(normalized_values, 0, 1)

            if render_tab_state.inverse:
                normalized_values = 1 - normalized_values

            renders = (apply_float_colormap(normalized_values, render_tab_state.colormap).cpu().numpy())
        elif render_tab_state.render_mode in ["language similarity", "3D language similarity"]:
            render_colors = render_colors[0, ..., 0:3].clamp(0, 1)
            # Apply top 10% filter to the heatmap
            flat = render_colors.reshape(-1, 3)
            intensity = flat.norm(dim=1)
            threshold = torch.quantile(intensity, 0.9)
            mask = intensity >= threshold
            filtered = torch.zeros_like(flat)
            # Set filtered-away Gaussians to grey (0.5, 0.5, 0.5)
            filtered[mask] = flat[mask]
            # Normalize only the non-grey (top 10%) values to [0, 1]
            if mask.any():
                vals = filtered[mask]
                min_v = vals.min()
                max_v = vals.max()
                norm_vals = (vals - min_v) / (max_v - min_v + 1e-8)
                filtered[mask] = norm_vals
            filtered = filtered.reshape(render_colors.shape)
            renders = apply_float_colormap(filtered, render_tab_state.colormap)
            # add the render_colors to the renders
            # Subtract grey (0.5, 0.5, 0.5) from the render_colors before adding
            grey = torch.tensor([0.5, 0.5, 0.5], device=render_colors.device).view(1, 1, 3)
            renders = renders + (render_colors - grey)
            renders = renders.cpu().numpy()

        elif render_tab_state.render_mode in ["number of gaussians", "average opacity", "average scale"]:
            accumulated_values = render_colors[0, ..., 0:1]
            if render_tab_state.render_mode in ["average opacity", "average scale"]:
                accumulated_weights = render_alphas[0, ..., 0:1]
                heatmap_values = accumulated_values / accumulated_weights.clamp(min=1e-10)
            else:
                heatmap_values = accumulated_values

            if render_tab_state.normalize_nearfar:
                min_val = heatmap_values.min()
                max_val = heatmap_values.max()
                if (max_val - min_val) > 1e-9:
                    normalized_values = (heatmap_values - min_val) / (max_val - min_val)
                else:
                    normalized_values = torch.zeros_like(heatmap_values)
            else:
                heatmap_min = render_tab_state.heatmap_min
                heatmap_max = render_tab_state.heatmap_max
                if (heatmap_max - heatmap_min) < 1e-9:
                    normalized_values = torch.zeros_like(heatmap_values)
                else:
                    normalized_values = (heatmap_values - heatmap_min) / (heatmap_max - heatmap_min)

            normalized_values = torch.clip(normalized_values, 0, 1)

            if render_tab_state.inverse:
                normalized_values = 1 - normalized_values

            renders = (apply_float_colormap(normalized_values, render_tab_state.colormap).cpu().numpy())
        return renders

    server = viser.ViserServer(port=args.port, verbose=False)
    #url = server.request_share_url(verbose=True)
    #logger.info(f"Viewer URL: {url}")
    _ = GsplatViewer(
        server=server,
        render_fn=viewer_render_fn,
        output_dir=Path(args.output_dir),
        mode="rendering",
        load_callback=load_new_model,
        live_update_path=args.ckpt[0] if args.live_update else None,
        #animation_cache_callback=load_npz_cache,
        load_video_callback=load_video,
        export_callback=export_model,
        model_state_callback = update_model_state,
    )
    logger.info("Viewer running... Ctrl+C to exit.")
    time.sleep(100000)


if __name__ == "__main__":
    """
    # Use single GPU to view the scene
    CUDA_VISIBLE_DEVICES=9 python -m simple_viewer \
        --ckpt results/garden/ckpts/ckpt_6999_rank0.pt \
        --output_dir results/garden/ \
        --port 8082

    CUDA_VISIBLE_DEVICES=9 python -m simple_viewer \
        --output_dir results/garden/ \
        --port 8082
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir", type=str, default="results/", help="where to dump outputs"
    )
    parser.add_argument(
        "--scene_grid", type=int, default=1, help="repeat the scene into a grid of NxN"
    )
    parser.add_argument(
        "--ckpt", type=str, nargs="+", default=None, help="path to the .pt or .pth file of the scene"
    )
    parser.add_argument(
        "--ae_ckpt", type=str, default=None, help="path to the autoencoder .pth file"
    )
    parser.add_argument(
        "--port", type=int, default=8080, help="port for the viewer server"
    )
    parser.add_argument(
        "--with_ut", action="store_true", help="use uncentered transform"
    )
    parser.add_argument("--with_eval3d", action="store_true", help="use eval 3D")
    parser.add_argument("--live_update", action="store_true", help="enable live update of .ply files")
    args = parser.parse_args()
    assert args.scene_grid % 2 == 1, "scene_grid must be odd"

    cli(main, args, verbose=True)
