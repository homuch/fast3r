# IMPORTANT: Model Checkpoint Requirement
# This script requires a pre-trained Fast3R model checkpoint.
# Please download an appropriate checkpoint and update the 'model_checkpoint_path'
# in the DEFAULT_ARGS configuration below.
# For example, the 'super_long_training_5175604' checkpoint mentioned in
# the project resources or notebooks might be suitable.
# Ensure the path points to the directory containing the '.hydra' and 'checkpoints' folders,
# or to a specific .pth/.ckpt file if using a raw model dump.

import os
import glob
import time
import shutil
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import torch
import numpy as np
import hydra
from omegaconf import OmegaConf, DictConfig
import trimesh
import open3d as o3d # For voxel downsampling in export_combined_ply

# Fast3R specific imports
import rootutils
try:
    rootutils.setup_root(search_from=__file__, indicator=".project-root", pythonpath=True)
except Exception as e:
    print(f"Failed to setup rootutils: {e}. Assuming paths are already set up.")
    # Fallback for environments where .project-root might not be in the expected place
    # or if __file__ is not defined (e.g. interactive execution)
    try:
        if '__file__' not in globals(): __file__ = os.getcwd()
        project_root = Path(__file__).resolve().parent
        while not (project_root / ".project-root").exists() and project_root != project_root.parent:
            project_root = project_root.parent
        if (project_root / ".project-root").exists():
            os.environ["PYTHONPATH"] = str(project_root) + os.pathsep + os.environ.get("PYTHONPATH", "")
        else: # Final fallback
            print("Warning: .project-root not found.PYTHONPATH modification might be incomplete.")
    except Exception as e_fallback:
        print(f"Fallback rootutils setup failed: {e_fallback}")


from fast3r.models.multiview_dust3r_module import MultiViewDUSt3RLitModule
from fast3r.dust3r.inference_multiview import inference as dust3r_inference
from fast3r.dust3r.model import FlashDUSt3R
from fast3r.dust3r.utils.image import load_images, rgb
from fast3r.dust3r.viz import pts3d_to_trimesh, cat_meshes # Used by export_combined_ply
from fast3r.dust3r.cloud_opt.init_im_poses import fast_pnp # For camera pose estimation if needed later

# (Copied and adapted from notebooks/demo_multiview.ipynb)
def export_combined_ply(
    preds: List[Dict], 
    views: List[Dict], 
    export_ply_path: str, 
    pts3d_key_to_visualize: str = "pts3d_local_aligned_to_global",
    conf_key_to_visualize: str = "conf_local",
    min_conf_thr_percentile: float = 0.0, 
    flip_axes: bool = False, 
    max_num_points: Optional[int] = None, 
    sampling_strategy: str = 'uniform'
) -> Tuple[np.ndarray, np.ndarray]:
    all_points = []
    all_colors = []

    for i, pred in enumerate(preds):
        if pts3d_key_to_visualize not in pred or pred[pts3d_key_to_visualize] is None:
            logging.warning(f"View {i}: Key '{pts3d_key_to_visualize}' not found in prediction. Skipping.")
            continue
        if conf_key_to_visualize not in pred or pred[conf_key_to_visualize] is None:
            logging.warning(f"View {i}: Key '{conf_key_to_visualize}' not found in prediction. Using zero confidence.")
            conf = np.zeros_like(pred[pts3d_key_to_visualize].cpu().numpy().squeeze()[..., 0]) # dummy conf
        else:
            conf = pred[conf_key_to_visualize].cpu().numpy().squeeze()

        pts3d = pred[pts3d_key_to_visualize].cpu().numpy().squeeze()
        img_rgb = views[i]['img'].cpu().numpy().squeeze().transpose(1, 2, 0)  # Shape: (H, W, 3)
        
        conf_thr = np.percentile(conf, min_conf_thr_percentile)

        x, y, z = pts3d[..., 0].flatten(), pts3d[..., 1].flatten(), pts3d[..., 2].flatten()
        r, g, b = img_rgb[..., 0].flatten(), img_rgb[..., 1].flatten(), img_rgb[..., 2].flatten()
        conf_flat = conf.flatten()

        mask = conf_flat >= conf_thr # Use >= to include points at the threshold
        
        if not np.any(mask): # if no points meet confidence, skip this view
            logging.warning(f"View {i}: No points passed confidence threshold {conf_thr} (percentile {min_conf_thr_percentile}). Skipping.")
            continue

        x, y, z = x[mask], y[mask], z[mask]
        r_vals, g_vals, b_vals = r[mask], g[mask], b[mask]

        r_vals = ((r_vals + 1) * 127.5).astype(np.uint8).clip(0, 255)
        g_vals = ((g_vals + 1) * 127.5).astype(np.uint8).clip(0, 255)
        b_vals = ((b_vals + 1) * 127.5).astype(np.uint8).clip(0, 255)

        points = np.vstack([x, y, z]).T
        colors = np.vstack([r_vals, g_vals, b_vals]).T

        if flip_axes:
            points = points[:, [0, 2, 1]]
            points[:, 2] = -points[:, 2]

        all_points.append(points)
        all_colors.append(colors)

    if not all_points:
        logging.warning("No points collected from any view after filtering. Cannot create PLY.")
        # Create an empty PLY file or handle as an error
        if export_ply_path:
            Path(export_ply_path).touch() # Create an empty file
        return np.array([]), np.array([])


    all_points = np.vstack(all_points)
    all_colors = np.vstack(all_colors)

    if max_num_points is not None and len(all_points) > max_num_points:
        if sampling_strategy == 'uniform':
            indices = np.random.choice(len(all_points), size=max_num_points, replace=False)
            all_points = all_points[indices]
            all_colors = all_colors[indices]
        elif sampling_strategy in ['voxel', 'farthest_point']:
            try:
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(all_points)
                pcd.colors = o3d.utility.Vector3dVector(all_colors.astype(np.float64) / 255.0)
                
                if sampling_strategy == 'voxel':
                    # Heuristic for voxel size
                    bounding_box = pcd.get_axis_aligned_bounding_box()
                    extent = bounding_box.get_extent()
                    volume = extent[0] * extent[1] * extent[2]
                    if volume <= 0: volume = 1.0 # handle empty or flat point clouds
                    voxel_size = (volume / max_num_points) ** (1/3)
                    if voxel_size <= 1e-6: voxel_size = 0.01 # Minimum voxel size
                    down_pcd = pcd.voxel_down_sample(voxel_size)
                else: # farthest_point
                    down_pcd = pcd.farthest_point_down_sample(max_num_points)
                
                all_points = np.asarray(down_pcd.points)
                all_colors = (np.asarray(down_pcd.colors) * 255.0).astype(np.uint8)
            except Exception as e:
                logging.error(f"Open3D based sampling failed: {e}. Falling back to uniform sampling.")
                indices = np.random.choice(len(all_points), size=max_num_points, replace=False)
                all_points = all_points[indices]
                all_colors = all_colors[indices]
            else:
                raise ValueError(f"Unsupported sampling strategy: {sampling_strategy}")

    if export_ply_path:
        try:
            point_cloud = trimesh.PointCloud(vertices=all_points, colors=all_colors)
            # Ensure parent directory exists
            Path(export_ply_path).parent.mkdir(parents=True, exist_ok=True)
            point_cloud.export(export_ply_path)
            logging.info(f"Saved point cloud to {export_ply_path}")
        except Exception as e:
            logging.error(f"Failed to export PLY file: {e}")

    return all_points, all_colors

def load_model(checkpoint_path: str, device: torch.device) -> Tuple[Optional[MultiViewDUSt3RLitModule], Optional[torch.nn.Module]]:
    # This function adapts the logic from the notebook for loading either
    # a raw FlashDUSt3R model or a Lightning checkpoint.
    # For this project, we'll focus on the Lightning checkpoint loading,
    # as it's more aligned with the training framework of fast3r.

    # Assuming checkpoint_path points to a lightning checkpoint dir like in the notebook:
    # checkpoint_dir = "/path/to/checkpoint_root/dust3r/fast3r/logs/super_long_training/runs/super_long_training_5175604"
    
    checkpoint_dir = Path(checkpoint_path)
    hydra_cfg_path = checkpoint_dir / ".hydra" / "config.yaml"

    if not hydra_cfg_path.exists():
        logging.error(f"Hydra config not found at {hydra_cfg_path}")
        # Fallback: Try to load as a direct .pth or .ckpt file (FlashDUSt3R or simple Lightning)
        if Path(checkpoint_path).suffix in ['.pth', '.ckpt'] and Path(checkpoint_path).is_file():
            logging.info(f"Attempting to load {checkpoint_path} as a raw model file.")
            try:
                model = FlashDUSt3R.from_pretrained(checkpoint_path).to(device)
                # Wrap it in a dummy LitModule or return model directly
                # For simplicity, we'll assume direct model usage if not a full hydra checkpoint
                logging.warning("Loaded as raw FlashDUSt3R model. Full LitModule functionality (like alignment) might not be available directly.")
                return None, model # No lit_module, just the network
            except Exception as e:
                logging.error(f"Failed to load {checkpoint_path} as raw model: {e}")
                return None, None
        return None, None

    logging.info(f"Loading model from checkpoint directory: {checkpoint_dir}")
    cfg = OmegaConf.load(hydra_cfg_path)

    # Adapting config replacements from notebook
    def replace_dust3r_in_config(cfg_node):
        for key, value in cfg_node.items():
            if isinstance(value, DictConfig):
                replace_dust3r_in_config(value)
            elif isinstance(value, str):
                if "dust3r." in value and "fast3r.dust3r." not in value:
                    cfg_node[key] = value.replace("dust3r.", "fast3r.dust3r.")
        return cfg_node

    def replace_src_in_config(cfg_node):
        for key, value in cfg_node.items():
            if isinstance(value, DictConfig):
                replace_src_in_config(value)
            elif isinstance(value, str) and "src." in value:
                cfg_node[key] = value.replace("src.", "fast3r.")
        return cfg_node

    cfg.model.net = replace_dust3r_in_config(cfg.model.net)
    cfg.model = replace_src_in_config(cfg.model)
    
    # Default settings from notebook if keys exist
    if "encoder_args" in cfg.model.net:
        cfg.model.net.encoder_args.patch_embed_cls = "PatchEmbedDust3R"
        if "head_args" in cfg.model.net and cfg.model.net.head_args is not None: # Ensure head_args exists and is not None
             cfg.model.net.head_args.landscape_only = False
        elif "head_args" not in cfg.model.net or cfg.model.net.head_args is None: # if head_args is missing or None
             OmegaConf.update(cfg.model.net, "head_args", {"landscape_only": False}, merge=True)
    else:
        cfg.model.net.patch_embed_cls = "PatchEmbedDust3R"
        cfg.model.net.landscape_only = False
    
    if cfg.model.net.decoder_args is not None: # Ensure decoder_args exists
        cfg.model.net.decoder_args.random_image_idx_embedding = True
        cfg.model.net.decoder_args.attn_bias_for_inference_enabled = False
    else: # If decoder_args is missing or None
        OmegaConf.update(cfg.model.net, "decoder_args", {
            "random_image_idx_embedding": True, 
            "attn_bias_for_inference_enabled": False
        }, merge=True)


    lit_module = hydra.utils.instantiate(cfg.model, train_criterion=None, validation_criterion=None)
    
    actual_ckpt_path = checkpoint_dir / "checkpoints" / "last.ckpt"
    if os.path.isdir(actual_ckpt_path): # DeepSpeed checkpoint
        logging.info("DeepSpeed checkpoint detected. Attempting conversion if necessary.")
        aggregated_ckpt_path = checkpoint_dir / "checkpoints" / "last_aggregated.ckpt"
        if not aggregated_ckpt_path.exists():
            try:
                from lightning.pytorch.utilities.deepspeed import convert_zero_checkpoint_to_fp32_state_dict
                convert_zero_checkpoint_to_fp32_state_dict(
                    checkpoint_dir=str(actual_ckpt_path), 
                    output_file=str(aggregated_ckpt_path)
                )
                actual_ckpt_path = aggregated_ckpt_path
            except Exception as e:
                logging.error(f"Failed to convert DeepSpeed checkpoint: {e}")
                return None, None
        else:
            actual_ckpt_path = aggregated_ckpt_path
    
    if not actual_ckpt_path.exists():
        logging.error(f"Checkpoint file not found at {actual_ckpt_path}")
        # Try to find any .ckpt in the checkpoints folder
        ckpts = list((checkpoint_dir / "checkpoints").glob("*.ckpt"))
        if ckpts:
            actual_ckpt_path = ckpts[0] # Take the first one found
            logging.warning(f"Using {actual_ckpt_path} as fallback.")
        else:
            return None, None


    try:
        lit_module_loaded = MultiViewDUSt3RLitModule.load_from_checkpoint(
            checkpoint_path=str(actual_ckpt_path),
            net=lit_module.net, # Pass the instantiated net
            train_criterion=None, # As per notebook
            validation_criterion=None, # As per notebook
            strict=False # Allow some mismatches if necessary, e.g. if criterion parts are missing
        )
        lit_module_loaded.eval()
        model = lit_module_loaded.net.to(device)
        logging.info("Model loaded successfully from Lightning checkpoint.")
        return lit_module_loaded, model
    except Exception as e:
        logging.error(f"Failed to load model from checkpoint {actual_ckpt_path}: {e}")
        # Try loading just the network if LitModule fails
        try:
            checkpoint = torch.load(actual_ckpt_path, map_location=device)
            # Assuming the network is stored under 'state_dict' and needs prefix 'net.'
            state_dict = {k.replace("net.", ""): v for k, v in checkpoint['state_dict'].items() if k.startswith("net.")}
            lit_module.net.load_state_dict(state_dict)
            lit_module.net.to(device).eval()
            logging.warning("Loaded model network state_dict directly into instantiated net. LitModule instance might not be fully functional for alignment.")
            return lit_module, lit_module.net # Return the instantiated lit_module, but its state is not from load_from_checkpoint
        except Exception as e_fallback:
            logging.error(f"Fallback loading of network state_dict also failed: {e_fallback}")
            return None, None

def process_scene(scene_name: str, seq_id: str, image_files: List[str], model: torch.nn.Module, lit_module: Optional[MultiViewDUSt3RLitModule], device: torch.device, output_dir: str, args: DictConfig):
    logging.info(f"Processing {scene_name} - {seq_id} with {len(image_files)} images.")
    
    output_ply_filename = f"{scene_name}-{seq_id}.ply"
    output_ply_path = Path(output_dir) / output_ply_filename

    # 1. Load images
    # Adapted from get_reconstructed_scene in notebook
    # Note: The notebook's load_images takes 'size', 'rotate_clockwise_90', 'crop_to_landscape'
    # These could be part of 'args'
    loaded_views = load_images(image_files, size=args.image_size, verbose=not args.silent)

    # 2. Run inference
    # Adapted from get_reconstructed_scene in notebook
    # Note: dust3r_inference takes 'model', 'device', 'dtype', 'verbose', 'profiling'
    start_time = time.time()
    with torch.no_grad(): # Ensure no gradients are computed during inference
         preds_output = dust3r_inference(loaded_views, model, device, dtype=torch.float32, verbose=not args.silent, profiling=args.profiling)
    end_time = time.time()
    logging.info(f"Inference for {scene_name}-{seq_id} took {end_time - start_time:.2f} seconds.")

    # 3. Align local pts3d to global (if lit_module is available)
    if lit_module and 'preds' in preds_output and preds_output['preds'] and 'views' in preds_output and preds_output['views']:
        # The notebook uses min_conf_thr_percentile=85 for alignment
        # This could also be a parameter in 'args'
        try:
            # Ensure preds_output['preds'] and preds_output['views'] are lists of dicts as expected
            # The inference function already returns a dict with 'preds' and 'views' as top-level keys
            # and their values are lists of dicts.
            lit_module.align_local_pts3d_to_global(
                preds=preds_output['preds'], 
                views=preds_output['views'], 
                min_conf_thr_percentile=args.align_conf_percentile
            )
            logging.info("Local to global alignment complete.")
        except Exception as e:
            logging.error(f"Error during local to global alignment: {e}")
            # Decide if to proceed without alignment or skip saving
    elif not lit_module:
        logging.warning("LitModule not available, skipping local to global alignment. Output PLY might not be globally consistent.")
    else:
        logging.warning("Skipping local to global alignment due to missing 'preds' or 'views' in inference output.")


    # 4. Export PLY
    # The notebook uses specific keys: "pts3d_local_aligned_to_global", "conf_local"
    # It also has min_conf_thr_percentile, flip_axes, max_num_points, sampling_strategy
    if 'preds' in preds_output and preds_output['preds'] and 'views' in preds_output and preds_output['views']:
        export_combined_ply(
            preds=preds_output['preds'],
            views=preds_output['views'],
            export_ply_path=str(output_ply_path),
            pts3d_key_to_visualize=args.ply_pts_key,
            conf_key_to_visualize=args.ply_conf_key,
            min_conf_thr_percentile=args.ply_conf_percentile,
            flip_axes=args.ply_flip_axes,
            max_num_points=args.ply_max_points,
            sampling_strategy=args.ply_sampling_strategy
        )
    else:
        logging.error(f"Cannot export PLY for {scene_name}-{seq_id} due to missing 'preds' or 'views' in inference output.")

# Placeholder for configuration and main loop (will be detailed in next steps)
DEFAULT_ARGS = OmegaConf.create({
    "data_base_path": "./data/7scenes/",  # USER ACTION: Update this path to your 7Scenes dataset location
    "model_checkpoint_path": "./models/fast3r_super_long_training_5175604/",  # USER ACTION: Update this path to your model checkpoint directory
    "output_dir": "test",
    "sequences": [
        {"scene_name": "chess", "scene_id": "seq-03"},
        {"scene_name": "fire", "scene_id": "seq-03"},
        {"scene_name": "heads", "scene_id": "seq-01"},
        {"scene_name": "office", "scene_id": "seq-02"},
        {"scene_name": "office", "scene_id": "seq-06"},
        {"scene_name": "office", "scene_id": "seq-07"},
        {"scene_name": "office", "scene_id": "seq-09"},
        {"scene_name": "pumpkin", "scene_id": "seq-01"},
        {"scene_name": "redkitchen", "scene_id": "seq-03"},
        {"scene_name": "redkitchen", "scene_id": "seq-04"},
        {"scene_name": "redkitchen", "scene_id": "seq-06"},
        {"scene_name": "redkitchen", "scene_id": "seq-12"},
        {"scene_name": "redkitchen", "scene_id": "seq-14"},
        {"scene_name": "stairs", "scene_id": "seq-01"},
    ],
    "image_glob_pattern": "frame-*.color.png", # Or specific frame numbers/step
    "image_size": 512, # From notebook
    "silent": False,
    "profiling": False, # Set to True for timing info
    "align_conf_percentile": 85.0, # From notebook's alignment step
    "ply_pts_key": "pts3d_local_aligned_to_global", # From notebook
    "ply_conf_key": "conf_local", # From notebook
    "ply_conf_percentile": 0.0, # Default from notebook's export_combined_ply, means effectively no filtering based on this percentile unless value is higher
    "ply_flip_axes": True, # Often needed for standard viewers
    "ply_max_points": 1_000_000, # Example from notebook
    "ply_sampling_strategy": "uniform", # Example from notebook
    "max_frames_per_sequence": None, # Optional: limit number of frames from the glob
    "frame_step": 1, # Optional: pick every Nth frame from the globbed list
    "model_max_parallel_views": 20, # From notebook (model.set_max_parallel_views_for_head(20))
})

# Basic logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main(cfg: DictConfig): # Changed to accept DictConfig
    logging.info("Starting 7Scenes processing script.")
    # If not using Hydra's decorator to populate cfg, then cfg would be DEFAULT_ARGS passed directly.
    # For this subtask, we assume cfg is populated (e.g. by passing DEFAULT_ARGS).
    logging.info(f"Configuration: {OmegaConf.to_yaml(cfg)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    if not Path(cfg.model_checkpoint_path).exists():
        logging.error(f"Model checkpoint path not found: {cfg.model_checkpoint_path}")
        logging.error("Please update 'model_checkpoint_path' in the script or configuration.")
        return
    
    if not Path(cfg.data_base_path).exists() or not Path(cfg.data_base_path).is_dir():
        logging.error(f"Data base path not found or is not a directory: {cfg.data_base_path}")
        logging.error("Please update 'data_base_path' in the script or configuration.")
        return

    lit_module, model = load_model(cfg.model_checkpoint_path, device)

    if model is None:
        logging.error(f"Failed to load model from {cfg.model_checkpoint_path}.")
        return
    
    if hasattr(model, 'set_max_parallel_views_for_head') and cfg.model_max_parallel_views is not None:
        model.set_max_parallel_views_for_head(cfg.model_max_parallel_views)
        logging.info(f"Set model_max_parallel_views_for_head to {cfg.model_max_parallel_views}")
    elif cfg.model_max_parallel_views is not None: # only warn if it was set but function doesn't exist
        logging.warning("Model does not have 'set_max_parallel_views_for_head' method, or cfg.model_max_parallel_views not set.")


    output_path = Path(cfg.output_dir)
    if not output_path.exists():
        output_path.mkdir(parents=True, exist_ok=True)
        logging.info(f"Created output directory: {output_path}")

    if not cfg.sequences:
        logging.warning("No sequences defined in the configuration. Exiting.")
        return

    for seq_info in cfg.sequences:
        scene_name = seq_info["scene_name"]
        scene_id = seq_info["scene_id"] # This is typically like 'seq-03'
        
        logging.info(f"Preparing to process: {scene_name} / {scene_id}")

        image_dir_path = Path(cfg.data_base_path) / scene_name / scene_id
        
        if not image_dir_path.is_dir():
            logging.warning(f"Image directory not found for {scene_name}/{scene_id} at {image_dir_path}. Skipping.")
            continue

        image_files = sorted(glob.glob(str(image_dir_path / cfg.image_glob_pattern)))
        
        if not image_files:
            logging.warning(f"No images found in {image_dir_path} matching pattern '{cfg.image_glob_pattern}'. Skipping sequence {scene_name}/{scene_id}.")
            continue
        
        # Frame selection logic
        if cfg.max_frames_per_sequence is not None and cfg.max_frames_per_sequence > 0:
            image_files = image_files[:cfg.max_frames_per_sequence]
            logging.info(f"Limited to a maximum of {len(image_files)} frames for {scene_name}/{scene_id}.")

        if cfg.frame_step is not None and cfg.frame_step > 1:
            image_files = image_files[::cfg.frame_step]
            logging.info(f"Selected every {cfg.frame_step}-th frame, resulting in {len(image_files)} frames for {scene_name}/{scene_id}.")

        if not image_files: # After potential filtering
            logging.warning(f"No images remaining for {scene_name}/{scene_id} after frame selection. Skipping.")
            continue
            
        logging.info(f"Processing {len(image_files)} frames for {scene_name}/{scene_id}.")

        try:
            process_scene(
                scene_name=scene_name,
                seq_id=scene_id, # Pass the full sequence ID like "seq-03"
                image_files=image_files,
                model=model,
                lit_module=lit_module,
                device=device,
                output_dir=str(output_path), # Ensure it's a string
                args=cfg # Pass the whole config for process_scene to use sub-args
            )
            logging.info(f"Successfully processed and saved {scene_name}-{scene_id}.ply")
        except Exception as e:
            logging.error(f"Error processing {scene_name}-{scene_id}: {e}", exc_info=True) # Add exc_info for traceback
            # Continue to the next sequence

    logging.info("All specified sequences processed.")

if __name__ == "__main__":
    # For direct script execution without full Hydra CLI,
    # we directly pass the DEFAULT_ARGS.
    # If you were to use Hydra CLI, you'd decorate main with 
    # @hydra.main(config_path="configs", config_name="config") or similar
    # and Hydra would populate 'cfg'.
    
    # This makes the script runnable as `python process_7scenes.py`
    # User would need to edit DEFAULT_ARGS in the script itself for paths.
    main(DEFAULT_ARGS)
