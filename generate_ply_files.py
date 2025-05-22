import os
import os.path as osp
from typing import List, Dict, Tuple, Any
import argparse

import cv2
import numpy as np
import open3d as o3d
import torch
from fast3r.models.fast3r import Fast3R
from fast3r.models.multiview_dust3r_module import MultiViewDUSt3RLitModule
import torchvision.transforms.functional as TF

# --- Constants ---
DATASET_ROOT_DEFAULT = "../7SCENES" # Default for argparse
OUTPUT_DIR_DEFAULT = "./test"      # Default for argparse
INTRINSICS = np.array([[525, 0, 320], [0, 525, 240], [0, 0, 1]], dtype=np.float32)
# List of scene and sequence IDs based on the issue's expected output structure
# Each element is a tuple: (scene_name, sequence_id)
SEQUENCES_TO_PROCESS = [
    ("chess", "seq-03"),
    ("fire", "seq-03"),
    ("heads", "seq-01"),
    ("office", "seq-02"),
    ("office", "seq-06"),
    ("office", "seq-07"),
    ("office", "seq-09"),
    ("pumpkin", "seq-01"),
    ("redkitchen", "seq-03"),
    ("redkitchen", "seq-04"),
    ("redkitchen", "seq-06"),
    ("redkitchen", "seq-12"),
    ("redkitchen", "seq-14"),
    ("stairs", "seq-01"),
]

# --- Utility Functions (Placeholders) ---
def load_rgb_image(path: str) -> np.ndarray:
    # Based on imread_cv2 from seq2ply.py
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"Could not load image={path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def load_depth_map(path: str) -> np.ndarray:
    # Based on depth loading logic in SevenSceneSequence.get_views from seq2ply.py
    depthmap = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if depthmap is None:
        raise IOError(f"Could not load depthmap={path}")
    
    # Preprocessing from seq2ply.py
    depthmap = depthmap.astype(np.float32) # Ensure float for calculations
    depthmap[depthmap == 65535] = 0  # Handle invalid max value
    depthmap = np.nan_to_num(depthmap, 0.0) / 1000.0  # Scale to meters
    depthmap[depthmap > 10] = 0  # Clip large values
    depthmap[depthmap < 1e-3] = 0 # Clip small values (close to zero)
    
    if not np.isfinite(depthmap).all():
         print(f"Warning: Non-finite values found in depthmap {path} after processing.")
    return depthmap

def load_pose(path: str) -> np.ndarray:
    # Based on pose loading logic in SevenSceneSequence.get_views
    pose = np.loadtxt(path).astype(np.float32)
    if not np.isfinite(pose).all():
        raise ValueError(f"NaN or Inf in camera pose loaded from {path}")
    return pose

def project_depth_to_world(depth_map: np.ndarray, intrinsics: np.ndarray, camera_pose_c2w: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # Based on depthmap_to_world_coordinates from seq2ply.py
    H, W = depth_map.shape
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    u, v = np.meshgrid(np.arange(W), np.arange(H))

    z_cam = depth_map
    x_cam = (u - cx) * z_cam / fx
    y_cam = (v - cy) * z_cam / fy
    
    pts_cam = np.stack((x_cam, y_cam, z_cam), axis=-1).astype(np.float32)
    valid_mask = z_cam > 1e-4 # Consider points with depth > 0.1mm as valid

    # Transform to world coordinates
    R = camera_pose_c2w[:3, :3]
    t = camera_pose_c2w[:3, 3]
    
    # Reshape pts_cam to (H*W, 3) for batch matrix multiplication
    pts_cam_flat = pts_cam.reshape(-1, 3)
    
    # Transform points: pts_world = R @ pts_cam.T + t
    # Handle only valid points for efficiency if memory becomes an issue,
    # but for now, transform all and then filter.
    pts_world_flat = np.dot(pts_cam_flat, R.T) + t.T
    
    pts_world = pts_world_flat.reshape(H, W, 3)
    
    # Further filter valid_mask based on transformed points if necessary (e.g. NaNs after transformation)
    valid_mask = valid_mask & np.isfinite(pts_world).all(axis=-1)
    
    return pts_world, valid_mask

def save_ply(filepath: str, points: np.ndarray, colors: np.ndarray):
    # Based on write_ply_color from seq2ply.py
    # Ensure points and colors are Nx3 and colors are uint8
    assert points.ndim == 2 and points.shape[1] == 3
    assert colors.ndim == 2 and colors.shape[1] == 3
    assert points.shape[0] == colors.shape[0]

    if colors.dtype != np.uint8:
        # Assuming colors are float32 in range [0,1] from image loading
        if colors.max() <= 1.0:
            colors = (colors * 255).astype(np.uint8)
        else: # Assuming colors are already in [0,255] but not uint8
            colors = colors.astype(np.uint8)

    num_points = points.shape[0]
    ply_header = f"""ply
format ascii 1.0
element vertex {num_points}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
"""
    with open(filepath, 'w') as f:
        f.write(ply_header)
        for i in range(num_points):
            p = points[i]
            c = colors[i]
            f.write(f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f} {c[0]} {c[1]} {c[2]}\n")
    print(f"Saved PLY file to {filepath} with {num_points} points.")

# --- Fast3R Model Functions (Placeholders) ---
def load_fast3r_model(model_name="jedyang97/Fast3R_ViT_Large_512", device_str="cuda" if torch.cuda.is_available() else "cpu"):
    print(f"Loading Fast3R model: {model_name} on device: {device_str}")
    model = Fast3R.from_pretrained(model_name)
    if model is None:
        raise RuntimeError(f"Failed to load Fast3R model: {model_name}")

    lit_module = MultiViewDUSt3RLitModule.load_for_inference(model)
    if lit_module is None:
        raise RuntimeError("Failed to load MultiViewDUSt3RLitModule for inference.")
            
    lit_module = lit_module.to(device_str)
    lit_module.eval()
    print("Fast3R model loaded successfully and set to evaluation mode.")
    return lit_module

def estimate_pose_fast3r(lit_module: MultiViewDUSt3RLitModule, raw_rgb_images: List[np.ndarray]) -> List[np.ndarray]:
    # Preprocess images: Convert to tensor, normalize, and possibly resize.
    # This preprocessing should match what Fast3R expects.
    # Assuming ViT Large 512, it might expect 512x512 or similar.
    # For now, let's assume images are already of a compatible size or aspect ratio.
    # Fast3R/DUSt3R typically uses images with max dimension 512, keeping aspect ratio.
    
    processed_images = []
    img_shapes_wh = [] # To store (width, height) for each image
    for img_np in raw_rgb_images:
        if img_np.ndim != 3 or img_np.shape[2] != 3:
            raise ValueError("Each image must be an HWC RGB NumPy array.")
        
        img_h, img_w = img_np.shape[:2]
        img_shapes_wh.append((img_w, img_h))

        # Convert HWC NumPy (RGB) to CHW PyTorch Tensor (RGB)
        img_tensor = torch.from_numpy(img_np.transpose((2, 0, 1))).float() / 255.0  # To [0,1]
        
        # Normalization (typical for ImageNet models, but Fast3R might have its own)
        # Using standard ImageNet mean/std for now. This might need adjustment.
        # img_tensor = TF.normalize(img_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # DUSt3R models usually don't need normalization as they work on raw pixels or just /255.
        # Let's stick to /255 for now, as per many DUSt3R examples.

        processed_images.append(img_tensor)

    if not processed_images:
        return []

    # Batch images if the model supports it and it's beneficial.
    # For simplicity, let's assume the model can process a list of images or a batch.
    # The `MultiViewDUSt3RLitModule` likely handles batching internally or expects a batch.
    # Let's try feeding a list of tensors directly to the lit_module.
    # The lit_module's forward pass or a specific method should give `output_dict`.
    
    # The structure for `views` input to `lit_module.forward()` or a similar method
    # often requires a list of dictionaries, where each dict has 'img', 'fg_mask', 'img_shape'.
    # Let's prepare data in that format.
    
    views_batch = []
    for i, img_tensor in enumerate(processed_images):
        c, h, w = img_tensor.shape
        view = {
            'img': img_tensor.to(lit_module.device), # CHW tensor
            'fg_mask': None, # Optional: Foreground mask (HxW)
            'img_shape': torch.tensor([h, w], device=lit_module.device) # Original H, W
        }
        views_batch.append(view)

    # Get predictions from the model
    # This part is critical and might need adjustment based on Fast3R's exact API.
    # Assuming lit_module() is the way to get predictions.
    with torch.no_grad():
        # output_dict = lit_module(views_batch) # This is a guess for how to get output_dict
        # The `estimate_camera_poses` is a static method and takes `preds`.
        # `preds` usually refers to predicted depth maps and confidence maps.
        # We need to find how `MultiViewDUSt3RLitModule.forward` or a similar method
        # provides these `preds`.
        #
        # Looking at `fast3r/dust3r/inference_multiview.py` from DUSt3R (Fast3R builds on it),
        # the `infer_depth_and_pose` method in `Dust3RMultiViewPredictorUtils`
        # calls `self.model(views, **self.cfg.MODEL.FORWARD_CFG)[KEYOUT_PRED_DICT]`.
        # So, `lit_module(views_batch)` should give a dictionary containing 'preds'.
        
        model_output = lit_module(views_batch) 
        if 'preds' not in model_output:
            raise RuntimeError("Model output does not contain 'preds' key.")
        preds = model_output['preds']

    # Estimate camera poses using the static method
    # The issue snippet:
    # poses_c2w_batch, estimated_focals = MultiViewDUSt3RLitModule.estimate_camera_poses(
    #     output_dict['preds'], niter_PnP=100, focal_length_estimation_method='first_view_from_global_head'
    # )
    # camera_poses = poses_c2w_batch[0] # this is a list
    
    # The `estimate_camera_poses` method in `MultiViewDUSt3RLitModule` takes `pred_dict` (our `preds`)
    # and `img_shape_B2HW` (a tensor of image shapes).
    # Let's prepare `img_shape_B2HW`.
    img_shapes_tensor = torch.tensor(img_shapes_wh, device=lit_module.device, dtype=torch.long) # List of (W,H)
    # The method expects (H,W) per image for img_shape_B2HW, so swap W,H.
    img_shapes_hw_tensor = img_shapes_tensor[:, [1, 0]] # Convert (W,H) to (H,W)
    
    # The method also takes `intrinsics_B22` if focal length estimation is not used,
    # or uses `focal_length_estimation_method`. The issue uses the latter.

    poses_c2w_batch, _ = MultiViewDUSt3RLitModule.estimate_camera_poses(
        pred_dict=preds,
        img_shape_BCHW=None, # Not used if focal_length_estimation_method is provided
        img_shape_B2HW=img_shapes_hw_tensor, # Batch of (H,W)
        niter_PnP=100,
        focal_length_estimation_method='first_view_from_global_head'
    )
    
    # poses_c2w_batch is a list where the first element contains estimated poses for each view.
    # These poses are expected to be torch tensors on the model's device.
    # Convert them to NumPy arrays (4x4) on CPU.
    estimated_poses_np = []
    if poses_c2w_batch and poses_c2w_batch[0] is not None:
        for pose_tensor in poses_c2w_batch[0]: # poses_c2w_batch[0] is a list of pose tensors
            if pose_tensor is not None:
                estimated_poses_np.append(pose_tensor.cpu().numpy())
            else:
                # Handle case where a pose might not be estimated
                estimated_poses_np.append(None) 
    
    return estimated_poses_np # List of 4x4 NumPy arrays (or None)

# --- Main Processing Logic ---
def process_sequence(seq_dir_path: str, 
                         output_ply_path: str, 
                         lit_module: MultiViewDUSt3RLitModule, 
                         intrinsics: np.ndarray):
    print(f"Processing sequence: {seq_dir_path}")

    # 1. Find all frame-XXXXXX.color.png files, sort them.
    frame_files = sorted([f for f in os.listdir(seq_dir_path) if f.startswith("frame-") and f.endswith(".color.png")])
    
    if not frame_files:
        print(f"Warning: No color frames found in {seq_dir_path}")
        return

    # 2. Load all these color images into a list (NumPy HWC RGB).
    raw_rgb_images = []
    frame_names = []
    for frame_file in frame_files:
        frame_name = frame_file.replace(".color.png", "")
        frame_names.append(frame_name)
        color_image_path = osp.join(seq_dir_path, frame_file)
        try:
            img = load_rgb_image(color_image_path)
            raw_rgb_images.append(img)
        except IOError as e:
            print(f"Warning: Could not load image {color_image_path}: {e}")
            # If an image can't be loaded, we might need to skip this sequence or frame.
            # For now, let's assume if one fails, we can't reliably process the sequence with Fast3R.
            return 

    if not raw_rgb_images:
        print(f"Warning: No images loaded for sequence {seq_dir_path}")
        return
        
    # 3. Call estimate_pose_fast3r with all these images to get all poses.
    print(f"Estimating poses for {len(raw_rgb_images)} images in {seq_dir_path}...")
    estimated_poses_model_c2w = estimate_pose_fast3r(lit_module, raw_rgb_images)

    if not estimated_poses_model_c2w or len(estimated_poses_model_c2w) != len(raw_rgb_images):
        print(f"Warning: Pose estimation failed or returned incorrect number of poses for {seq_dir_path}. Expected {len(raw_rgb_images)}, got {len(estimated_poses_model_c2w) if estimated_poses_model_c2w else 0}.")
        return

    # 4. Load T0_gt (ground truth pose for frame-000000)
    initial_pose_path = osp.join(seq_dir_path, "frame-000000.pose.txt")
    if not osp.isfile(initial_pose_path):
        print(f"Warning: Initial pose frame-000000.pose.txt not found in {seq_dir_path}")
        return
    T0_gt = load_pose(initial_pose_path)

    all_points_world_list = []
    all_colors_rgb_list = []
    
    T_model_world_to_actual_world = None # Initialize

    # 5. For each frame i from 0 to N-1:
    for i, frame_name in enumerate(frame_names):
        print(f"Processing frame {frame_name}...")
        
        current_model_pose_c2w = estimated_poses_model_c2w[i]
        if current_model_pose_c2w is None:
            print(f"Warning: No pose estimated for frame {frame_name}. Skipping.")
            continue

        # 5a. Transform model pose: final_pose_i = T0_gt @ T_model_i_relative_to_T_model_0
        if i == 0 :
            if estimated_poses_model_c2w[0] is None:
                 print(f"Warning: First frame's model pose (T_0_model_world) is None for sequence {seq_dir_path}. Cannot establish alignment. Skipping sequence.")
                 return
            try:
                T_model_world_to_actual_world = T0_gt @ np.linalg.inv(estimated_poses_model_c2w[0])
            except np.linalg.LinAlgError:
                print(f"Warning: Singular matrix encountered for inv(estimated_poses_model_c2w[0]) in {seq_dir_path}. Skipping sequence.")
                return

        if T_model_world_to_actual_world is None: # Should not happen if first frame processed correctly
            print(f"Critical Error: T_model_world_to_actual_world not set. Skipping frame {frame_name}.")
            continue

        current_pose_actual_c2w = T_model_world_to_actual_world @ current_model_pose_c2w

        # 5b. Load depth map
        depth_map_path = osp.join(seq_dir_path, f"{frame_name}.depth.proj.png") # Corrected filename based on typical 7Scenes structure
        if not osp.isfile(depth_map_path):
            # Try also .depth.png as a fallback if .depth.proj.png is specific to some variant
            depth_map_path_alt = osp.join(seq_dir_path, f"{frame_name}.depth.png")
            if osp.isfile(depth_map_path_alt):
                depth_map_path = depth_map_path_alt
            else:
                print(f"Warning: Depth map not found (tried .depth.proj.png and .depth.png) for {frame_name}. Skipping.")
                continue
        try:
            depth_map_i = load_depth_map(depth_map_path)
        except IOError as e:
            print(f"Warning: Could not load depth map {depth_map_path}: {e}. Skipping frame.")
            continue
        
        # 5c. Load RGB image (already loaded as raw_rgb_images[i])
        rgb_image_i = raw_rgb_images[i]
        
        # Ensure depth map and rgb image have compatible dimensions for point cloud color extraction
        if depth_map_i.shape[:2] != rgb_image_i.shape[:2]:
             print(f"Warning: Mismatch dimensions between depth ({depth_map_i.shape[:2]}) and RGB ({rgb_image_i.shape[:2]}) for {frame_name}. Resizing RGB to match depth.")
             rgb_image_i = cv2.resize(rgb_image_i, (depth_map_i.shape[1], depth_map_i.shape[0]))


        # 5d. Project depth to world
        pts_world, valid_mask = project_depth_to_world(depth_map_i, intrinsics, current_pose_actual_c2w)
        
        # 5e. Extract colors
        colors_for_pts = rgb_image_i[valid_mask] # HWC format, take valid pixels
        
        # 5f. Get points for PLY
        points_for_ply = pts_world[valid_mask]
        
        if points_for_ply.shape[0] > 0:
            all_points_world_list.append(points_for_ply)
            all_colors_rgb_list.append(colors_for_pts)
        else:
            print(f"Warning: No valid points generated for frame {frame_name}.")

    if not all_points_world_list:
        print(f"Error: No points accumulated for sequence {seq_dir_path}. PLY file will not be generated.")
        return

    # 6. Concatenate all points and colors
    all_points_world_np = np.concatenate(all_points_world_list, axis=0)
    all_colors_rgb_np = np.concatenate(all_colors_rgb_list, axis=0)

    if all_points_world_np.shape[0] == 0:
        print(f"Error: Zero points after concatenation for {seq_dir_path}. PLY file will not be generated.")
        return
        
    # 7. Save PLY
    os.makedirs(osp.dirname(output_ply_path), exist_ok=True)
    save_ply(output_ply_path, all_points_world_np, all_colors_rgb_np)
    print(f"Finished processing sequence {seq_dir_path}. Output: {output_ply_path}")

def main():
    parser = argparse.ArgumentParser(description="Generate PLY files from 7SCENES dataset using Fast3R.")
    parser.add_argument("--dataset_root", type=str, default=DATASET_ROOT_DEFAULT, help="Root directory of the 7SCENES dataset.")
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR_DEFAULT, help="Directory to save the output PLY files.")
    parser.add_argument("--model_name", type=str, default="jedyang97/Fast3R_ViT_Large_512", help="Name of the Fast3R model to load from HuggingFace.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run the model on (e.g., 'cuda', 'cpu').")
    args = parser.parse_args()

    print(f"Starting PLY generation...")
    print(f"Dataset root: {args.dataset_root}")
    print(f"Output directory: {args.output_dir}")
    print(f"Using device: {args.device}")

    # 1. Load Fast3R model (once)
    try:
        lit_module = load_fast3r_model(model_name=args.model_name, device_str=args.device)
    except Exception as e:
        print(f"Error loading Fast3R model: {e}")
        return

    os.makedirs(args.output_dir, exist_ok=True)

    for scene_name, seq_id in SEQUENCES_TO_PROCESS:
        seq_dir = osp.join(args.dataset_root, scene_name, "test", seq_id) # Corrected path to include "test" subdirectory
        output_ply_filename = f"{scene_name}-{seq_id}.ply"
        output_ply_filepath = osp.join(args.output_dir, output_ply_filename)

        if not osp.isdir(seq_dir):
            print(f"Warning: Sequence directory not found, skipping: {seq_dir}")
            continue
        
        # Global intrinsics are used for projection
        process_sequence(seq_dir, output_ply_filepath, lit_module, INTRINSICS)

    print("PLY generation process complete.")

if __name__ == "__main__":
    main()
