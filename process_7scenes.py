#!/usr/bin/env python3

import torch
import os
import glob
import numpy as np
from PIL import Image
from fast3r.dust3r.utils.image import load_images
from fast3r.dust3r.inference_multiview import inference
from fast3r.models.fast3r import Fast3R
from fast3r.models.multiview_dust3r_module import MultiViewDUSt3RLitModule

def write_ply_color(name, xyz, rgb):
    """
    Save a colored point cloud to a .ply file.

    Args:
        name (str): Path to save the .ply file.
        xyz (np.ndarray): (N, 3) array of 3D coordinates.
        rgb (np.ndarray): (N, 3) array of uint8 RGB colors.
    """
    assert xyz.shape[0] == rgb.shape[0] and xyz.shape[1] == 3 and rgb.shape[1] == 3
    N = xyz.shape[0]
    if N == 0:
        print(f"Warning: No points to write for {name}. Skipping PLY file creation.")
        return

    ply_header = f"""ply
format ascii 1.0
element vertex {N}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
"""
    # Ensure rgb is uint8
    if rgb.dtype != np.uint8:
        print(f"Warning: RGB data for {name} is not uint8 ({rgb.dtype}). Clamping and converting.")
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)

    with open(name, 'w') as f:
        f.write(ply_header)
        for i in range(N): # Loop to avoid creating massive intermediate string list
            p = xyz[i]
            c = rgb[i]
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {c[0]} {c[1]} {c[2]}\n")
    print(f"  Saved PLY file: {name} with {N} points.")

def main():
    """Main function to load the model and prepare for processing."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Attempting to load model on {device}...")

    try:
        model = Fast3R.from_pretrained("jedyang97/Fast3R_ViT_Large_512")
        model = model.to(device)
        lit_module = MultiViewDUSt3RLitModule.load_for_inference(model)
        model.eval()
        lit_module.eval()
        print(f"Model loaded successfully on {device}.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    dataset_base_dir = "../7SCENES"
    output_base_dir = "./test"
    os.makedirs(output_base_dir, exist_ok=True)

    scenes = ["chess", "fire", "heads", "office", "pumpkin", "redkitchen", "stairs"]

    for scene_name in scenes:
        scene_path = os.path.join(dataset_base_dir, scene_name)
        
        # Find seq-xx directories within the 'test' subdirectory
        seq_dirs_pattern = os.path.join(scene_path, "test", "seq-*")
        sequence_directories = sorted(glob.glob(seq_dirs_pattern))

        for seq_dir_path in sequence_directories:
            seq_name = os.path.basename(seq_dir_path)
            
            image_files_pattern = os.path.join(seq_dir_path, "*.color.png")
            filelist = sorted(glob.glob(image_files_pattern))

            if not filelist:
                print(f"Warning: No images found in {seq_dir_path} using pattern {image_files_pattern}")
                continue

            images = load_images(filelist, size=512, verbose=False)
            print(f"Processing {seq_dir_path}, loaded {len(images)} images.")

            # Run Inference
            output_dict, _ = inference(images, model, device, dtype=torch.float32, verbose=False, profiling=False)
            preds = output_dict['preds']

            # Estimate Camera Poses
            poses_c2w_batch, _ = MultiViewDUSt3RLitModule.estimate_camera_poses(
                preds, 
                niter_PnP=100, 
                focal_length_estimation_method='first_view_from_global_head'
            )
            poses_c2w_model_frame = poses_c2w_batch[0]

            # Temporary Output (for verification)
            print(f"  Estimated poses_c2w_model_frame shape: {poses_c2w_model_frame.shape}")
            print(f"  Number of predictions (views): {len(preds)}")

            # Define Ground Truth Pose File Path
            gt_pose_file = os.path.join(seq_dir_path, "frame-000000.pose.txt")

            # Get Model's Estimated Pose for First Camera (corresponds to frame-000000)
            # Ensure it's a NumPy array on CPU for np.linalg.inv
            T_model_cam0_est = poses_c2w_model_frame[0].cpu().numpy() 

            # Initialize T_world_model to identity (no alignment if GT is missing)
            T_world_model = np.eye(4)

            if os.path.exists(gt_pose_file):
                T_world_cam0_gt = np.loadtxt(gt_pose_file)
                print(f"  Loaded ground truth pose from {gt_pose_file}")

                # Calculate Alignment Transformation
                T_world_model = T_world_cam0_gt @ np.linalg.inv(T_model_cam0_est)
                print(f"  Calculated T_world_model for alignment.")
            else:
                print(f"  Warning: Ground truth pose file not found at {gt_pose_file}. Using identity for T_world_model (no alignment).")
            
            # Temporary Output for T_world_model verification
            print(f"  T_world_model (first row): {T_world_model[0]}")

            # Initialize lists for aggregating points and colors for the current sequence
            all_points_world = []
            all_colors = []

            num_views = len(preds)
            for view_idx in range(num_views):
                pred_for_view = preds[view_idx]
                
                # Extract points in model frame
                # pts3d_in_other_view has shape (1, H, W, 3)
                pts_model_frame_tensor = pred_for_view['pts3d_in_other_view']
                pts_model_frame = pts_model_frame_tensor.squeeze(0).cpu().numpy().reshape(-1, 3)

                # Extract colors
                current_image_tensor = images[view_idx] # images is a list of (C,H,W) tensors
                img_np = current_image_tensor.permute(1, 2, 0).cpu().numpy() # H, W, C
                img_colors_uint8 = (img_np * 255).astype(np.uint8)
                colors_for_view = img_colors_uint8.reshape(-1, 3) # H*W, 3

                # Transform points to world frame
                # Convert pts_model_frame to homogeneous coordinates
                pts_model_homo = np.hstack((pts_model_frame, np.ones((pts_model_frame.shape[0], 1))))
                # Transform points: T_world_model @ pts_model_homo.T
                pts_world_homo = (T_world_model @ pts_model_homo.T).T
                # Convert back to Cartesian coordinates
                pts_world = pts_world_homo[:, :3] / pts_world_homo[:, 3:4]

                all_points_world.append(pts_world)
                all_colors.append(colors_for_view)

            if not all_points_world or not all_colors:
                print(f"  No points or colors extracted for {seq_dir_path}. Skipping further processing for this sequence.")
                continue
            
            final_points = np.concatenate(all_points_world, axis=0)
            final_colors = np.concatenate(all_colors, axis=0)
            print(f"  Aggregated {final_points.shape[0]} points for {seq_dir_path}")

            # Save the aggregated point cloud
            output_filename = f"{scene_name}-{seq_name}.ply" # Using scene_name from outer loop and seq_name
            output_ply_path = os.path.join(output_base_dir, output_filename)
            write_ply_color(output_ply_path, final_points, final_colors)


if __name__ == "__main__":
    main()
