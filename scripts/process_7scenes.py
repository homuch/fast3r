# Script to process 7SCENES dataset
import os
import glob
import torch
import numpy as np
from plyfile import PlyData, PlyElement
from fast3r.dust3r.utils.image import load_images
from fast3r.dust3r.inference_multiview import inference
from fast3r.models.fast3r import Fast3R
from fast3r.models.multiview_dust3r_module import MultiViewDUSt3RLitModule

TARGET_SEQUENCES = [
    ('chess', 'seq-03'),
    ('fire', 'seq-03'),
    ('heads', 'seq-01'),
    ('office', 'seq-02'),
    ('office', 'seq-06'),
    ('office', 'seq-07'),
    ('office', 'seq-09'),
    ('pumpkin', 'seq-01'),
    ('redkitchen', 'seq-03'),
    ('redkitchen', 'seq-04'),
    ('redkitchen', 'seq-06'),
    ('redkitchen', 'seq-12'),
    ('redkitchen', 'seq-14'),
    ('stairs', 'seq-01'),
]

OUTPUT_DIR = "test"

def save_ply(filepath, points, colors=None):
    """Saves a point cloud to a .ply file.

    Args:
        filepath (str): The path to save the .ply file.
        points (np.ndarray): NxD array of point coordinates (typically D=3).
        colors (np.ndarray, optional): NxK array of point colors (typically K=3 for RGB).
                                     Values should be uint8 (0-255).
    """
    points = np.asarray(points)
    if points.ndim == 1: # Single point
        points = points[np.newaxis, :]
    if points.shape[0] == 0:
        print(f"Warning: Attempting to save an empty point cloud to {filepath}")
        # Create an empty file or a file with just headers?
        # For now, let's just return to avoid errors with PlyElement
        return

    vertex_data = [tuple(p) for p in points]
    vertex_dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4')]

    if colors is not None:
        colors = np.asarray(colors)
        if colors.ndim == 1: # Single color
            colors = colors[np.newaxis, :]
        assert colors.shape[0] == points.shape[0], "Mismatch between number of points and colors"
        assert colors.shape[1] == 3, "Colors should be RGB"
        assert colors.dtype == np.uint8, "Color values should be uint8"
        
        new_vertex_data = []
        for i in range(len(vertex_data)):
            new_vertex_data.append(vertex_data[i] + tuple(colors[i]))
        vertex_data = new_vertex_data
        vertex_dtype.extend([('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])

    el = PlyElement.describe(np.array(vertex_data, dtype=vertex_dtype), 'vertex')
    PlyData([el], text=True).write(filepath)
    print(f"Saved point cloud to {filepath}")

def main():
    # Create the main output directory if it doesn't exist
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # --- Setup ---
    print("Setting up model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Fast3R.from_pretrained("jedyang97/Fast3R_ViT_Large_512")
    model = model.to(device)
    print("Creating MultiViewDUSt3RLitModule for inference...")
    lit_module = MultiViewDUSt3RLitModule.load_for_inference(model)
    model.eval() 
    lit_module.eval()
    print("MultiViewDUSt3RLitModule setup complete.")
    print("Model setup complete.")

    base_path = "../7SCENES" # Assuming 7SCENES dataset is one level up

    for scene_name, sequence_id in TARGET_SEQUENCES:
        print(f"Processing sequence: {scene_name} - {sequence_id}")

        sequence_data_path = os.path.join(
            base_path, scene_name, "test", sequence_id
        )
        
        image_files_pattern = os.path.join(sequence_data_path, "frame-*.color.png")
        image_files = sorted(glob.glob(image_files_pattern))

        output_ply_path = os.path.join(OUTPUT_DIR, f"{scene_name}-{sequence_id}.ply")

        print(f"  Found image files: {image_files}")
        print(f"  Intended output PLY path: {output_ply_path}")

        if not image_files:
            print(f"No image files found for {scene_name} {sequence_id}. Skipping.")
            print("-" * 20)
            continue
        
        print(f"Loading {len(image_files)} images...")
        images = load_images(image_files, size=512) 

        print("Running inference...")
        output_dict, profiling_info = inference(
            images,
            model,
            device,
            dtype=torch.float32,
            verbose=True,
            profiling=True, 
        )
        print(f"Inference complete for {scene_name} {sequence_id}.")
        print(f"Output keys: {output_dict.keys()}")
        if 'preds' in output_dict:
            print(f"Number of predictions (views from inference): {len(output_dict['preds'])}")
            if len(output_dict['preds']) > 0:
                first_pred_keys = output_dict['preds'][0].keys()
                print(f"Keys in first prediction: {first_pred_keys}")
                if 'pts3d_in_other_view' in first_pred_keys:
                    print(f"Shape of 'pts3d_in_other_view' for first view: {output_dict['preds'][0]['pts3d_in_other_view'].shape}")
        
        # Estimate Camera Poses
        print("Estimating camera poses...")
        poses_c2w_batch, _ = lit_module.estimate_camera_poses(
            output_dict['preds'],
            niter_PnP=100, 
            focal_length_estimation_method='first_view_from_global_head'
        )
        camera_poses_c2w = poses_c2w_batch[0] # list of [4,4] camera-to-world poses
        print(f"Estimated {len(camera_poses_c2w)} camera poses.")

        # Transform and Aggregate Points
        all_world_points = []
        # all_world_colors = [] # Uncomment if color extraction is implemented

        if 'preds' in output_dict and len(output_dict['preds']) == len(camera_poses_c2w):
            for view_idx, pred in enumerate(output_dict['preds']):
                if 'pts3d_in_other_view' not in pred:
                    print(f"Warning: 'pts3d_in_other_view' not found for view {view_idx}. Skipping this view.")
                    continue

                pts_cam_view = pred['pts3d_in_other_view'].cpu().numpy() 
                
                pts_cam_view = pts_cam_view.squeeze() 
                if pts_cam_view.ndim == 3: 
                    pts_cam_view = pts_cam_view.reshape(-1, 3)
                
                if pts_cam_view.shape[0] == 0: 
                    continue

                pose_c2w = camera_poses_c2w[view_idx] 

                if isinstance(pose_c2w, torch.Tensor):
                    pose_c2w = pose_c2w.cpu().numpy()

                pts_cam_view_homogeneous = np.hstack((pts_cam_view, np.ones((pts_cam_view.shape[0], 1))))
                pts_world_homogeneous = (pose_c2w @ pts_cam_view_homogeneous.T).T
                pts_world = pts_world_homogeneous[:, :3] / pts_world_homogeneous[:, 3:4]
                
                all_world_points.append(pts_world)

                # TODO: Extract colors if available
                # colors_view = pred.get('rgb_colors_for_pts3d_in_other_view') # Example key
                # if colors_view is not None:
                #    colors_view = colors_view.cpu().numpy().squeeze().reshape(-1, 3)
                #    all_world_colors.append(colors_view)


            if all_world_points:
                aggregated_points = np.concatenate(all_world_points, axis=0)
                print(f"Aggregated {aggregated_points.shape[0]} points from {len(output_dict['preds'])} views.")
                
                # aggregated_colors_final = None
                # if all_world_colors and len(all_world_colors) == len(all_world_points) : # ensure all views had colors
                #    aggregated_colors_final = np.concatenate(all_world_colors, axis=0)
                #    # Ensure colors are uint8, assuming they are float [0,1] from model
                #    if aggregated_colors_final.dtype != np.uint8:
                #        aggregated_colors_final = (aggregated_colors_final * 255).astype(np.uint8)
                
                save_ply(output_ply_path, aggregated_points, colors=None) # Pass aggregated_colors_final if implemented
            else:
                print(f"No points were aggregated for {scene_name} {sequence_id}. PLY file not saved.")
        else:
            print(f"Mismatch in number of predictions ({len(output_dict.get('preds', []))}) and poses ({len(camera_poses_c2w)}), or no predictions found for {scene_name} {sequence_id}.")

        print("-" * 20)

if __name__ == "__main__":
    main()
