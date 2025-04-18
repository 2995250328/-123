#!/usr/bin/env python3
# Copyright © Niantic, Inc. 2022.

import argparse
import logging
import math
import time
from distutils.util import strtobool
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader

import dsacstarsample
from ace_network import Regressor, scores_to_points
from dataset import CamLocDataset

import ace_vis_util as vutil
from ace_visualizer import ACEVisualizer
from ace_util import get_pixel_grid, to_homogeneous
from predict_pose import PosePredictor, pose_to_vector, weight_average

import os
from scipy.spatial import KDTree
from scipy.spatial.transform import Rotation as R
import threading

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


_logger = logging.getLogger(__name__)


def _strtobool(x):
    return bool(strtobool(x))


if __name__ == '__main__':
    # Setup logging.
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(
        description='Test a trained network on a specific scene.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('scene', type=Path,
                        help='path to a scene in the dataset folder, e.g. "datasets/Cambridge_GreatCourt"')

    parser.add_argument('network', type=Path, help='path to a network trained for the scene (just the head weights)')

    parser.add_argument('--encoder_path', type=Path, default=Path(__file__).parent / "ace_encoder_pretrained.pt",
                        help='file containing pre-trained encoder weights')

    parser.add_argument('--point_header_path', type=Path, default=Path(__file__).parent / "point_header.pt",
                        help='file containing pre-trained point header weights')

    parser.add_argument('--session', '-sid', default='',
                        help='custom session name appended to output files, '
                             'useful to separate different runs of a script')

    parser.add_argument('--image_resolution', type=int, default=480, help='base image resolution')

    # keypoint detection 
    parser.add_argument('--keypoint_threshold', type=float, default=0.0001,
                        help='keypoint detection threshold')

    parser.add_argument('--max_keypoints', type=int, default=1000,
                        help='The maximum number of keypoint detected')

    parser.add_argument('--remove_border', type=int, default=4,
                        help='The maximum number of keypoint detected')

    # test mode
    parser.add_argument('--test_mode', '-tm', type=int, default=1, choices=[0, 1, 2], 
                        help='test mode: 0 = ACE original mode, 1 = use sequence information, '
                        '2 = fast mode, use keypoint of one frame')


    # ACE is RGB-only, no need for this param.
    # parser.add_argument('--mode', '-m', type=int, default=1, choices=[1, 2], help='test mode: 1 = RGB, 2 = RGB-D')

    # DSACStar RANSAC parameters. ACE Keeps them at default.
    parser.add_argument('--hypotheses', '-hyps', type=int, default=64,
                        help='number of hypotheses, i.e. number of RANSAC iterations')

    parser.add_argument('--threshold', '-t', type=float, default=10,
                        help='inlier threshold in pixels (RGB) or centimeters (RGB-D)')

    parser.add_argument('--inlieralpha', '-ia', type=float, default=100,
                        help='alpha parameter of the soft inlier count; controls the softness of the '
                             'hypotheses score distribution; lower means softer')

    parser.add_argument('--maxpixelerror', '-maxerrr', type=float, default=100,
                        help='maximum reprojection (RGB, in px) or 3D distance (RGB-D, in cm) error when checking '
                             'pose consistency towards all measurements; error is clamped to this value for stability')

    # Params for the visualization. If enabled, it will slow down relocalisation considerably. But you get a nice video :)
    parser.add_argument('--render_visualization', type=_strtobool, default=False,
                        help='create a video of the mapping process')

    parser.add_argument('--render_target_path', type=Path, default='renderings',
                        help='target folder for renderings, visualizer will create a subfolder with the map name')

    parser.add_argument('--render_flipped_portrait', type=_strtobool, default=False,
                        help='flag for wayspots dataset where images are sideways portrait')

    parser.add_argument('--render_sparse_queries', type=_strtobool, default=False,
                        help='set to true if your queries are not a smooth video')

    parser.add_argument('--render_pose_error_threshold', type=int, default=20,
                        help='pose error threshold for the visualisation in cm/deg')

    parser.add_argument('--render_map_depth_filter', type=int, default=10,
                        help='to clean up the ACE point cloud remove points too far away')

    parser.add_argument('--render_camera_z_offset', type=int, default=4,
                        help='zoom out of the scene by moving render camera backwards, in meters')

    parser.add_argument('--render_frame_skip', type=int, default=1,
                        help='skip every xth frame for long and dense query sequences')

    opt = parser.parse_args()

    device = torch.device("cuda")
    num_workers = 6

    scene_path = Path(opt.scene)
    head_network_path = Path(opt.network)
    encoder_path = Path(opt.encoder_path)
    point_header_path = Path(opt.point_header_path)
    session = opt.session

    # Setup dataset.
    testset = CamLocDataset(
        scene_path / "test",
        mode=0,  # Default for ACE, we don't need scene coordinates/RGB-D.
        image_height=opt.image_resolution,
    )
    _logger.info(f'Test images found: {len(testset)}')

    # Setup dataloader. Batch size 1 by default.
    testset_loader = DataLoader(testset, shuffle=False, num_workers=6)

    # Load network weights.
    encoder_state_dict = torch.load(encoder_path, map_location="cpu")
    _logger.info(f"Loaded encoder from: {encoder_path}")
    head_state_dict = torch.load(head_network_path, map_location="cpu")
    _logger.info(f"Loaded head weights from: {head_network_path}")
    point_header_state_dict = torch.load(point_header_path, map_location="cpu")
    _logger.info(f"Loaded encoder from: {point_header_path}")

    # Create regressor.
    # network = Regressor.create_from_split_state_dict(encoder_state_dict, head_state_dict)
    network = Regressor.create_from_split_state_dict_with_point(encoder_state_dict, head_state_dict, point_header_state_dict)

    # Setup for evaluation.
    network = network.to(device)
    network.eval()

    # Save the outputs in the same folder as the network being evaluated.
    output_dir = head_network_path.parent
    scene_name = scene_path.name
    # This will contain aggregate scene stats (median translation/rotation errors, and avg processing time per frame).
    test_log_file = output_dir / f'test_{scene_name}_{opt.session}.txt'
    _logger.info(f"Saving test aggregate statistics to: {test_log_file}")
    # This will contain each frame's pose (stored as quaternion + translation) and errors.
    pose_log_file = output_dir / f'poses_{scene_name}_{opt.session}.txt'
    _logger.info(f"Saving per-frame poses and errors to: {pose_log_file}")

    # Setup output files.
    test_log = open(test_log_file, 'w', 1)
    pose_log = open(pose_log_file, 'w', 1)

    # Metrics of interest.
    avg_batch_time = 0
    num_batches = 0

    # Keep track of rotation and translation errors for calculation of the median error.
    rErrs = []
    tErrs = []

    # Percentage of frames predicted within certain thresholds from their GT pose.
    pct10_5 = 0
    pct5 = 0
    pct2 = 0
    pct1 = 0

    # Generate video of training process
    if opt.render_visualization:
        # infer rendering folder from map file name
        target_path = vutil.get_rendering_target_path(
            opt.render_target_path,
            opt.network)
        ace_visualizer = ACEVisualizer(target_path,
                                       opt.render_flipped_portrait,
                                       opt.render_map_depth_filter,
                                       reloc_vis_error_threshold=opt.render_pose_error_threshold)

        # we need to pass the training set in case the visualiser has to regenerate the map point cloud
        trainset = CamLocDataset(
            scene_path / "train",
            mode=0,  # Default for ACE, we don't need scene coordinates/RGB-D.
            image_height=opt.image_resolution,
            load_kpts=False,
        )

        # Setup dataloader. Batch size 1 by default.
        trainset_loader = DataLoader(trainset, shuffle=False, num_workers=2)

        ace_visualizer.setup_reloc_visualisation(
            frame_count=len(testset),
            data_loader=trainset_loader,
            network=network,
            camera_z_offset=opt.render_camera_z_offset,
            reloc_frame_skip=opt.render_frame_skip)
    else:
        ace_visualizer = None

    # Testing loop.
    pixel_grid_2HW = get_pixel_grid(network.OUTPUT_SUBSAMPLE)
    pose_pred = PosePredictor()
    last_image, last_points = None, None

    testing_start_time = time.time()
    with torch.no_grad():
        for image_B1HW, _, gt_pose_B44, _, intrinsics_B33, _, _, filenames, gray_images in testset_loader:

            t1 = time.time()

            batch_start_time = time.time()
            batch_size = image_B1HW.shape[0]

            image_B1HW = image_B1HW.to(device, non_blocking=True)


            # Predict scene coordinates.
            with autocast(enabled=True):
                if opt.test_mode > 0:
                    scene_coordinates_B3HW, scores = network.get_scene_coordinates_and_points(image_B1HW)
                    scores = scores.float()
                    keypoints, _ = scores_to_points(scores, opt.keypoint_threshold, opt.max_keypoints, opt.remove_border)
                else:
                    scene_coordinates_B3HW = network(image_B1HW)


            # We need them on the CPU to run RANSAC.
            scene_coordinates_B3HW = scene_coordinates_B3HW.float().cpu()
            if opt.test_mode > 0:
                keypoints = [kpts.cpu() for kpts in keypoints]

            # Each frame is processed independently.
            for frame_idx, (scene_coordinates_3HW, gt_pose_44, intrinsics_33, frame_path, gray_image) in enumerate(
                    zip(scene_coordinates_B3HW, gt_pose_B44, intrinsics_B33, filenames, gray_images)):

                t2 = time.time()

                # Extract focal length and principal point from the intrinsics matrix.
                focal_length = intrinsics_33[0, 0].item()
                ppX = intrinsics_33[0, 2].item()
                ppY = intrinsics_33[1, 2].item()
                # We support a single focal length.
                assert torch.allclose(intrinsics_33[0, 0], intrinsics_33[1, 1])

                # Remove path from file name
                frame_name = Path(frame_path).name

                # Allocate output variable.
                out_pose = torch.zeros((4, 4))


                # Compute the pose via RANSAC.
                # inlier_count = dsacstar.forward_rgb(
                #     scene_coordinates_3HW.unsqueeze(0),
                #     out_pose,
                #     opt.hypotheses,
                #     opt.threshold,
                #     focal_length,
                #     ppX,
                #     ppY,
                #     opt.inlieralpha,
                #     opt.maxpixelerror,
                #     network.OUTPUT_SUBSAMPLE,
                # )

                ################################## PnP RANSAC Keypoints ################################
                t3 = time.time()
                good_estimation = False
                if opt.test_mode > 0:
                    # kpts = track_info[1:, 1:]
                    # valid_kpts = kpts[kpt_mask[1:]]

                    valid_kpts = keypoints[frame_idx]

                    _, _, iH, iW = image_B1HW.shape
                    _, fH, fW = scene_coordinates_3HW.shape
                    scale_x, scale_y = float(fW) / iW, float(fH) / iH
                    scaled_kpts = valid_kpts.clone() - network.OUTPUT_SUBSAMPLE / 2
                    scaled_kpts[:, 0] *= scale_x  
                    scaled_kpts[:, 1] *= scale_y  
                    scaled_kpts = torch.round(scaled_kpts)
                    scaled_kpts[:, 0] = torch.clamp(scaled_kpts[:, 0], min=0, max=fW-1)
                    scaled_kpts[:, 1] = torch.clamp(scaled_kpts[:, 1], min=0, max=fH-1)
                    scaled_kpts = scaled_kpts.int()

                    index_tensor = scaled_kpts[:, 1] * fW + scaled_kpts[:, 0]
                    points_map_3d = scene_coordinates_3HW.permute(1, 2, 0).view(-1, 3)
                    points_3d = points_map_3d[index_tensor]

                    pixel_positions_B2HW = pixel_grid_2HW[:, :fH, :fW].clone()  # It's 2xHxW (actual H and W) now.
                    points_map_2d = pixel_positions_B2HW.permute(1, 2, 0).view(-1, 2)
                    points_2d = points_map_2d[index_tensor]

                    inliers = torch.zeros(len(points_2d), dtype=torch.int)

                    inlier_count = dsacstarsample.forward_sequence_rgb(
                        points_3d,
                        points_2d,
                        inliers,
                        out_pose,
                        4,
                        focal_length,
                        ppX,
                        ppY,
                        opt.maxpixelerror)
                    good_estimation = inlier_count >= 20

                    if opt.test_mode == 1:
                        # predict pose using histroical information
                        gray_image = gray_image.numpy()
                        K = intrinsics_33.numpy()
                        predicted_pose, predicted_inlier_num, predict_points, track_status = pose_pred.predict(gray_image, K, opt)

                        # compute pose
                        post_pose, good_estimation = weight_average(predicted_pose, predicted_inlier_num, out_pose.numpy(), inlier_count)

                        # update pose predictor
                        if good_estimation:
                            out_pose = torch.from_numpy(post_pose)
                            if inlier_count > 50:
                                inliers = inliers > 0
                                good_points_2d = points_2d[inliers].numpy()
                                good_points_3d = points_3d[inliers].numpy()
                                good_valid_kpts = valid_kpts[inliers].numpy()
                                pixel_bias = np.abs(good_points_2d - good_valid_kpts)
                                accurate_idx = (pixel_bias[:, 0] < 4) & (pixel_bias[:, 1] < 4)
                                pose_pred.update(gray_image, out_pose.numpy(), predict_points, track_status, good_valid_kpts[accurate_idx], good_points_3d[accurate_idx], K)

                t4 = time.time()


                if not good_estimation:
                    inlier_count = dsacstarsample.forward_rgb(
                        scene_coordinates_3HW.unsqueeze(0),
                        out_pose,
                        opt.hypotheses,
                        opt.threshold,
                        focal_length,
                        ppX,
                        ppY,
                        opt.inlieralpha,
                        opt.maxpixelerror,
                        network.OUTPUT_SUBSAMPLE,
                    )                    

                t5 = time.time()

                ########################################################################################

                # Calculate translation error.
                t_err = float(torch.norm(gt_pose_44[0:3, 3] - out_pose[0:3, 3]))

                # Rotation error.
                gt_R = gt_pose_44[0:3, 0:3].numpy()
                out_R = out_pose[0:3, 0:3].numpy()

                r_err = np.matmul(out_R, np.transpose(gt_R))
                # Compute angle-axis representation.
                r_err = cv2.Rodrigues(r_err)[0]
                # Extract the angle.
                r_err = np.linalg.norm(r_err) * 180 / math.pi

                # print("inlier_count = {}".format(inlier_count))
                # print("Rotation Error: {:.2f} deg, Translation Error: {:.1f} cm".format(r_err, 100*t_err))
                _logger.info(f"Rotation Error: {r_err:.2f}deg, Translation Error: {t_err * 100:.1f}cm")

                if ace_visualizer is not None:
                    ace_visualizer.render_reloc_frame(
                        query_pose=gt_pose_44.numpy(),
                        query_file=frame_path,
                        est_pose=out_pose.numpy(),
                        est_error=max(r_err, t_err*100),
                        sparse_query=opt.render_sparse_queries)

                # Save the errors.
                rErrs.append(r_err)
                tErrs.append(t_err * 100)

                # Check various thresholds.
                if r_err < 5 and t_err < 0.1:  # 10cm/5deg
                    pct10_5 += 1
                if r_err < 5 and t_err < 0.05:  # 5cm/5deg
                    pct5 += 1
                if r_err < 2 and t_err < 0.02:  # 2cm/2deg
                    pct2 += 1
                if r_err < 1 and t_err < 0.01:  # 1cm/1deg
                    pct1 += 1

                # Write estimated pose to pose file (inverse).
                out_pose = out_pose.inverse()

                # Translation.
                t = out_pose[0:3, 3]

                # Rotation to axis angle.
                rot, _ = cv2.Rodrigues(out_pose[0:3, 0:3].numpy())
                angle = np.linalg.norm(rot)
                axis = rot / angle

                # Axis angle to quaternion.
                q_w = math.cos(angle * 0.5)
                q_xyz = math.sin(angle * 0.5) * axis

                # Write to output file. All in a single line.
                pose_log.write(f"{frame_name} "
                               f"{q_w} {q_xyz[0].item()} {q_xyz[1].item()} {q_xyz[2].item()} "
                               f"{t[0]} {t[1]} {t[2]} "
                               f"{r_err} {t_err} {inlier_count}\n")

                t7 = time.time()
                dt1 = t2 - t1
                dt2 = t3 - t2
                dt3 = t4 - t3
                dt4 = t5 - t4
                sum_t = t5 - t1


            avg_batch_time += time.time() - batch_start_time
            num_batches += 1

    total_frames = len(rErrs)
    assert total_frames == len(testset)

    # Compute median errors.
    tErrs.sort()
    rErrs.sort()
    median_idx = total_frames // 2
    median_rErr = rErrs[median_idx]
    median_tErr = tErrs[median_idx]

    avg_rErr = sum(rErrs) / len(rErrs)
    avg_tErr = sum(tErrs) / len(tErrs)

    # Compute average time.
    avg_time = avg_batch_time / num_batches

    # Compute final metrics.
    pct10_5 = pct10_5 / total_frames * 100
    pct5 = pct5 / total_frames * 100
    pct2 = pct2 / total_frames * 100
    pct1 = pct1 / total_frames * 100

    _logger.info("===================================================")
    _logger.info("Test complete.")

    _logger.info('Accuracy:')
    _logger.info(f'\t10cm/5deg: {pct10_5:.1f}%')
    _logger.info(f'\t5cm/5deg: {pct5:.1f}%')
    _logger.info(f'\t2cm/2deg: {pct2:.1f}%')
    _logger.info(f'\t1cm/1deg: {pct1:.1f}%')

    _logger.info(f"Median Error: {median_rErr:.1f}deg, {median_tErr:.1f}cm")
    _logger.info(f"Average Error: {avg_rErr:.1f}deg, {avg_tErr:.1f}cm")
    _logger.info(f"Avg. processing time: {avg_time * 1000:4.1f}ms")

    # Write to the test log file as well.
    test_log.write(f"{median_rErr} {median_tErr} {avg_time}\n")
    test_log.write(f'\t10cm/5deg: {pct10_5:.1f}%\n')
    test_log.write(f'\t5cm/5deg: {pct5:.1f}%\n')
    test_log.write(f'\t2cm/2deg: {pct2:.1f}%\n')
    test_log.write(f'\t1cm/1deg: {pct1:.1f}%\n')
    test_log.write(f"Median Error: {median_rErr:.1f}deg, {median_tErr:.1f}cm\n")
    test_log.write(f"Average Error: {avg_rErr:.1f}deg, {avg_tErr:.1f}cm\n")
    test_log.write(f"Avg. processing time: {avg_time * 1000:4.1f}ms\n")


    test_log.close()
    pose_log.close()
