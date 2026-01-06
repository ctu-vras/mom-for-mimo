"""
Script to replay the retargeted movement.
Modified from https://gitlab.fel.cvut.cz/body-schema/motion-retargeting/-/blob/main/MIMo_implementation/motion_retargeting.py?ref_type=heads
"""

import gymnasium as gym
import mujoco.viewer
import numpy as np
import mimoEnv.utils as utils
import mujoco
from PIL import Image
import time
import mom_mimoEnv  # Importing the environments to use in gym.make
import mimoEnv  # Importing the environments to use in gym.make
import os
import inspect
from pathlib import Path

from motion_retargeting.handle_data import load_skeleton_wrapped, load_yaml
from motion_retargeting.vision import VisionHelper, CameraRecorder
from motion_retargeting.touch import PlotTouchHelper
from motion_retargeting.plot_motion_figure import PlotMotionFigure

from motion_retargeting.apply_mocap import ApplyMotion


OUT_VIDEO_FPS = 100

SAVE_TOUCH = True  # Save the touch video
SAVE_VISION = True  # Save the images from eyes and convert them into a video
SAVE_MIMO = True  # Save the wholde body video
LOAD_TSV = True  # Load the skeleton data from the tsv file (use when the .npy files are not generated)
PHYSICS_ON = True  # Have physics and collisions on 
LIST_GEOMS = False  # List geoms and bodies when needed
VIEW_SKELETON = False  # View plt skeleton with coordinate systems 
MOTION = True  # Run motion on the mocap sites
SAVE_MESH_SECTIONS = False


def initialize(init_dict, recording_name):
    scene_directory = os.path.abspath(os.path.join(__file__, "..", "mom_mimoEnv", "assets"))
    print(f"Processing the {recording} recording...........................................................................")
    data_folder = "motion_retargeting/data/" + recording_name  # Path to data directory with angles
    video_folder = data_folder + '/videos'
    Path(video_folder).mkdir(parents=True, exist_ok=True)
    site_names_file = os.path.join(scene_directory, "mocap", init_dict["mocap_sites"] + ".xml")
    scene_path = os.path.join(scene_directory, init_dict["scene"] + ".xml")

    view_orientation = list(map(int, init_dict["mimo_orientation"].split(" ")))

    return site_names_file, video_folder, data_folder, scene_path, view_orientation


def list_gb(env):
    for i in range(24):  # List geoms if needed
        geoms = utils.get_geoms_for_body(env.model, body_id=i)
        print(i, geoms)

    for j in range(24):  # List bodies if needed
        body_name = utils.get_body_id(env.model, body_id=j)
        print(j, body_name)


def run(site_names_file, video_folder, data_folder, scene_path, view_orientation):
    """ Creates the environment and then does, what the initialization flags indicate
    """
    env = gym.make("MIMoAdultRetargeting-v0", model_path=scene_path, width=1280, height=1280)
    # env.seed(42)
    _ = env.reset()

    if SAVE_VISION:
        vision_helper = VisionHelper(video_folder, env)

    if SAVE_MIMO:
        front_camera = CameraRecorder("frontal", video_folder, env)
    
    if LIST_GEOMS:
        list_gb(env)

    if LOAD_TSV:
        load_skeleton_wrapped(data_folder)

    if VIEW_SKELETON:
        plot_figure_holder = PlotMotionFigure(data_folder)

    if MOTION:
        motion = ApplyMotion(env, data_folder, site_names_file)
        chunk_range = range(motion.length)
    else:
        chunk_range = range(100)

    # chunk_range = range(100)
    if SAVE_TOUCH:
        # still_touch_plot = PlotTouchHelper(env=env, show_still_figure=True, in_touch_sensor_filename=sensor_init_file)
        touch_plot = PlotTouchHelper(env, video_folder, chunk_range[-1]+1, view_orientation=view_orientation, mesh_visualization=False)
        if SAVE_MESH_SECTIONS:
            touch_plot.heatmap_utilities.plot_sections()

    print(f"Running {chunk_range[-1]} frames")
    for line_count in chunk_range:
        if MOTION:
            motion.step(line_count)

        if PHYSICS_ON:
            mujoco.mj_step(env.model, env.data)
        else:
            mujoco.mj_forward(env.model, env.data)

        if SAVE_MIMO:
            front_camera.save_camera_snapshot(line_count)

        if SAVE_VISION:
            vision_helper.save_binocular_snapshot(line_count)
        
        if SAVE_TOUCH:
            touch_plot.view_touches(line_count)
            touch_plot.save_snapshot(line_count)
            # still_touch_plot.view_touches(env)
            # still_touch_plot.save_snapshot()
            
        if VIEW_SKELETON:
            plot_figure_holder.step(line_count)

        env.render()  # flush data

    if SAVE_TOUCH:
        print("Saving touch video")
        touch_plot.save_whole_touch_video(fps=OUT_VIDEO_FPS)
        touch_plot.save_timeseries()

    if SAVE_VISION:
        print("Saving vision videos")
        vision_helper.save_binocular_videos(fps=OUT_VIDEO_FPS)

    if SAVE_MIMO:
        print("Saving mimo video")
        front_camera.save_camera_videos(fps=OUT_VIDEO_FPS)

    print("Stop simulation")
    env.close()
    print("MuJoCo window closed")


if __name__ == "__main__":
    init_dict = load_yaml("motion_retargeting/run_config.yaml")
    for recording in init_dict:
        site_names_file, video_folder, data_folder, scene_path,view_orientation = initialize(init_dict[recording], recording)
        run(site_names_file, video_folder, data_folder, scene_path, view_orientation)
