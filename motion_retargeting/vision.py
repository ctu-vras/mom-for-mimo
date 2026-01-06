import cv2
import os
import numpy as np
import shutil


class VisionHelper:
    def __init__(self, data_path, env):
        self.env = env
        self.vision_video_path = data_path
        self.left_eye_name = "eye_left"
        self.right_eye_name = "eye_right"
        self.eye_img_path = os.path.join(data_path, "vision_temp")
        create_or_clear_folder(self.eye_img_path)

    def save_binocular_snapshot(self, specifier):
        """ Save current infant view from both eyes to appropriate temp folder

        Args:
            specifier (str): The string differentiating the file saved in this instance from others
        """
        save_camera_snapshot(self.env, self.eye_img_path, self.left_eye_name, specifier)
        save_camera_snapshot(self.env, self.eye_img_path, self.right_eye_name, specifier)


    def save_binocular_videos(self, fps=100):
        """ Save a video of both infant eyes

        Args:
            fps (int): Frames per second of the resulting videos
        """
        save_video_from_imgs(self.eye_img_path, self.vision_video_path, img_name_spec=self.left_eye_name, video_name="left", fps=fps, do_not_remove_dir=True)
        save_video_from_imgs(self.eye_img_path, self.vision_video_path, img_name_spec=self.right_eye_name, video_name="right", fps=fps)


class CameraRecorder:
    def __init__(self, camera_name, data_path, env):
        self.env = env
        self.camera_name = camera_name
        self.video_path = data_path
        self.cam_img_path = os.path.join(data_path, "camera_temp")
        create_or_clear_folder(self.cam_img_path)

    def save_camera_snapshot(self, specifier):
        """ Save current camera view to the buffer

        Args:
            specifier (str): The string differentiating the file saved in this instance from others
        """
        save_camera_snapshot(self.env, self.cam_img_path, self.camera_name, specifier)

    def save_camera_videos(self, fps=100):
        """ Save a video from the images from buffer

        Args:
            fps (int): Frames per second of the resulting videos
        """
        print("saving camera video")
        save_video_from_imgs(self.cam_img_path, self.video_path, img_name_spec="", fps=fps, video_name=self.camera_name)


def save_camera_snapshot(env, cam_img_path, camera_name, specifier):
    """ Save current camera view to the buffer

        Args:
            cam_img_path (str): The path to save the image to
            camera_name (str): The name of the MuJoCo camera
            specifier (str): The string differentiating the file saved in this instance from others
    """
    img = env.mujoco_renderer.render(render_mode="rgb_array", camera_name=camera_name)
    img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2BGR)
    filepath = os.path.join(cam_img_path, f"{camera_name}_{specifier:04}.png")
    cv2.imwrite(filepath, img)


def save_video_from_imgs(images_path, video_path, img_name_spec="", fps=100, video_name=None, do_not_remove_dir=False):
    """ Save the images from images to an .avi file in the video_path directory

    Args:
        images_path (str): Path to a folder where the images to be glued into the video are stored
        video_path (str): The path to the directory to store the resulting video in
        img_name_spec (str): The specific string differentiating the target images from other in the folder
        fps (int): Frames per second of the resulting video
        video_name (str): The name of the resulting video file
        do_not_remove_dir (Bool): Determins if the directories, where the images are stored should be removed or not
    """
    # Check if the video name is alright
    if video_name is None:
        video_name = "video.avi"
    if not video_name.endswith(".avi"):
        video_name = video_name + ".avi"

    images = [img for img in sorted(os.listdir(images_path)) if img.endswith(".png") and img.startswith(img_name_spec)]

    frame = cv2.imread(os.path.join(images_path, images[0]))
    height, width, _ = frame.shape

    # Init the video writer and write the video
    video = cv2.VideoWriter(os.path.join(video_path, video_name), 0, fps/4, (width,height))

    for i, image in enumerate(images):
        if i % 4 == 0:
            video.write(cv2.imread(os.path.join(images_path, image)))

    video.release()

    if not do_not_remove_dir:
        remove_directory(images_path)


def create_or_clear_folder(folder_path):
    """ Create a new folder, if the folder exists, delete it and create a new one

    Args:
        folder_path (str): Path of the folder to be updated
    """
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
    os.makedirs(folder_path)

def remove_directory(folder_path):
    """ Remove a folder and all subfolders and included files 

    Args:
        folder_path (str): Path of the folder to be removed completely
    """
    if os.path.exists(folder_path):
        for root, dirs, files in os.walk(folder_path, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))
        os.rmdir(folder_path)
        