import numpy as np
import os
import mujoco

from motion_retargeting.handle_data import get_skeleton_position_filepath, load_file, \
    load_mocap_to_mimo_names, get_mocap_position_filepath, load_mocap_to_mimo_names, \
    save_to_npy
from xml_work.xml_utils import get_names_from_xml
import mimoEnv.utils as mimo_utils

class ApplyMotion:

    def __init__(self, env, data_directory, site_names_file):
        self.env = env
        self.data_directory = data_directory

        # Load necessary data from files in the data_directory
        filename = get_skeleton_position_filepath(data_directory)
        skeleton_positions = load_file(filename, is_dict=True)
        mimo_site_names = get_names_from_xml(site_names_file, dict_out=True)
        filename = get_mocap_position_filepath(data_directory)
        mocap_positions = load_file(filename, is_dict=True)

        # Merge the two dictionaries for easy access
        positions = skeleton_positions | mocap_positions

        names_dict = load_mocap_to_mimo_names()

        self.positions = []
        self.names = []
        self.body_indexes = []

        self.head_index = None

        # Order the position and names into lists corresponding with the mocap input
        for mocap_name in names_dict.keys():
            mimo_name = names_dict[mocap_name]
            if mimo_name[2:] not in mimo_site_names:
                print("This is not here... ", names_dict[mocap_name])
                continue
            self.positions.append(positions[mocap_name])
            self.names.append(mimo_name)
            self.body_indexes.append(mimo_site_names[mimo_name[2:]])

        # Save the ordered positions as np arrays ordered (time, dims, keypoints)
        self.positions = np.array(self.positions).transpose((1,0,2))
        self.length = self.positions.shape[0]
        
        # Set the offset to move the mocap figure to match the position of the MIMo model 
        mimo_hips = np.array((0.0, 0.0, 1.0))  
        mocap_hips = np.array(skeleton_positions["Hips"][0])/1000
        self.offset = mimo_hips-mocap_hips # This is the hip placement

        self.generate_init_position()
        # self.init_position()

    def init_position(self):
        """ Set models into the pre-determined init_position"""
        filepath = os.path.join(self.data_directory, "init_pose.npy")
        initial_qpos = load_file(filepath, is_dict=True)
        if initial_qpos:
            for joint_name in initial_qpos:
                mimo_utils.set_joint_qpos(self.env.model, self.env.data, joint_name, initial_qpos[joint_name])
        mujoco.mj_step(self.env.model, self.env.data)
        input()

    def generate_init_position(self, first_frame=3):
        """" Sets the initial positions and wait for a 100 steps for the models to settle.
            Then capture and save the values of the joints of the models to recreate later.
            Args:
                first_frame (int): The frame to generate the init position from
        """
        for i in range(100):
            for body_pos, body_index, name in zip(self.positions[first_frame], self.body_indexes, self.names):
                # Convert mm to m and move to match the model position
                if name.startswith("a_mocap_bb"):
                    self.env.data.mocap_pos[body_index] = body_pos/1000 + self.offset
                else:
                    self.env.data.mocap_pos[body_index] = body_pos/1000 + self.offset

            mujoco.mj_step(self.env.model, self.env.data)

            self.env.render()  # flush data

        joint_ids = _get_joints(self.env)
        joint_dict = {self.env.data.joint(joint_id).name: joint_id for joint_id in joint_ids}
        joint_values = {}
        for joint_name in joint_dict:
            joint_index = mimo_utils.get_joint_qpos_addr(self.env.model, joint_dict[joint_name])
            joint_values[joint_name] = self.env.data.qpos[joint_index]

        filepath = os.path.join(self.data_directory, "init_pose.npy")
        save_to_npy(joint_values, filepath)

    def step(self, i):
        """" Move the mocap sites into positions in frame i
            Args:
                i (int): The movement frame
        """
        for body_pos, body_index in zip(self.positions[i], self.body_indexes):
            # Convert mm to m and move to match the model position
            if np.any(np.isnan(body_pos)):  # Do not set nan value, just keep the last
                continue
            self.env.data.mocap_pos[body_index] = body_pos/1000 + self.offset


def _get_joints(env):
    """ Returns the IDs of the joints associated with MIMO in :attr:`.mimo_joints`.
    """
    joints = []
    for i in range(env.model.njnt):
        joint_name = env.model.joint(i).name
        if joint_name.startswith("robot:"):
            joints.append(i)
    joints = np.asarray(joints)

    return joints