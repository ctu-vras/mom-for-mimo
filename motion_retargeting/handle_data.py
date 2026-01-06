import numpy as np
import csv
from operator import itemgetter
import os
from scipy.spatial.transform import Rotation
import yaml


def load_data_from_tsv(filepath, info_period = 8):
    """ Load the motion data from the specified .tsv file and returns it
        This function was created for data exported from QTM with header and time information

    Args:
        filepath (str): The path to the .tsv file containing the skeleton motion data
    Returns:
        tuple containing
        - skeleton (dict: np.ndarray) - Dictionary with the quaternion data corresponding with each keypoint (dict: (time, dims))
        - names (list) - List of keypoint names, corresponds with dictionary keys of skeleton and s_position
        - nan_segments (list) - List of touples of indexes where a chunk of any data is missing
        - s_position (dict: np.ndarray) - Dictionary with the position data of all the keypoints (dict: (time, dims))
    """
    nan_segments = []
    s_positions = {}
    if filepath.endswith("_s_Q.tsv"):
        is_skeleton = True
        skeleton = {}
        data_dicts = (skeleton, s_positions, nan_segments)
    else:
        is_skeleton = False
        data_dicts = (s_positions, nan_segments)
    useful_data = False

    with open(filepath) as tsv_file:
        tsv_reader = csv.reader(tsv_file, delimiter="\t")
        # Go through lines in the .tsv file
        for line in tsv_reader:
            if useful_data:  # If this line contains the position and orientation data
                data_dicts = extract_data_from_line(line, info_period, data_dicts, keys, is_skeleton)

            # If this line contains the names of the keypoints and is the last line before useful data
            elif line[0] == 'Frame':
                useful_data = True
                names = list(itemgetter(*list(range(2, len(line) - 1, info_period)))(line))
                keys = []
                # Init the dictionaries
                for name in names:
                    if name.startswith("Q_"): name = name[2:]
                    if name.endswith(" X"): name = name[:-2]
                    if is_skeleton: skeleton[name] = []
                    s_positions[name] = []
                    keys.append(name)
    return keys, data_dicts


def extract_data_from_line(line, info_period, data_dicts, names, is_skeleton):
    """ Extract the information from a line based on the info period into the data_dicts
    
        Args:
            line (str): The line loaded from a file
            info_period (int): The period between two chunks of information to extract
            data_dicts (List[dicts]): The dictionaries holding the loaded data this line will be loaded into
            names (List[str]): The names of the markeres in the order from the loaded file
            is_skeleton (Bool): True, when the line contains skeleton data
    """
    if is_skeleton:
        data_dicts = extract_skeleton_from_line(line, info_period, data_dicts, names)
    else:
        data_dicts = extract_markers_from_line(line, info_period, data_dicts, names)

    return data_dicts


def extract_markers_from_line(line, info_period, data_dicts, names):
    """ Extract the marker information from a line based on the info period into the data_dicts
    
        Args:
            line (str): The line loaded from a file
            info_period (int): The period between two chunks of information to extract
            data_dicts (List[dicts]): The dictionaries holding the loaded data this line will be loaded into
            names (List[str]): The names of the markeres in the order from the loaded file
    """
    s_positions, nan_segments = data_dicts
    
    is_real = True
    for i, name in enumerate(names):  # Read the data an
        positions = list(map(float, line[i * info_period + 2:i * info_period + 5]))
        # Check for the nan segments
        if np.any(np.isnan(positions)):
            is_real = False
        # Write the loaded data to the dictionaries
        s_positions[name].append(positions)
    nan_segments.append(is_real)
    return s_positions, nan_segments


def extract_skeleton_from_line(line, info_period, data_dicts, names):
    """ Extract the skeleton information from a line based on the info period into the data_dicts
    
        Args:
            line (str): The line loaded from a file
            info_period (int): The period between two chunks of information to extract
            data_dicts (List[dicts]): The dictionaries holding the loaded data this line will be loaded into
            names (List[str]): The names of the markeres in the order from the loaded file
    """
    skeleton, s_positions, nan_segments = data_dicts
    
    is_real = True
    for i, name in enumerate(names):  # Read the data an
        orientation = list(map(float, line[i * info_period + 3 + 3:i * info_period + 3 + 7]))
        positions = list(map(float, line[i * info_period + 3:i * info_period + 6]))
        # Check for the nan segments
        if np.any(np.isnan(positions)):
            is_real = False
        # Write the loaded data to the dictionaries
        skeleton[name].append(orientation)
        s_positions[name].append(positions)
    nan_segments.append(is_real)

    return skeleton, s_positions, nan_segments


def save_to_npy(data, filepath):
    """ Save data to a .npy file with the specified path
    
        Args:
            filepath (str): The file path to store the data to
    """
    np.save(filepath, data)


def load_file(filename, is_dict=False):
    """ Load data from a .npy file with the specified path, if it is a dictionary, unwrap it
    
        Args:
            filepath (str): The file path to load the data from
            is_dict (bool): Flags if the file contains a dictionary to load it correctly
            
        Returns:
            read_data (np.array|dict): The loaded data
    """
    read_data = np.load(filename, allow_pickle=True)
    if is_dict:
        read_data = read_data.item()
    return read_data


def load_skeleton_wrapped(data_folder, out_folder=None):
    """ Load a .tsv skeleton file and save the data in .npy files
        saves
            - skeleton file with the quaterninon orientation dictionary
            - names file with a list of names of skeleton keypoints
            - positions file with a positions dictionary
            - local euler angle orientations (x,y,z) 
    
        Args:
            data_folder (str): The directory path with a skeleton .tsv file
            out_folder (str): The directory path to store the .npy files in. If None, save the file in the data_folder
    """
    if out_folder is None:
        out_folder = data_folder

    skeleton_tsv_name = find_skeleton_in_directory(data_folder)

    names, data = load_data_from_tsv(skeleton_tsv_name)

    marker_tsv_name = skeleton_tsv_name[:-8] + ".tsv"
    _, m_data = load_data_from_tsv(marker_tsv_name, info_period=3)

    skeleton, s_positions, _ = data
    m_positions, _ = m_data

    save_to_npy(skeleton, get_skeleton_filepath(out_folder))
    save_to_npy(names, get_labels_filepath(out_folder))
    save_to_npy(s_positions, get_skeleton_position_filepath(out_folder))
    save_to_npy(m_positions, get_mocap_position_filepath(out_folder))

    save_global_to_local_coordinates(out_folder)


def find_skeleton_in_directory(directory):
    """ Find the first skeleton file ending with s_Q.tsv is the specified directory
    
        Args:
            directory (str): The directory path to find a .tsv skeleton file
        Returns:
            file_path (str): The path to the skeleton file
        Raises:
            Exception: If no skeleton file found in the directory
    """
    for file in os.listdir(directory):
        if file.endswith("s_Q.tsv"):
            return os.path.join(directory, file)
        
    raise Exception(f"Oh no, skeleton file not found in this directory: {directory}")


def get_skeleton_filepath(directory_path):
    """" Returns the path to a quaternion skeleton file
        Args:
            directory_path (str): Path to the directory of the file
        Returns:
            str: Path to a quaternion skeleton file
    """
    return os.path.join(directory_path, "quaternion_skeleton.npy")


def get_labels_filepath(directory_path):
    """" Returns the path to a skeleton labels file
        Args:
            directory_path (str): Path to the directory of the file
        Returns:
            str: Path to a skeleton labels file
    """
    return os.path.join(directory_path, "skeleton_labels.npy")


def get_skeleton_angles_filepath(directory_path):
    """" Returns the path to an euler angles skeleton file
        Args:
            directory_path (str): Path to the directory of the file
        Returns:
            str: Path to an euler angles skeleton file
    """
    return os.path.join(directory_path, "euler_skeleton.npy")


def get_skeleton_position_filepath(directory_path):
    """" Returns the path to a positions skeleton file
        Args:
            directory_path (str): Path to the directory of the file
        Returns:
            str: Path to a positions skeleton file
    """
    return os.path.join(directory_path, "skeleton_position_data.npy")


def get_mocap_position_filepath(directory_path):
    """" Returns the path to a mocap positions file
        Args:
            directory_path (str): Path to the directory of the file
        Returns:
            str: Path to a mocap positions file
    """
    return os.path.join(directory_path, "mocap_position_data.npy")


def get_data_directory(specific_folder):
    """" Returns the path to a data directory in "motion_retargeting/data/"
        Args:
            specific_folder (str): Name of a folder in "motion_retargeting/data/"
        Returns:
            str: Path to the specific folder
    """
    return os.path.join("motion_retargeting/data/", specific_folder)


def get_body_tree():
    """" Returns the body tree of a Qualisys Sports Markerset skeleton"
        Returns:
            list: List of lists of the body tree. The base is 'Hips'
    """
    spine = ['Hips', 'Spine', 'Spine1', 'Spine2', 'Neck', 'Head']
    left_arm = ['Spine2', 'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftForeArmRoll', 'LeftHand']
    right_arm = ['Spine2', 'RightShoulder', 'RightArm', 'RightForeArm', 'RightForeArmRoll', 'RightHand']
    left_leg = ['Hips', 'LeftUpLeg', 'LeftLeg', 'LeftFoot', 'LeftToeBase']
    right_leg = ['Hips', 'RightUpLeg', 'RightLeg', 'RightFoot', 'RightToeBase']

    body = [spine, left_arm, right_arm, left_leg, right_leg]

    return body


def save_global_to_local_coordinates(folder):
    """" Converts the global quaternion orientations to local euler angles (x,y,z)

    First, load skeleton and label data from npy files in the 'folder' directory.
    The angle coordinate system is changed from global (as exported from qtm) to local
        following the structure, defined in 'body' 

    Then the quaternion representation is replaced by euler angle notation.

    Last, the new skeleton is saved in a npy file 

    Args:
        folder (str): Folder containing the quaternion skeleton data. The converted data will also be stored here

    """

    body = get_body_tree()
    
    labels_filename = get_labels_filepath(folder)
    skeleton_filename = get_skeleton_filepath(folder)
    euler_skeleton_filename = get_skeleton_angles_filepath(folder)

    skeleton = load_file(skeleton_filename, is_dict=True)
    names = load_file(labels_filename)

    new_skeleton = {}

    for name in names:
        new_skeleton[name] = []

    for q in skeleton['Hips']:
        new_skeleton['Hips'].append(Rotation.as_euler(Rotation.from_quat(q), "xyz"))

    for body_chunk in body:
        for i in range(len(body_chunk)-1):
            base_label = body_chunk[i]
            this_label = body_chunk[i+1]
            base_qs = skeleton[base_label]
            this_qs = skeleton[this_label]
            for j in range(len(base_qs)):
                base_quat = base_qs[j]
                this_quat = this_qs[j]
                base_rot_mat = Rotation.from_quat(base_quat)
                this_global_rot_mat = Rotation.from_quat(this_quat)
                this_local_rot_mat = base_rot_mat.inv()*this_global_rot_mat
                these_local_angles = Rotation.as_euler(this_local_rot_mat, "xyz")
                new_skeleton[this_label].append(these_local_angles)

    save_to_npy(new_skeleton, euler_skeleton_filename)


def load_mocap_to_mimo_names():
    """ Load a dictionary of mocap to MIMo mom names from a .yaml config file
        Returns:
            dict: Dictionary of mocap to MIMo mom names
    """
    with open('motion_retargeting/names.yaml', 'r') as f:
        data = yaml.load(f, Loader=yaml.SafeLoader)
    return data

def load_yaml(filepath):
    """ Load a dictionary from a .yaml config file
        Returns:
            dict: Loaded dictionary
    """
    with open(filepath, 'r') as f:
        data = yaml.load(f, Loader=yaml.SafeLoader)
    return data

def load_site_name_id_dict():
    filename = "/home/noemi/mom-for-mimo/mom_mimoEnv/assets/mocap/name_id_dict.npy"
    return load_file(filename, is_dict=True)

def save_site_name_id_dict(data):
    filename = "/home/noemi/mom-for-mimo/mom_mimoEnv/assets/mocap/name_id_dict.npy"
    save_to_npy(data, filename)

if __name__ == "__main__":
    print("Nothing is happening here :(")




