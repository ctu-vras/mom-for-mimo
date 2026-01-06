import xml.etree.ElementTree as ET

import mimoEnv  # Importing the environments to use in gym.make
import mujoco
import numpy as np
import yaml

from xml_work.get_capsule_sizes import get_body_and_geom_data
from motion_retargeting.handle_data import load_mocap_to_mimo_names

def edit_range_to_unlimited(root):
    # finding the joint tag and their child attributes.
    for joint in root.iter('joint'):
        range = joint.set('range', '-180 180')
        range = joint.get('range')
        name = joint.get('name')


def edit_sizes(root):
    for geom in root.iter('geom'):
        size = geom.get('size')
        name = geom.get('name')
        if name is None or geom.get('type')!="capsule": continue
        print(name, size)


def get_names_dict(root):
    names = {}
    i = 0
    for body in root.iter('body'):
        name = body.get('name')
        if name is None: continue
        names[name] = i
        i += 1
    return names


def get_names_list(root):
    names = []
    for body in root.iter('body'):
        name = body.get('name')
        if name is None: continue
        names.append(name)
    return names


def load_root_from_file(file_name):
    tree = ET.parse(file_name)
    root = tree.getroot()
    return root


def get_names_from_xml(file_name, dict_out=False):
    root = load_root_from_file(file_name)
    if dict_out:
        names = get_names_dict(root)
    else:
        names = get_names_list(root)
    return names


def generate_mocap_xml(out_file_name, body_names, positions):
    root = ET.Element("body")

    for name in body_names:
        pos = ' '.join(map(str, positions[name]))

        m_name = name[2:]

        new_body = ET.SubElement(root, "body", name=m_name, mocap="true", pos=pos)
        new_site = ET.SubElement(new_body, "site", name=m_name, type="sphere", size="0.01", material="mocap")
    
    tree = ET.ElementTree(root)

    ET.indent(tree, space='\t', level=0)
    
    with open (out_file_name, "wb") as files:
        tree.write(files)


def generate_mocap_xml_no_sites(out_file_name, body_names, positions):
    root = ET.Element("body")

    for name in body_names:
        pos = ' '.join(map(str, positions[name]))

        m_name = name[2:]

        new_body = ET.SubElement(root, "body", name=m_name, mocap="true", pos=pos)
    
    tree = ET.ElementTree(root)

    ET.indent(tree, space='\t', level=0)

    with open (out_file_name, "wb") as files:
        tree.write(files)


def remove_mocap_sites(file_name):
    root = load_root_from_file(file_name)
    # Source - https://stackoverflow.com/a
    # Posted by iceblueorbitz
    # Retrieved 2025-12-08, License - CC BY-SA 3.0

    def iterator(parents, nested=False):
        for child in reversed(parents):
            name = child.get("name")
            if nested:
                if len(child) >= 1:
                    iterator(child, nested=True)
            if name is not None and name.startswith("a_mocap"):  # Add your entire condition here
                parents.remove(child)

    iterator(root, nested=True)

    tree = ET.ElementTree(root)
    with open (file_name, "wb") as files:
        tree.write(files)


def generate_mocap_equality_xml(out_file_name):
    """ example of running this function:
            generate_mocap_equality_xml("xml_work/test_equality.xml", get_names_from_xml('/home/noemi/mom-for-mimo/mom_mimoEnv/assets/adult/adult_military_model.xml'))
    """
    root = ET.Element("equality")

    with open('motion_retargeting/equality.yaml', 'r') as f:
        names_dict = yaml.load(f, Loader=yaml.SafeLoader)

    for m_name in names_dict:
        a_name = names_dict[m_name]

        new_weld = ET.SubElement(root, "weld", body1=a_name, body2=m_name, solref="0.01 1", solimp=".95 .99 .0001")
    
    tree = ET.ElementTree(root)

    ET.indent(tree, space='\t', level=0)
    
    with open (out_file_name, "wb") as files:
        tree.write(files)



def get_positions(model, data, names):
    positions = {}
    # Do forward kinematics
    mujoco.mj_kinematics(model, data)

    for body in names:
        # Get the ID of the body we want to track
        body_id = model.body(body).id

        # Get the position of the body from the data
        body_pos = data.xpos[body_id]
        body_quat = data.xquat[body_id]
        positions[body] = (body_pos, body_quat)
    return(positions)


def generate_mocap_xml_example():
    # Load the mujoco model basic.xml
    model = mujoco.MjModel.from_xml_path('mom_mimoEnv/assets/testing.xml')
    data = mujoco.MjData(model)

    names = get_names_from_xml("/home/noemi/mom-for-mimo/mom_mimoEnv/assets/adult/adult_military_model.xml")
    positions = get_positions(model, data, names)
    generate_mocap_xml("xml_work/test_w_pos.xml", names, positions)


def create_model_with_new_params(file_name, simple=False):
    root = load_root_from_file(file_name)
    body_data, geom_data = get_body_and_geom_data(simple=simple)
    for body in root.iter('body'):
        name = body.get('name')
        if name in body_data.keys():
            pos = body_data[name]
            body.set("pos", pos)
    for geom in root.iter('geom'):
        name = geom.get('name')
        if name in geom_data.keys():
            size, mass, pos = geom_data[name]
            geom.set("size", size)
            geom.set("pos", pos)
            geom.set("mass", mass)

    tree = ET.ElementTree(root)

    ET.indent(tree, space='\t', level=0)
    
    out_file_name = file_name[:-4] + "_new.xml"
    with open (out_file_name, "wb") as files:
        tree.write(files)
    return out_file_name


def save_a_norange_version(model_path):
    # parsing from the string.
    tree = ET.parse(model_path)
    root = tree.getroot()

    edit_range_to_unlimited(root)
    out_path = model_path[:-4] + "_norange.xml"

    tree.write(out_path)


def generate_sites_with_positions(pos_dict_path, out_file_name, pos_dict_path2, mocap_mimo_names_path):
    read_data = np.load(pos_dict_path, allow_pickle=True)
    read_data = read_data.item()    
    read_data2 = np.load(pos_dict_path2, allow_pickle=True)
    read_data2 = read_data2.item()

    read_data = read_data | read_data2

    names = load_mocap_to_mimo_names()

    positions = {key: np.array(read_data[key][0]) / 1000 for key in names}

    root = ET.Element("body")

    for name in names:
        pos = ' '.join(map(str, positions[name]))

        name = names[name][2:]
        new_body = ET.SubElement(root, "body", name=name, mocap="true", pos=pos)
        if "bb" in name:
            material = "mocap_bb"
        else:
            material = "mocap"
        new_site = ET.SubElement(new_body, "site", type="sphere", size="0.01", material=material)
    
    tree = ET.ElementTree(root)

    ET.indent(tree, space='\t', level=0)
    
    with open (out_file_name, "wb") as files:
        tree.write(files)

def try_mocap_gen():
    path = "/home/noemi/mom-for-mimo/motion_retargeting/data/mom/skeleton_position_data.npy"
    path2 = "/home/noemi/mom-for-mimo/motion_retargeting/data/mom/mocap_position_data.npy"
    out_path = "/home/noemi/mom-for-mimo/mom_mimoEnv/assets/mocap/baseline.xml"
    mocap_mimo_names_path = "/home/noemi/mom-for-mimo/motion_retargeting/names.yaml"
    generate_sites_with_positions(path, out_path, path2, mocap_mimo_names_path)


if __name__ == "__main__":
    
    try_mocap_gen()
    # xml_path = "/home/noemi/mom-for-mimo/mom_mimoEnv/assets/adult/adult_military_simple.xml"
    # save_a_norange_version(xml_path)
    # get_names_from_xml("/home/noemi/mom-for-mimo/mom_mimoEnv/assets/adult/adult_military_simple.xml")
    # create_model_with_new_params(xml_path)
    # generate_mocap_equality_xml("xml_work/test_equality.xml", get_names_from_xml(xml_path))
    # print("WARNING!!! to get mocap keypoints, we must run mocap key test")