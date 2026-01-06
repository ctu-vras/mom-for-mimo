import mujoco
import numpy as np
import os
import mom_mimoEnv  # importing the environments to use in gym.make
import mimoEnv  # importing the environments to use in gym.make
import xml_work.xml_utils as xu
from motion_retargeting.handle_data import load_mocap_to_mimo_names, save_site_name_id_dict


def get_positions(model, data, names):
    positions = {}
    # Do forward kinematics
    mujoco.mj_kinematics(model, data)

    for site in names:
        # Get the ID of the site we want to track
        site_id = model.site(site).id

        # Get the position of the site from the data
        site_pos = data.site_xpos[site_id]
        positions[site] = site_pos
    return positions

def generate_and_save_site_name_id_dict(model, names):
    ids = {}
    for site in names:
        # Get the ID of the site we want to track
        site_id = model.site(site).id

        ids[site] = site_id
    save_site_name_id_dict(ids)
    return ids

def generate_mocap_points(mocap_sites_path, mocap_bodies_path):
    # Load the mujoco model basic.xml
    model = mujoco.MjModel.from_xml_path('mom_mimoEnv/assets/generation.xml')
    data = mujoco.MjData(model)

    names = load_mocap_to_mimo_names().values()
    positions = get_positions(model, data, names)
    generate_and_save_site_name_id_dict(model, names)
    xu.generate_mocap_xml(mocap_sites_path, names, positions)
    xu.generate_mocap_xml_no_sites(mocap_bodies_path, names, positions)


if __name__ == "__main__":
    SPECIFIER = "_mom1"
    xml_path = "/home/noemi/mom-for-mimo/mom_mimoEnv/assets/adult/mom/hands_down/mom.xml"
    simple = False
    mocap_sites_path = "/home/noemi/mom-for-mimo/mom_mimoEnv/assets/mocap/mocap_sites"+SPECIFIER+".xml"
    mocap_bodies_path = "/home/noemi/mom-for-mimo/mom_mimoEnv/assets/mocap/mocap_bodies"+SPECIFIER+".xml"
    equality_path = "/home/noemi/mom-for-mimo/mom_mimoEnv/assets/mocap/mocap_equalities.xml"
    # create a v2 version and a norange version of it
    new_filepath = xu.create_model_with_new_params(xml_path, simple=simple)
    xu.save_a_norange_version(new_filepath)
    # create a simple version and a norange version of it
    folder_path, filename = os.path.split(xml_path)
    xml_path_simple = os.path.join(folder_path, filename[:-4] + "_simple.xml")
    new_filepath = xu.create_model_with_new_params(xml_path_simple, simple=True)
    xu.save_a_norange_version(new_filepath)
    # generate new mocap points from generation.xml scene
    generate_mocap_points(mocap_sites_path, mocap_bodies_path)
    # generate mocap equality from the equality.yaml file
    xu.generate_mocap_equality_xml(equality_path)
    print("All files generated.")

