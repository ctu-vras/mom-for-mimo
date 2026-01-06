import numpy as np
import yaml

from xml_work.get_mass import get_mass_dict


def get_capsule_size(capsule_diameter, capsule_len, is_body=False, is_upper_arm=False, bc_radius=None):
    capsule_radius = capsule_diameter / 2
    capsule_half_len = capsule_len / 2
    if is_body:
        cylinder_halflength = capsule_half_len - capsule_radius
    elif is_upper_arm:
        cylinder_halflength = capsule_half_len - capsule_radius/2
    elif bc_radius is not None:
        cylinder_halflength = capsule_half_len + bc_radius/2
    else:
        cylinder_halflength = capsule_half_len

    capsule_radius = np.round(capsule_radius, 4)
    cylinder_halflength = np.round(cylinder_halflength, 4)
    return [capsule_radius, cylinder_halflength]

def get_foot_size(r1, l1, r2, r3):
    foot_breadth = 2 * r1
    ball_length = 0.182
    foot_len = 0.246
    box = 0.056999999999999995
    r1 = foot_breadth / 2
    l1 = (ball_length - r1) / 2
    print("size:", r1, l1, r2, r3)
    print("position:", 0, l1, 2*l1, foot_len - r1 - box- r3)
    return r1 + 2 *l1 + r2 + 2*r3


def load_sizes_yaml():
    with open('xml_work/my_sizes.yaml', 'r') as f:
        data = yaml.load(f, Loader=yaml.SafeLoader)
    return data


def string_one(a):
    if isinstance(a, np.ndarray) or isinstance(a, list):
        out = ' '.join(map(str, a))
    else:
        out = str(a)
    return out


def string_triple(a, b, c):        
    return [string_one(a), string_one(b), string_one(c)]


def get_body_and_geom_data(simple=False):
    loaded_data = load_sizes_yaml()
    mass_dict = get_mass_dict(loaded_data["mass_kg"])

    geom_data = {}  # size, mass, position data
    body_data = {}  # position data

    # torso
    torso_len = loaded_data["torso"]["l"]
    torso_capsule_names = ["lb", "cb", "ub1", "ub2", "shoulders"]
    summed_diameters = sum([loaded_data[name]["h"] for name in torso_capsule_names])
    overlap = summed_diameters - torso_len
    
    num_of_torso_capsules = len(torso_capsule_names)
    torso_offset = overlap/(num_of_torso_capsules-1)
    
    # body (bottom to top)
    s_bottom = get_capsule_size(loaded_data["lb"]["h"], loaded_data["lb"]["w"], is_body=True)
    m_bottom = mass_dict["body5"]
    p_bottom = [0, 0, 0]
    body_data["a_mimo"] = string_one([0, 0, 0])
    geom_data["a_lb"] = string_triple(s_bottom, m_bottom, p_bottom)

    # center body

    s_cb = get_capsule_size(loaded_data["cb"]["h"], loaded_data["cb"]["w"], is_body=True)
    m_cb = mass_dict["body4"]
    p_cb = [0, 0, 0]
    z_cb = s_bottom[0] + s_cb[0] - torso_offset
    body_data["a_lower_body"] = string_one([0, 0, z_cb])
    geom_data["a_cb"] = string_triple(s_cb, m_cb, p_cb)


    # upper body 

    s_ub1 = get_capsule_size(loaded_data["ub1"]["h"], loaded_data["ub1"]["w"], is_body=True)
    m_ub1 = mass_dict["body3"]
    p_ub1 = [0,0,0]
    z_ub1 = s_ub1[0] + s_cb[0] - torso_offset
    body_data["a_upper_body"] = string_one([0, 0, z_ub1])
    geom_data["a_ub1"] = string_triple(s_ub1, m_ub1, p_ub1)

    s_ub2 = get_capsule_size(loaded_data["ub2"]["h"], loaded_data["ub2"]["w"], is_body=True)
    m_ub2 = mass_dict["body2"]
    z_ub2 = s_ub1[0] + s_ub2[0] - torso_offset
    p_ub2 = [0,0,z_ub2]
    geom_data["a_ub2"] = string_triple(s_ub2, m_ub2, p_ub2)

    # breasts
    r_breasts = loaded_data["breasts"]["d"] / 2
    gap_breasts = loaded_data["breasts"]["w"]
    breast_z_torso_position = loaded_data["breasts"]["h"]
    m_breasts = 0
    breast_x_offset = s_ub2[0]
    breast_y_offset = gap_breasts/2 + r_breasts
    breast_z_offset = breast_z_torso_position - z_cb - z_ub1 - s_bottom[0]
    p_l_breast = [breast_x_offset, breast_y_offset, breast_z_offset]
    p_r_breast = [breast_x_offset, -breast_y_offset, breast_z_offset]
    p_uniboob = [breast_x_offset, 0, breast_z_offset]
    s_uniboob = get_capsule_size(1.8 * r_breasts, gap_breasts + 2*r_breasts)

    geom_data["a_left_boob"] = string_triple(r_breasts, m_breasts, p_l_breast)
    geom_data["a_right_boob"] = string_triple(r_breasts, m_breasts, p_r_breast)
    geom_data["a_uni_boob"] = string_triple(s_uniboob, m_breasts, p_uniboob)


    # shoulders
    s_shoulder = get_capsule_size(loaded_data["shoulders"]["h"], loaded_data["shoulders"]["w"], is_body=True)
    s_shoulder[1] /= 2  # because shoulders are split into two capsules
    m_shoulder = mass_dict["shoulder"]
    left_p_shoulder = [0, s_shoulder[1], 0]
    right_p_shoulder = [0, -s_shoulder[1], 0]

    shoulder_hike = s_shoulder[0] + s_ub2[0] - torso_offset + z_ub2
    body_data["a_left_shoulder_help"] = string_one([0, 0, shoulder_hike])
    body_data["a_right_shoulder_help"] = string_one([0, 0, shoulder_hike])
    geom_data["a_right_shoulder_helper"] = string_triple(s_shoulder, m_shoulder, right_p_shoulder)
    geom_data["a_left_shoulder_helper"] = string_triple(s_shoulder, m_shoulder, left_p_shoulder)


    # neck

    s_neck = get_capsule_size(loaded_data["neck"]["w"], loaded_data["neck"]["h"])    
    m_neck = mass_dict["neck"]
    p_neck = [0,0,2*s_neck[1]]
    neck_hike = shoulder_hike + s_shoulder[0] - s_neck[1]
    body_data["a_neck"] = string_one([0, 0, neck_hike])
    geom_data["a_neck"] = string_triple(s_neck, m_neck, p_neck)

    # head
    head_r = loaded_data["head"]["d"] / 2 
    head_top_r = loaded_data["head"]["d_top"] / 2
    head_heigth = loaded_data["head"]["h"]
    m_head = mass_dict["head_up"]
    p_head = [0, 0, head_r]
    m_head_top = mass_dict["head_down"]
    p_head_top = [-0.01, 0, head_heigth - head_top_r]

    body_data["a_head"] = string_one([0, 0, 2*s_neck[1]])
    geom_data["a_head"] = string_triple(head_r, m_head, p_head)
    geom_data["a_headtop"] = string_triple(head_top_r, m_head_top, p_head_top)
    

    # upper arm

    s_upper_arm = get_capsule_size(loaded_data["upper arm"]["w"], loaded_data["upper arm"]["h"], is_upper_arm=True)
    m_upper_arm = mass_dict["upper_arm"]
    p_upper_arm = [0, 0, s_upper_arm[1]]
    shoulder_offset = s_shoulder[0] + 2* s_shoulder[1] + s_upper_arm[0]
    body_data["a_right_upper_arm"] = string_one([0, -shoulder_offset, 0])
    body_data["a_left_upper_arm"] = string_one([0, shoulder_offset, 0])
    geom_data["a_right_uarm1"] = string_triple(s_upper_arm, m_upper_arm, p_upper_arm)
    geom_data["a_left_uarm1"] = string_triple(s_upper_arm, m_upper_arm, p_upper_arm)

    # lower arm
    s_lower_arm = get_capsule_size(loaded_data["lower arm"]["w"], loaded_data["lower arm"]["h"])
    m_lower_arm = mass_dict["lower_arm"]
    p_lower_arm = [0, 0, s_lower_arm[1]]

    body_data["a_right_lower_arm"] = string_one([0, 0, 2*s_upper_arm[1]])
    body_data["a_left_lower_arm"] = string_one([0, 0, 2*s_upper_arm[1]])
    geom_data["a_right_larm"] = string_triple(s_lower_arm, m_lower_arm, p_lower_arm)
    geom_data["a_left_larm"] = string_triple(s_lower_arm, m_lower_arm, p_lower_arm)

    # hands
    magic = 0.04 if not simple else 0  # This magic constant handles the hand inconsistency of the models
    sr_hand = [0.008, -0.01, 2*s_lower_arm[1]+s_lower_arm[0]+magic]  
    sl_hand = [0.008, 0.01, 2*s_lower_arm[1]+s_lower_arm[0]+magic]  
    body_data["a_right_hand"] = string_one(sr_hand)
    body_data["a_left_hand"] = string_one(sl_hand)

    if simple:
        s_palm = [loaded_data["hand"]["palm_w"]/2, loaded_data["hand"]["palm_h"]/2, loaded_data["hand"]["palm_l"]/2]
        m_palm = mass_dict["hand"] / 2
        p_palm = [0, 0, s_palm[2]]

        s_palm1 = [s_palm[1], s_palm[0]]
        m_palm1 = 0
        p_palm1 = [0, 0, 2*s_palm[2]]
        
        geom_data["a_left_hand1"] = string_triple(s_palm, m_palm, p_palm)
        geom_data["a_left_hand2"] = string_triple(s_palm1, m_palm1, p_palm1)
        geom_data["a_right_hand1"] = string_triple(s_palm, m_palm, p_palm)
        geom_data["a_right_hand2"] = string_triple(s_palm1, m_palm1, p_palm1)
        

        s_fingers2 = [loaded_data["hand"]["palm_h"]/2, loaded_data["hand"]["hand_b"]/2]
        s_fingers = [s_fingers2[1], s_palm[1], loaded_data["hand"]["finger_l"]/2 - s_fingers2[0]]

        m_fingers = mass_dict["hand"] / 2
        p_fingers = [0, 0, s_fingers[2]]
        
        m_fingers2 = 0
        p_fingers2 = [0, 0, 2*s_fingers[2]]

        geom_data["a_left_fingers1"] = string_triple(s_fingers, m_fingers, p_fingers)
        geom_data["a_left_fingers2"] = string_triple(s_fingers2, m_fingers2, p_fingers2)

        geom_data["a_right_fingers1"] = string_triple(s_fingers, m_fingers, p_fingers)
        geom_data["a_right_fingers2"] = string_triple(s_fingers2, m_fingers2, p_fingers2)

        body_data["a_left_fingers"] = string_one([s_fingers[0]-s_palm[0], 0, p_palm1[2]])
        body_data["a_right_fingers"] = string_one([s_fingers[0]-s_palm[0], 0, p_palm1[2]])



    # upper legs

    s_upper_leg = get_capsule_size(loaded_data["upper leg"]["w"], loaded_data["upper leg"]["h_out"], is_upper_arm=True)
    m_upper_leg = mass_dict["upper_leg"] 
    p_upper_leg = [0, 0, -s_upper_leg[1]]
    upper_leg_y_offset = loaded_data["upper leg"]["gap"]/2 + s_upper_leg[0]
    upper_leg_z_offset = s_bottom[0] - (loaded_data["upper leg"]["h_out"] - loaded_data["upper leg"]["h_in"])

    body_data["a_right_upper_leg"] = string_one([0, -upper_leg_y_offset, -upper_leg_z_offset])
    body_data["a_left_upper_leg"] = string_one([0, upper_leg_y_offset, -upper_leg_z_offset])
    geom_data["a_right_uleg"] = string_triple(s_upper_leg, m_upper_leg, p_upper_leg)
    geom_data["a_left_uleg"] = string_triple(s_upper_leg, m_upper_leg, p_upper_leg)

    # lower legs

    s_lower_leg = get_capsule_size(loaded_data["lower leg"]["w"], loaded_data["lower leg"]["h"])
    m_lower_leg = mass_dict["lower_leg"] 
    p_lower_leg = [0, 0, -s_lower_leg[1]]
    lower_leg_z_offset = [0, 0, -2*s_upper_leg[1]-s_upper_leg[0]]

    body_data["a_right_lower_leg"] = string_one(lower_leg_z_offset)
    body_data["a_left_lower_leg"] = string_one(lower_leg_z_offset)

    geom_data["a_right_lleg"] = string_triple(s_lower_leg, m_lower_leg, p_lower_leg)
    geom_data["a_left_lleg"] = string_triple(s_lower_leg, m_lower_leg, p_lower_leg)

    # feet
    s_foot = [0, 0, -2*s_lower_leg[1] - s_lower_leg[0]]
    body_data["a_right_foot"] = string_one(s_foot)
    body_data["a_left_foot"] = string_one(s_foot)

    if simple:
        m_foot = mass_dict["foot"] / 5
        foot_h = loaded_data["foot"]["h"]
        foot_w = max(2*s_lower_leg[0], loaded_data["foot"]["w"])
        foot_full_len = loaded_data["foot"]["full_l"]
        toe_len = loaded_data["foot"]["toe_l"]

        s_left_foot1 = [foot_w/2, foot_h/2]
        p_left_foot1 = [0, 0, 0]

        s_left_foot2 = [(foot_full_len - toe_len)/2 - s_left_foot1[0], s_left_foot1[0], s_left_foot1[1]]
        p_left_foot2 = [s_left_foot2[0], 0, 0]

        s_left_foot3 = [s_left_foot1[1], s_left_foot1[0]]
        p_left_foot3 = [p_left_foot2[0] + s_left_foot2[0], 0, 0]

        geom_data["a_left_foot1"] = string_triple(s_left_foot1, m_foot, p_left_foot1)
        geom_data["a_left_foot2"] = string_triple(s_left_foot2, m_foot, p_left_foot2)
        geom_data["a_left_foot3"] = string_triple(s_left_foot3, m_foot, p_left_foot3)
        geom_data["a_right_foot1"] = string_triple(s_left_foot1, m_foot, p_left_foot1)
        geom_data["a_right_foot2"] = string_triple(s_left_foot2, m_foot, p_left_foot2)
        geom_data["a_right_foot3"] = string_triple(s_left_foot3, m_foot, p_left_foot3)

        

        s_left_toes2 = s_left_foot3
        s_left_toes1 = [toe_len/2-s_left_toes2[0], s_left_foot1[0], s_left_foot1[1]]
        p_left_toes1 = [s_left_toes1[0], 0, 0]
        p_left_toes2 = [2*s_left_toes1[0], 0, 0]

        body_data["a_left_toes"] = string_one(p_left_foot3)
        geom_data["a_left_toes1"] = string_triple(s_left_toes1, m_foot, p_left_toes1)
        geom_data["a_left_toes2"] = string_triple(s_left_toes2, m_foot, p_left_toes2)
        body_data["a_right_toes"] = string_one(p_left_foot3)
        geom_data["a_right_toes1"] = string_triple(s_left_toes1, m_foot, p_left_toes1)
        geom_data["a_right_toes2"] = string_triple(s_left_toes2, m_foot, p_left_toes2)

    return body_data, geom_data

def main_military():
    bottom_capsule_diameter = 23.1
    print("neck")
    get_capsule_size(2*5.22, 10.6)

    print("body: bottom up")
    get_capsule_size(bottom_capsule_diameter, 35.3)
    get_capsule_size(20.7, 27.9, is_body=True)
    get_capsule_size(20.9, 27.0, is_body=True)
    get_capsule_size(21.7, 26.8, is_body=True)
    get_capsule_size(16.0, 30.0, is_body=True)

    print("hands: top down")
    get_capsule_size(4.51*2, 33.4, is_upper_arm=True)
    get_capsule_size(3.32*2, 24.0, is_upper_arm=True)

    print("legs: top down")
    get_capsule_size(7.84*2, 31.5)
    get_capsule_size(4.67*2, 40.2)

    print("feet: back to front")
    print(get_foot_size(0.047, 0.09, 0.0303, 0.03))

# military info
# body: neck
# 0.0522 0.0008
# body: bottom up
# 0.1155 0.061
# 0.1035 0.036
# 0.1045 0.0305
# 0.1085 0.0255
# 0.08 0.07
# hands: top down
# 0.0451 0.1444
# 0.0332 0.12
# legs: top down
# 0.0784 0.2152
# 0.0467 0.201


if __name__ == "__main__":
    main_military()
