import matplotlib.pyplot as plt
import numpy as np
import mimoEnv.utils as mimo_utils
import distinctipy
import cv2
import os


class HeatmapUtilities:
    def __init__(self, env, bodies):
        self.env = env
        self.bodies = {env.data.body(body).name: body for body in bodies}
        self.borders = {}

        # Find centers or other borders of the body areas and initialize them
        # Legs
        self.borders["left_foot"] = self.find_borders("left_foot")
        self.borders["right_foot"] = self.find_borders("right_foot")

        self.borders["right_upper_leg"] = self.find_borders("right_upper_leg")
        self.borders["right_lower_leg"] = self.find_borders("right_lower_leg")
        self.borders["left_upper_leg"] = self.find_borders("left_upper_leg")
        self.borders["left_lower_leg"] = self.find_borders("left_lower_leg")

        # Arms
        self.borders["right_lower_arm"] = self.find_lower_arm_border("right_lower_arm")
        self.borders["left_lower_arm"] = self.find_lower_arm_border("left_lower_arm")
        self.borders["right_upper_arm"] = self.find_upper_arm_border("right_upper_arm")
        self.borders["left_upper_arm"] = self.find_upper_arm_border("left_upper_arm")

        # Hands
        self.borders["left_hand"] = self.find_borders("left_hand")
        self.borders["left_fingers"] = self.find_borders("left_fingers")
        self.borders["right_hand"] = self.find_borders("right_hand")
        self.borders["right_fingers"] = self.find_borders("right_fingers")

        # Body
        self.borders["hip"] = self.find_borders("hip")
        self.borders["lower_body"] = self.find_borders("lower_body")
        self.borders["upper_body"] = self.find_borders("upper_body")
        self.borders["head"] = self.find_borders("head")

    def find_borders(self, body_name):
        """ Locate the center of the given body
        
        Args:
            body_name (str): The name of the body"""
        body_id = self.bodies[body_name]
        vertecies = np.vstack([mesh.vertices for mesh in self.env.touch._submeshes[body_id]])
        middle = np.mean(vertecies, axis=0)
        return middle
    
    def find_lower_arm_border(self, body_name):
        """ Find the borders at the center of the body 
        with the exception of the vertical, where we find a point 30% from the bottom
        
        Args:
            body_name (str): The name of the body"""
        body_id = self.bodies[body_name]
        vertecies = np.vstack([mesh.vertices for mesh in self.env.touch._submeshes[body_id]])
        middle = np.mean(vertecies, axis=0)
        middle[2] = np.percentile(vertecies[:, 2], 30)
        return middle
    
    def find_upper_arm_border(self, body_name):
        """ Find the borders at the center of the body 
        with the exception of the vertical, where we find a point 30% from the top
        
        Args:
            body_name (str): The name of the body"""
        body_id = self.bodies[body_name]
        vertecies = np.vstack([mesh.vertices for mesh in self.env.touch._submeshes[body_id]])
        middle = np.mean(vertecies, axis=0)
        middle[2] = np.percentile(vertecies[:, 2], 70)
        return middle
    
    def get_body_section(self, body_name, rel_contact_pos, section_number):
        """ Returns the foot area. 
        
        Args:
            body_name (str): The name of the body
            rel_contact_pos (np.array): The position of the contact in the coordinate system of the body
            section_number (np.array): The section number of the body
        """
        section = section_number if type(section_number) is str else str(section_number) 
        if rel_contact_pos[1] <= self.borders[body_name][1]:
            section += "R"
        else:
            section += "L"
        if rel_contact_pos[0] <= self.borders[body_name][0]:
            section += "B"
        return section
    
    def get_foot_section(self, body_name, rel_contact_pos):
        """ Returns the foot area. 
        
        Args:
            body_name (str): The name of the body
            rel_contact_pos (np.array): The position of the contact in the coordinate system of the body
        """
        section = "1"
        if body_name.startswith("right"):
            section += "R"
        else:
            section += "L"
        if rel_contact_pos[0] <= self.borders[body_name][0]:
            section += "B"
        return section
        
    def get_head_section(self, rel_contact_pos):
        """ Returns the head area. The areas around the mouth are determined by a simple sloping line
        
        Args:
            rel_contact_pos (np.array): The position of the contact in the coordinate system of the body
        """
        body_name = "head"
        if rel_contact_pos[2] <= self.borders[body_name][2]:
            section = "13"
        else:
            section = "17"
        if rel_contact_pos[1] <= self.borders[body_name][1]:
            section += "R"
        else:
            section += "L"
        if rel_contact_pos[0] <= self.borders[body_name][0]:
            section += "B"

        # The magic number makes sure the mouth section is the right size
        magic_number = 0.6
        if section == "13L":
            if rel_contact_pos[2] <= magic_number * rel_contact_pos[0]:
                section = "11L"
        elif section == "13R":
            if rel_contact_pos[2] <= magic_number * rel_contact_pos[0]:
                section = "11R"
        return section
    
    def get_limb_section(self, body_name, rel_contact_pos, upper_section_number, 
                         lower_section_number, lower_arm=False):
        """ Returns the limb area. This function can be used for both arms and legs
        THe function determins if the touch happened to the upper or the lower part of the body
        
        Args:
            body_name (str): The name of the body
            rel_contact_pos (np.array): The position of the contact in the coordinate system of the body
            upper_section_number (int): The number label of the upper area
            lower_section_number (int): The number label of the lower area
            lower_arm (Bool): Signifies if the current limb is a lower arm
        """
        upper_section_number = upper_section_number if type(upper_section_number) is str else str(upper_section_number) 
        lower_section_number = lower_section_number if type(lower_section_number) is str else str(lower_section_number) 

        if rel_contact_pos[2] <= self.borders[body_name][2]:
            section = lower_section_number
        else:
            section = upper_section_number

        if body_name.startswith("right"):
            section += "R"
        else:
            section += "L"

        if lower_arm:
            if section.endswith("R"):
                if rel_contact_pos[1] <= self.borders[body_name][1]:
                    section += "B"
            else:
                if rel_contact_pos[1] >= self.borders[body_name][1]:
                    section += "B"
        else:
            if rel_contact_pos[0] <= self.borders[body_name][0]:
                section += "B"

        return section
    

    def get_hand_section(self, body_name, rel_contact_pos):
        """ Returns the hand area 
        
        Args:
            body_name (str): The name of the body
            rel_contact_pos (np.array): The position of the contact in the coordinate system of the body
        """
        section = "7"
        
        if body_name.startswith("right"):
            section += "R"
            if rel_contact_pos[1] <= self.borders[body_name][1]:
                section += "B"
        else:
            section += "L"
            if rel_contact_pos[1] >= self.borders[body_name][1]:
                section += "B"
            
        return section

    def get_contact_section(self, body_id, rel_contact_pos):
        """ Return the contact area label depending on the body id and contact position relative to it
        Works also for the adult bodies
        When no area fits the inputs, 0L is returned
        
        Args:
            body_id (int): The id of the body
            rel_contact_pos (np.array): The position of the contact in the coordinate system of the body
        """
        body_name = self.env.data.body(body_id).name

        # Make it work for adult sections
        if body_name.startswith("a_"):
            body_name = body_name[2:]

        section = "0L"
        match body_name:
            case "left_toes":
                section = "1L"
            case "right_toes":
                section = "1R"
            case "right_foot" | "left_foot":
                section = self.get_foot_section(body_name, rel_contact_pos)
            case "right_lower_leg" | "left_lower_leg":
                section = self.get_limb_section(body_name, rel_contact_pos, "3", "2")
            case "right_upper_leg" | "left_upper_leg":
                section = self.get_limb_section(body_name, rel_contact_pos, "4", "3")
            case "hip" | "lower_body":
                section = self.get_body_section(body_name, rel_contact_pos, "5")
            case "upper_body" | "chest":
                section = self.get_body_section(body_name, rel_contact_pos, "6")
            case "left_hand" | "left_fingers" | "right_hand" | "right_fingers":
                section = self.get_hand_section(body_name, rel_contact_pos)
            case "right_lower_arm" | "left_lower_arm":
                # Arms have flipped z axis to the rest, so the upper and lower section numbers are flipped
                section = self.get_limb_section(body_name, rel_contact_pos, "8", "9", lower_arm=True)
            case "right_upper_arm" | "left_upper_arm":
                # Arms have flipped z axis to the rest, so the upper and lower section numbers are flipped
                section = self.get_limb_section(body_name, rel_contact_pos, "9", "10")
            case "head":
                section = self.get_head_section(rel_contact_pos)
            case "left_eye":
                section = "17L"
            case "right_eye":
                section = "17R"
        return section
    
    def plot_sections(self, print_text=False, show_mesh=False, show_eyes=False):
        """ Visualize the areas on the MIMo body and save the back and front view separately.
        Goes through the bodies, simulates touches to them and colores them based on the area.
        
        Args:
            print_text (Bool): Determines, if the names of the sections should be shown
            show_mesh (Bool): Determines, if the body mesh should be shown
            show_eyes (Bool): Determines, if the infant eyes should be shown
        """
        LOCATIONS = ['1L', '1R','2L', '2R','3L','3R', '4L', '4R', '5L', '5R','6L', '6R', '7L', '7R', '8L', 
             '8R','9L','9R','10L','10R','11L','11R','13L','13R', '17L','17R']
        LOCATIONS_BACK = [location + 'B' for location in LOCATIONS if location not in ['11L', '11R']]

        locations = LOCATIONS + LOCATIONS_BACK

        sensor_section_dict = {name: [] for name in locations}
        meshes = self.env.touch.meshes

        target_pos = self.env.data.body("hip").xpos

        for body_id in meshes:
            mesh = meshes[body_id]
            rot_mat = mimo_utils.get_body_rotation(self.env.data, body_id)
            pos = mimo_utils.get_body_position(self.env.data, body_id)
            if mesh.vertices.shape[0] > 1:
                sensors = mesh.vertices
                rand_sensors = mesh.sample(2500)
                sensors = sensors if len(sensors) > len(rand_sensors) else rand_sensors
                for rel_sensor_pos in sensors:
                    section = self.get_contact_section(body_id, rel_sensor_pos)
                    abs_sensor_pos = rot_mat @ rel_sensor_pos + pos - target_pos
                    sensor_section_dict[section].append(abs_sensor_pos)

        plt.ion()
        fig = plt.figure(figsize=(6,6), dpi=600)
        ax = fig.add_subplot(111, projection='3d')
        lim=0.2
        ax.set_xlim([-lim, lim])     
        ax.set_ylim([-lim, lim])     
        ax.set_zlim([-lim, lim])     
        ax.set_axis_off()

        colors = distinctipy.get_colors(len(sensor_section_dict))
        for i, section in enumerate(sensor_section_dict):
            # if "B" not in section:
            #     continue
            data = np.array(sensor_section_dict[section])
            if data.shape[0] == 0:
                print(section)
                continue
            ax.scatter(data[:, 0], data[:, 1], data[:, 2], s=3, color=colors[i])
            t_pos = data.mean(axis=0)
            if print_text:
                ax.text(t_pos[0], t_pos[1], t_pos[2], section, fontsize=3, 
                        bbox=dict(facecolor=colors[i], alpha=0.5))

        if show_eyes:
            eye_names = ("left_eye", "right_eye")
            eye_sensor_positions = []
            
            for eye_name in eye_names:
                eye_id = self.env.data.body(eye_name).id
                eye_pos = self.env.touch.sensor_positions[eye_id][0]
                eye_pos = mimo_utils.body_pos_to_world(self.env.data, position=eye_pos, 
                                                            body_id=eye_id)
                eye_pos -= target_pos
                eye_sensor_positions.append(eye_pos)
            eye_sensor_positions = np.array(eye_sensor_positions).T
            ax.scatter(eye_sensor_positions[0], eye_sensor_positions[1],
                        eye_sensor_positions[2], s=2, c='k')
        
        if show_mesh:
            for bodyid in self.env.touch.meshes:
                # print(bodyid)
                mesh = self.env.touch.meshes[bodyid]
                if mesh.vertices.shape[0] > 1:
                    rot_mat = mimo_utils.get_body_rotation(self.env.data, bodyid)
                    vertices = []
                    for v in mesh.vertices:
                        vertices.append(rot_mat @ v)
                    vertices = np.array(vertices)
                    pos = mimo_utils.get_body_position(self.env.data, bodyid)
                    x = vertices[:, 0] + pos[0] - target_pos[0]
                    y = vertices[:, 1] + pos[1] - target_pos[1]
                    z = vertices[:, 2] + pos[2] - target_pos[2]
                    ax.plot_trisurf(x, y, z, triangles=mesh.faces, color="k", alpha=0.4, shade=True) 
        e, a, r = 0, 0, 0

        ax.view_init(elev=e, azim=a, roll=r)
        fig.canvas.draw()
        save_img(fig, "back")

        a = 180
        ax.view_init(elev=e, azim=a, roll=r)
        fig.canvas.draw()
        save_img(fig, "front")
        plt.close()

def save_img(fig, specifier):
    """ Save the image from the current fig as a .png file
    
    Args:
        specifier (str): The string differentiating the file saved in this instance from others
    """
    image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    image_from_plot = cv2.cvtColor(image_from_plot, cv2.COLOR_RGB2BGR)  # Make sure the color is allright
    path = "motion_retargeting/data/touch"
    filepath = os.path.join(path, f"heatmap_{specifier}.png")
    cv2.imwrite(filepath, image_from_plot)


