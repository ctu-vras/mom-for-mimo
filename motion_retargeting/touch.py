"""" This code is heavily inspired by the BabyBench2025 competition starter kit, 
    available https://github.com/babybench/BabyBench2025_Starter_Kit.git
"""

from motion_retargeting.vision import save_video_from_imgs, create_or_clear_folder
from motion_retargeting.handle_data import save_to_npy, load_file
from motion_retargeting.heatmap_sections import HeatmapUtilities

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import gymnasium as gym
import mujoco
import cv2
import trimesh
import mimoEnv.utils as mimo_utils
from copy import deepcopy
import os

NO_TOUCH = 0
MOM_TOUCH = 1
SELF_TOUCH = 2
OTHER_TOUCH = 3


class PlotTouchHelper:
    """ This class handles all that is needed to process, show and save the touch MIMo is sensing.

    The typical usage of this class:
    `pt = PlotTouchHelper(...)
     some loop
      pt.view_touches(...)
      pt.save_snapshot(...)
     pt.save_whole_touch_video(...)`

    Attributes:
        focus_body
        env
        base_body_id
        sensor_outputs
        goal_directory
        temp_path
        target_pos
        show_still_figure
        sensor_points
        timeseries
        mom_holder
        self_holder
        other_holder
        grey_points
        eye_plot_holder
        fig
        ax
    """
    def __init__(self, env, out_dir, time_duration, view_orientation=None, focus_body="hip", base_body_name='mimo_location', 
                 show_still_figure=False, in_touch_sensor_filename=None, out_touch_sensor_filename=None, mesh_visualization=False):
        
        self.focus_body = focus_body
        self.mesh_visualization = mesh_visualization
        self.env = env
        self.base_body_id = mimo_utils.get_body_id(self.env.model, body_name=base_body_name)
        self.sensor_outputs = None  # A holder of sensor outputs, is updated by calling self.update_timeseries
        self.eye_plot_holder = None
        self.init_plot_holders = False
        self.mesh_plot_holders = {}
        self.init_mesh_holders = False

        self.init_view_touches(view_orientation)
        self.init_timeseries(time_duration)

        # Initialize the directory to store the temporary images, that will be later glued to a video
        self.goal_directory = out_dir
        self.temp_path = os.path.join(out_dir, "touch_temp")
        create_or_clear_folder(self.temp_path)


        # Initialize the target position to focus the sensor plot to
        self.target_pos = np.zeros((3,))

        self.show_still_figure = show_still_figure
        if self.show_still_figure:
            if in_touch_sensor_filename:
                self.sensor_points = self.load_touch_sensors_layout(in_touch_sensor_filename)
                self.target_pos = np.mean(self.sensor_points[mimo_utils.get_body_id(env.model, body_name=focus_body)], axis=0)
            else:
                self.sensor_points = self.init_touch_sensors()
                if self.focus_body:
                    self.target_pos = deepcopy(env.data.body(self.focus_body).xpos)
            if out_touch_sensor_filename:
                self.save_touch_sensors_layout(out_touch_sensor_filename)

    def init_touch_sensors(self):
        """ Create a still representation of the sensor positions.
         
        Returns:
            Dict[int: np.ndarray]: The positions of sensors on each body.
        """
        root_id = mimo_utils.get_body_id(self.env.model, body_name='mimo_location')

        # Go through all bodies and note their child bodies
        subtree = mimo_utils.get_child_bodies(self.env.model, root_id)

        points_no_contact = {}

        # Find all touch sensors
        for body_id in subtree:
            if (body_id in self.env.touch.sensor_positions) and (body_id in self.env.touch.sensor_outputs):
                sensor_points = self.env.touch.sensor_positions[body_id]

                sensor_points = mimo_utils.body_pos_to_world(self.env.data, position=sensor_points, body_id=body_id)
                points_no_contact[body_id] = sensor_points

        return points_no_contact
    
    def init_timeseries(self, time_duration):
        """ Initialize the timeseries dictionary with the sections used for the heatmap.

        Args:
            time_duration (int): Number of frames of the whole recording.

        Returns:
            (Dict{str: np.ndarray}): A dictionary of arrays of nans for each heatmap section. The arrays are sized (time,).
        """
        sections = [f"{i}{lr}{fb}" for i in range(18) for lr in ["L", "R"] for fb in ["", "B"]]
        nans = np.empty(time_duration)
        nans[:] = np.nan

        touch_types = {"none": NO_TOUCH, "mom": MOM_TOUCH, "self": SELF_TOUCH, "other": OTHER_TOUCH}
        self.timeseries = {t_type: {s: {ts: np.zeros(time_duration, dtype=int) for ts in sections} for s in sections} for t_type in touch_types.values()}
        self.timeseries["touch types"] = touch_types

        child_bodies = mimo_utils.get_child_bodies(self.env.model, self.base_body_id)
        self.heatmap_utilities = HeatmapUtilities(self.env, child_bodies)

    def write_contact_timeseries(self, body_id, rel_contact_pos, contact_origin, rel_touching_body_contact_pos, touching_body_id, time):
        """ Updates the heatmap timeseries based on the current contact defined by the arguments.
        
        Args:
            body_id (int): The ID of the sensing body.
            rel_contact_pos (np.ndarray): The position of the contact in the coordinate frame of the sensing body.
            contact_origin (int): The contact origin flag.
            time (int): The ID of the current frame. Used to update the heatmap timeseries correctly.
        """
        section = self.heatmap_utilities.get_contact_section(body_id, rel_contact_pos)
        touching_section = self.heatmap_utilities.get_contact_section(touching_body_id, rel_touching_body_contact_pos)
        self.timeseries[contact_origin][section][touching_section][time] = 1
    
    def update_timeseries(self, time):
        """ Collects touches and its origins and based on those updates the `self.sensor_outputs` dictionary 
            also sorts the touches into a heatmap timeseries.
        
        Args:
            time (int): The ID of the current frame. Used to update the heatmap timeseries correctly.
        """
        contact_tuples = self.get_contacts()
        self.sensor_outputs = self.get_empty_sensor_dict()  # Initialize output dictionary

        for contact_id, geom_id, forces, contact_origin, contact_origin_geom in contact_tuples:
            body_id = self.geom_id_2_body_id(geom_id)
            touching_body = self.geom_id_2_body_id(contact_origin_geom)
            rel_contact_pos = self.env.touch.get_contact_position_relative(contact_id=contact_id, body_id=body_id)
            rel_touching_body_contact_pos = self.env.touch.get_contact_position_relative(contact_id=contact_id, body_id=touching_body)
            self.response_function(contact_id, body_id, forces, contact_origin, touching_body)  # This updates sensor_outputs 
            self.write_contact_timeseries(body_id, rel_contact_pos, contact_origin, rel_touching_body_contact_pos, touching_body, time)

    def save_timeseries(self):
        filename = os.path.join(self.goal_directory, "timeseries.npy")
        save_to_npy(self.timeseries, filename)

    # =============== Working with the touches ========================================
    # =================================================================================

    def get_sorted_touches(self):
        """ Returns non activated and activated sensor positions and the corresponding force magnitudes.

        The non activated sensor positions are in an array sized (n, 3).
        The activated sensors are ordered in the dictionaries based on the touch origin (mom, self, other). The arrays are shaped (n, 3).
        The force magnitudes are ordered in a directory in the same manner as sctivated sensors.

        All positions are in the world frame.

        Returns:
            Tuple[np.ndarray, Dict[np.ndarray], Dict[np.ndarray]]: A touple with contents described above.
        """
        points_no_contact = []
        points_contact = {"mom": [], "self": [], "other": []}
        contact_magnitudes = {"mom": [], "self": [], "other": []}

        # Go through all bodies and note their child bodies
        child_bodies = mimo_utils.get_child_bodies(self.env.model, self.base_body_id)

        # Find all contact points and note the force magnitudes
        for body_id in child_bodies:
            if (body_id in self.env.touch.sensor_positions) and (body_id in self.env.touch.sensor_outputs):
                force_vectors = self.sensor_outputs[body_id]["forces"]
                touch_origins = self.sensor_outputs[body_id]["touch origin"]

                force_magnitude = np.linalg.norm(force_vectors, axis=-1, ord=2)

                sensor_points = self.env.touch.sensor_positions[body_id]
                sensor_points = mimo_utils.body_pos_to_world(self.env.data, position=sensor_points, body_id=body_id)

                # Sort the touches based on touch origin
                no_touch_points = sensor_points[touch_origins==NO_TOUCH]
                mom_touch_points = sensor_points[touch_origins==MOM_TOUCH]
                mom_touch_force_magnitudes = force_magnitude[touch_origins==MOM_TOUCH]
                self_touch_points = sensor_points[touch_origins==SELF_TOUCH]
                self_touch_force_magnitudes = force_magnitude[touch_origins==SELF_TOUCH]
                other_touch_points = sensor_points[touch_origins==OTHER_TOUCH]
                other_touch_force_magnitudes = force_magnitude[touch_origins==OTHER_TOUCH]

                # Save the position and force magnitude information in the appropriate dictionary and list
                points_no_contact.append(no_touch_points)
                points_contact["mom"].append(mom_touch_points)
                points_contact["self"].append(self_touch_points)
                points_contact["other"].append(other_touch_points)
                contact_magnitudes["mom"].append(mom_touch_force_magnitudes)
                contact_magnitudes["self"].append(self_touch_force_magnitudes)
                contact_magnitudes["other"].append(other_touch_force_magnitudes)
                
        points_no_contact = np.concatenate(points_no_contact)
        for k in points_contact.keys():
            points_contact[k] = np.concatenate(points_contact[k])
            contact_magnitudes[k] = np.concatenate(contact_magnitudes[k])

        return points_no_contact, points_contact, contact_magnitudes

    def plot_coloured_touches(self, point_size, points_no_contact, points_contact, contact_magnitudes):
        """ Plots all sensors. Uses different collors for different touch origins.
        
        Args:
            point_size (float): The size of the points will be point_size**2, this argument is used only for the initialization.
            points_no_contact (np.ndarray): The positions of the sensors without activity.
            points_contact (Dict[np.ndarray]): The dictionary with positions of the activated sensors. They are sorted in the entries based on the touch origin.
            contact_magnitudes (Dict[np.ndarray]): The dictionary with force magnitudes related to the points.
        """
        if self.focus_body:
            self.target_pos = self.env.data.body(self.focus_body).xpos

        # Subtract all by ball position to center on ball
        xs_gray, ys_gray, zs_gray = self.move_points_to_target(points_no_contact)
        xs_mom, ys_mom, zs_mom = self.move_points_to_target(points_contact["mom"])
        xs_self, ys_self, zs_self = self.move_points_to_target(points_contact["self"])
        xs_other, ys_other, zs_other = self.move_points_to_target(points_contact["other"])

        color_dict, size_dict = self.get_dict_colors_and_sizes(contact_magnitudes, point_size, points_contact)
        
        # Draw sensor points
        if not self.init_plot_holders:
            self.mom_holder = self.ax.scatter(xs_mom, ys_mom, zs_mom)
            self.self_holder = self.ax.scatter(xs_self, ys_self, zs_self,depthshade=False)
            self.other_holder = self.ax.scatter(xs_other, ys_other, zs_other)
            if not self.mesh_visualization:
                self.grey_points = self.ax.scatter(xs_gray, ys_gray, zs_gray, color="k", s=point_size, alpha=.15)
            
            # Color and size must be set separately from the scatter call to keep the option to change them (see https://github.com/matplotlib/matplotlib/issues/27555)
            self.set_color_size(self.mom_holder, color_dict["mom"], size_dict["mom"]) 
            self.set_color_size(self.self_holder, color_dict["self"], size_dict["self"]) 
            self.set_color_size(self.other_holder, color_dict["other"], size_dict["other"]) 
            self.init_plot_holders = True
        else:
            self.set_scatter_data(self.mom_holder, (xs_mom, ys_mom, zs_mom))
            self.set_scatter_data(self.self_holder, (xs_self, ys_self, zs_self))
            self.set_scatter_data(self.other_holder, (xs_other, ys_other, zs_other))
            if not self.mesh_visualization:
                self.set_scatter_data(self.grey_points, (xs_gray, ys_gray, zs_gray))
            
            self.set_color_size(self.mom_holder, color_dict["mom"], size_dict["mom"]) 
            self.set_color_size(self.self_holder, color_dict["self"], size_dict["self"]) 
            self.set_color_size(self.other_holder, color_dict["other"], size_dict["other"]) 

    # =============== Visualization helper functions ==================================
    # =================================================================================
    
    def save_touch_sensors_layout(self, filepath):
        save_to_npy(self.sensor_points, filepath)

    def load_touch_sensors_layout(self, file_path):
        return load_file(file_path, is_dict=True)

    def init_view_touches(self, view_orientation=None, lim=0.25):
        """ Initializes the figure for touch visualization.

        Create the figure and ax,
        Set view angle,
        Set all ax limits to the value of param lim,
        Do not show ax labels.

        Args:
            view_orientation (List|None): A touple with elevation, azimuth and roll of the plot view.
            lim (float): The x,y,z limits of the plot view.
        """
        if not view_orientation:
            e, a, r = 90, 0, 0
        else:
            e, a, r = view_orientation
        plt.ion()
        self.fig = plt.figure(figsize=(6,6), dpi=600)
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.view_init(elev=e, azim=a, roll=r)
        self.ax.set_xlim([-lim, lim])     
        self.ax.set_ylim([-lim, lim])     
        self.ax.set_zlim([-lim, lim])     
        self.ax.set_axis_off()

    def draw_new_data(self):
        """ Draw the set data to the plot and then flush all events.
        This is used for the animation feature of the plot.
        """
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def set_scatter_data(self, plot_holder, data):
        """ Set 3d data to scatter holder.
        
            Args:
                plot_holder (Path3DCollection): The scatter plot holder.
                data (tuple of 3 1D np.arrays): Data to be set to the new plot.
        """
        plot_holder._offsets3d = (data[0], data[1], data[2])

    def set_color_size(self, plot_holder, red_colors, sizes):
        """ Set color and size to the scatter plot.

        Beware that if the color and size is set as an argument in the 'scatter' function calling, 
            this function will do nothing.
            It is a feature, see https://github.com/matplotlib/matplotlib/issues/27555
        
        Args:
            plot_holder (Path3DCollection): The scatter plot holder.
            red_colors (float|np.array): Value(s) to set the red colored dots.
            sizes (float|np.array): Data to be set to the new plot.
        """
        plot_holder.set_facecolor(red_colors)
        plot_holder.set_sizes(sizes)

    def draw_eyes(self, point_size=5):
        """ Draw eyes into the plot. Gets the eye sensor locations and updates the eye holder.
        
        Args:
            point_size (float): The size of the eye will be point_size**2, this argument is used only for the initialization.
        """
        eye_names = ("left_eye", "right_eye")
        eye_sensor_positions = []
        
        for eye_name in eye_names:
            eye_id = self.env.data.body(eye_name).id
            eye_pos = self.env.touch.sensor_positions[eye_id][0]
            eye_pos = mimo_utils.body_pos_to_world(self.env.data, position=eye_pos, 
                                                           body_id=eye_id)
            eye_pos -= self.target_pos
            eye_sensor_positions.append(eye_pos)
        eye_sensor_positions = np.array(eye_sensor_positions).T

        if not self.eye_plot_holder:
            self.eye_plot_holder = self.ax.scatter(eye_sensor_positions[0], eye_sensor_positions[1], 
                                                   eye_sensor_positions[2], c="k", s=point_size, alpha=1)
        else:
            self.set_scatter_data(self.eye_plot_holder, eye_sensor_positions)

    def get_dict_colors_and_sizes(self, force_dict, point_size, points_dict):
        """ Returns a dictionary with colors and a dictionary with sizes.

        Both are calculated relative to force magnitudes.
        
        Args:
            force_dict (Dict[np.ndarray]): The dictionary with force magnitudes related to the points.
            point_size (float): The size of the points will be point_size**2, this argument is used only for the initialization.
            points_dict (Dict[np.ndarray]): The dictionary with positions of the activated sensors.

        Returns:
            Tuple[Dict[np.ndarray], Dict[np.ndarray]]: A dictionary with colors and a dictionary with sizes of the points to be scattered.
        """
        in_color_dict = {"mom": "r", "self": "b", "other": "y"}
        color_dict = {}
        sizes_dict = {}

        for k in in_color_dict.keys():
            color_dict[k], sizes_dict[k] = self.get_colors_and_sizes(force_dict[k], point_size, points_dict[k], in_color_dict[k])
        
        return color_dict, sizes_dict

    def get_colors_and_sizes(self, forces, point_size, points, color_name):
        """ Returns an array with colors and an array with sizes.

        Both are calculated relative to force magnitudes.
        
        Args:
            forces (np.ndarray): The force magnitudes related to the points.
            point_size (float): The size of the points will be point_size**2, this argument is used only for the initialization.
            points (np.ndarray): The positions of the activated sensors.
            color_name (str): The name of the color of the sensor activation.

        Returns:
            Tuple[np.array, np.array]: An array with colors and an array with sizes of the points to be scattered.
        """
        color = np.array(colors.to_rgb(color_name))
        # Set sizes and opacities depending on the force magnitudes
        if len(forces) > 0 and np.amax(forces) > 1e-7:
            size_min = point_size
            size_max = 2*size_min
            sizes = forces / np.amax(forces) * (size_max - size_min) + size_min
            opacity_min = 0.4
            opacity_max = 0.5
            opacities = forces / np.amax(forces) * (opacity_max - opacity_min) + opacity_min
            # Opacities can't be set as an array, so must be set using color array
            red_colors = np.tile(np.hstack((color, [0])), (points.shape[0], 1))
            red_colors[:, 3] = opacities
        else:
            sizes = [5]
            red_colors = [0.4 * c for c in color]

        return red_colors, sizes

    def move_points_to_target(self, points):
        """ Moves points, so that the target position is at (0, 0, 0).
        
        Args:
            points (np.ndarray): Points to be moved. Shaped (n, 3).

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: The moved points separated into arrays shaped (n,).
        """
        xs = points[:, 0] - self.target_pos[0]
        ys = points[:, 1] - self.target_pos[1]
        zs = points[:, 2] - self.target_pos[2]

        return xs, ys, zs

    def view_touches(self, time, point_size=2):
        """ Plot current touches and update the heatmap params.

        This is the main function to be used every iteration to show the touch visualization.
        
        Args:
            time (int): The ID of the current frame. Used to update the heatmap timeseries correctly.
            point_size (float): The size of the eye will be point_size**2, this argument is used only for the initialization
        """
        self.update_timeseries(time)
        p_no_contact, p_contact, contact_m = self.get_sorted_touches()
        self.plot_coloured_touches(point_size, p_no_contact, p_contact, contact_m)
        self.draw_eyes(point_size)
        if self.mesh_visualization: self.plot_meshes()
        self.draw_new_data()

    def save_snapshot(self, specifier):
        """ Save an image of the current figure to a file in `self.temp_path` folder.

        Args:
            specifier (int): The numbers specific for this frame (i.e. the frame number).
        """
        image_from_plot = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
        image_from_plot = image_from_plot.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))
        image_from_plot = cv2.cvtColor(image_from_plot, cv2.COLOR_RGB2BGR)  # Make sure the color is allright

        filepath = os.path.join(self.temp_path, f"touch_{specifier:04}.png")
        cv2.imwrite(filepath, image_from_plot)
        
    def save_whole_touch_video(self, fps=100):
        """ Join the images from `self.temp_path` directory into an .avi video in the `self.goal_directory`.

        Args:
            fps (int): Frames per second of the resulting video.
        """
        plt.close(self.fig)
        if self.show_still_figure:
            video_name = "still_touch.avi"
        else:
            video_name = "touch.avi"
        save_video_from_imgs(self.temp_path, self.goal_directory, img_name_spec="touch", video_name=video_name, fps=fps)

    def plot_meshes(self):
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
                x = vertices[:, 0] + pos[0] - self.target_pos[0]
                y = vertices[:, 1] + pos[1] - self.target_pos[1]
                z = vertices[:, 2] + pos[2] - self.target_pos[2]
                if self.init_mesh_holders:
                    self.mesh_plot_holders[bodyid].remove()
                self.mesh_plot_holders[bodyid] = self.ax.plot_trisurf(x, y, z,
                                                triangles=mesh.faces, color="k", alpha=0.4, shade=True) 
        self.init_mesh_holders = True

    # =============== MIMo touch modified functions ===================================
    # =================================================================================

    def get_contacts(self):
        """ Collects all active contacts involving bodies with touch sensors and their origins.

        For each active contact with a sensing geom we build a tuple ``(contact_id, geom_id, forces, touch_origin)``, where
        `contact_id` is the ID of the contact in the MuJoCo arrays, `geom_id` is the ID of the sensing geom,
        `forces` is a numpy array of the raw output force, as determined by :attr:`.touch_type` and 
        `touch_origin` is the flag that indicates where the touch originated (NO_TOUCH, MOM_TOUCH, SELF_TOUCH, OTHER_TOUCH).

        Returns:
            List[Tuple[int, int, np.ndarray, int]]: A list of tuples with contact information described above.
        """
        contact_tuples = []
        for i in range(self.env.data.ncon):
            contact = self.env.data.contact[i]
            geom1 = contact.geom1
            geom2 = contact.geom2
            body1 = self.geom_id_2_body_id(geom1)
            body2 = self.geom_id_2_body_id(geom2)

            # Do we sense this contact at all
            if self.env.touch.has_sensors(body1) or self.env.touch.has_sensors(body2):
                touch_origin = OTHER_TOUCH

                rel_geoms = []
                if self.env.touch.has_sensors(body1):
                    rel_geoms.append(geom1)
                    if self.env.model.geom(geom2).name.startswith("a_"):
                        touch_origin = MOM_TOUCH
                        touch_origin_geoms = [geom2]
                if self.env.touch.has_sensors(body2):
                    rel_geoms.append(geom2)
                    if self.env.model.geom(geom1).name.startswith("a_"):
                        touch_origin = MOM_TOUCH
                        touch_origin_geoms = [geom1]

                if len(rel_geoms) == 2:
                    touch_origin = SELF_TOUCH
                    touch_origin_geoms = [geom2, geom1]

                raw_forces = self.env.touch.get_raw_force(i, body1)
                if abs(raw_forces[0]) < 1e-9:  # Contact probably inactive
                    continue

                for rel_geom, touch_origin_geom in zip(rel_geoms, touch_origin_geoms):
                    forces = self.env.touch.touch_function(i, rel_geom)
                    contact_tuples.append((i, rel_geom, forces, touch_origin, touch_origin_geom))

        return contact_tuples
    
    def response_function(self, contact_id, body_id, force, touch_origin, touch_origin_body):
        """ Response function. Distributes the output force linearly based on distance. 
            Updates the 'self.sensor_outputs' dict with the forces and the touch origins.

        For a contact and a raw force we get all sensors within a given distance to the contact point and then
        distribute the force such that the force reported at a sensor decreases linearly with distance between the
        sensor and the contact point. Finally, the total force is normalized such that the total force over all sensors
        for this contact is identical to the raw force. The scaling distance is given by double the distance between
        sensor points.

        Args:
            contact_id (int): The ID of the contact.
            body_id (int): The ID of the sensing body.
            force (np.ndarray): The raw force.
            touch_origin (np.ndarray): An array with touch origin flags (NO_TOUCH, MOM_TOUCH, SELF_TOUCH, OTHER_TOUCH).
        """
        scale = self.env.touch.sensor_scales[body_id]
        search_scale = 2*scale
        adjustment_scale = 1*scale
        contact_pos = self.env.touch.get_contact_position_relative(contact_id=contact_id, body_id=body_id)
        nearest_sensors, sensor_distances = self.env.touch.get_sensors_within_distance(contact_pos, body_id, search_scale)

        sensor_adjusted_forces = scale_linear(force, sensor_distances, scale=adjustment_scale)
        force_total = abs(np.sum(sensor_adjusted_forces[:, 0]))

        factor = abs(force[0] / force_total) if force_total > 1e-10 else 0

        if factor != 0:        
            self.sensor_outputs[body_id]["forces"][nearest_sensors] = sensor_adjusted_forces * factor
            self.sensor_outputs[body_id]["touch origin"][nearest_sensors] = touch_origin
            self.sensor_outputs[body_id]["touch_origin_body"][nearest_sensors] = touch_origin_body

    def get_empty_sensor_dict(self):
        """ Returns a dictionary with empty sensor outputs.

        Creates a dictionary of bodies with a dictionary of forces and touch origins.
            The "forces" entry contains an array of zeros for each body with sensors. A body with `n` sensors has an empty
            output array of shape (n, force dim). 

            The "touch origin" entry contains an array of zeros for each body with sensors. The shape of the array is (n,).
            Tis array can contain flags NO_TOUCH, MOM_TOUCH, SELF_TOUCH, OTHER_TOUCH.
            
        The output of this function is equivalent to the touch sensor output if
        there are no contacts.

        Returns:
            Dict[int, Dict[forces: np.ndarray, touch origin: str]: The dictionary of empty sensor outputs.
        """
        sensor_outputs = {}
        no_sensors = 0
        for body_id in self.env.touch.meshes:
            no_sensors += self.env.touch.get_sensor_count(body_id)
            sensor_outputs[body_id] = {"forces": np.zeros((self.env.touch.get_sensor_count(body_id), self.env.touch.touch_size), dtype=np.float32),
                                        "touch origin": np.zeros((self.env.touch.get_sensor_count(body_id)), dtype=np.float32),
                                        "touch_origin_body": np.zeros((self.env.touch.get_sensor_count(body_id)), dtype=np.float32)}
        print(f"{no_sensors=}")
        return sensor_outputs
    
    def geom_id_2_body_id(self, geom_id):
        """ Returns the ID of the sensing body 
        
        Args:
            geom_id (int): The ID of the sensing geom.
        Returns:
            int: The ID of the sensing body.
        """
        return self.env.model.geom(geom_id).bodyid.item()


def scale_linear(force, distances, scale):
    """ Used to scale forces linearly based on distance.

    Adjusts the force by a simple factor, such that force falls linearly from full at `distance = 0`
    to 0 at `distance >= scale`.

    Args:
        force (np.ndarray): The unadjusted force.
        distances (np.ndarray): The adjusted force reduces linearly with increasing distance.
        scale (float): The scaling limit. If ``distance >= scale`` the return value is reduced to 0.

    Returns:
        np.ndarray: The scaled force.
    """
    factor = (scale-distances) / scale
    factor[factor < 0] = 0
    out_force = (force[:, np.newaxis] * factor).T
    return out_force