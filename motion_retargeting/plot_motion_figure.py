import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
import motion_retargeting.handle_data as hd


class PlotHolder:
    """ This class is experimental with no real use, it plots the motioncapture skeleton with the coordinate systems"""
    def __init__(self, ax, fig):
        self.points = None
        self.xyz_init = False
        self.x = []
        self.y = []
        self.z = []
        self.fig = fig
        self.ax = ax

    def init_xyz(self, x, y, z, base):
        for i in range(x.shape[0]):
            self.x.append(self.plot(x[i], base[i], 'r')[0])
            self.y.append(self.plot(y[i], base[i], 'g')[0])
            self.z.append(self.plot(z[i], base[i], 'b')[0])
        self.xyz_init = True

    def set_xyz(self, x, y, z, base):
        if not self.xyz_init:
            self.init_xyz(x, y, z, base)
        else:
            for i in range(x.shape[0]):
                set_3d_data(self.x[i], np.stack((x[i], base[i])).T)
                set_3d_data(self.y[i], np.stack((y[i], base[i])).T)
                set_3d_data(self.z[i], np.stack((z[i], base[i])).T)
        
    def plot(self, p1, p2, color):
        return self.ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color=color)



def unpack_dict(dict):
    """
    unpack the saved dict into a numpy array of size (time, xyz, body part)
    """
    r = np.transpose(np.array(list(dict.values()), dtype=float), (1, 2, 0))
    print(r.shape)
    return(r)

def plot_coord_systems(plot_holder, positions, angles):
    size = 90
    positions = positions.T
    r = Rotation.from_quat(angles.T).as_matrix()
    x = r[:, 0]
    x = x * (size/np.linalg.norm(x)) + positions
    y = r[:, 1]
    y = y * (size/np.linalg.norm(y)) + positions
    z = r[:, 2]
    z = z * (size/np.linalg.norm(z)) + positions
    plot_holder.set_xyz(x, y, z, positions)


def plot_3d_line(ax, start, stop, color):
     print(np.stack((start[:, 0], stop[:, 0])).shape)
     
     ax.plot(np.stack((start[:, 0], stop[:, 0])), np.stack((start[:, 1], stop[:, 1])) , np.stack((start[:, 2], stop[:, 2])))

def plot_coord_sys(ax, base, angles):
    r = Rotation.from_euler("xyz", angles).as_matrix()
    print(r)
    x = r[:, 0] + base
    y = r[:, 1] + base
    z = r[:, 2] + base
    
    plot(ax, np.stack((base, x)), 'r')
    plot(ax, y, 'g')
    plot(ax, z, 'b')
     
def plot(ax, data, color='r'):
    ax.plot(data[0], data[1], data[2], color=color)

def set_3d_data(plot_holder, data):
    plot_holder.set_xdata(data[0])
    plot_holder.set_ydata(data[1])
    plot_holder.set_3d_properties(data[2])

class PlotMotionFigure:
    def __init__(self, data_folder):
        self.folder = data_folder
        self.positions = unpack_dict(hd.load_file(hd.get_skeleton_position_filepath(self.folder), is_dict=True))
        self.global_angles = unpack_dict(hd.load_file(hd.get_skeleton_filepath(self.folder), is_dict=True))
        
        self.local_angles = unpack_dict(hd.load_file(hd.get_skeleton_angles_filepath(self.folder), is_dict=True))

        self.body_tree = hd.get_body_tree()

        self.init_plot()
        self.create_labels_dict()

    def init_plot(self):
        plt.ion()
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        self.p = PlotHolder(ax, fig)
        i = 0
        self.p.points, = self.p.ax.plot(self.positions[i, 0], self.positions[i, 1], self.positions[i, 2], ".")
        self.p.ax.set_aspect('equal')

    def create_labels_dict(self):
        labels = hd.load_file(hd.get_labels_filepath(self.folder))
        self.labes_dict = {}
        for i, label in enumerate(labels):
            self.labes_dict[label] = i

    def step(self, i):
        pose = self.positions[i]
        set_3d_data(self.p.points, pose)
        plot_coord_systems(self.p, pose, self.global_angles[i])

        self.p.fig.canvas.draw()
        self.p.fig.canvas.flush_events()

    def get_lengths(self):
        points = self.positions[0]
        lengths = []
        for section in self.body_tree:
            for b_index in range(len(section) - 1):
                base_idx = self.labes_dict[section[b_index]]
                this_idx = self.labes_dict[section[b_index+1]]

                length = np.linalg.norm(points[:, base_idx] - points[:, this_idx])
                lengths.append(length)

        return np.array(lengths)


def run():
    """"
    step 1 DONE: draw it globally
    step 2: add local angles
        bro, that is not ok...
    """
    FOLDER_SPECIFICATION = "mom_pose2"
    mo_figure = PlotMotionFigure(FOLDER_SPECIFICATION)

    for i in range(mo_figure.positions.shape[0]):
        mo_figure.step(i)


if __name__ == "__main__":
    run()

