import os
import numpy as np
import mujoco

from mimoEnv.envs.mimo_env import MIMoEnv, DEFAULT_PROPRIOCEPTION_PARAMS, DEFAULT_VISION_PARAMS, DEFAULT_VESTIBULAR_PARAMS, DEFAULT_TOUCH_PARAMS_V2, DEFAULT_TOUCH_PARAMS 
import mimoEnv.utils as env_utils
from mimoTouch.touch import TrimeshTouch

""" List of possible target bodies.

:meta hide-value:
"""

RESET_POSITION = {"mimo_location": np.array([0.0579584, -0.00157173, 0.0566738, 0.892294, -0.0284863, -0.450353, -0.0135029])}

SCENE_DIRECTORY = os.path.abspath(os.path.join(__file__, "..", "..", "mom_mimoEnv", "assets"))

""" Path to the scene for this experiment.

:meta hide-value:
"""

XML = os.path.join(SCENE_DIRECTORY, "testing.xml")
# XML = os.path.join(SCENE_DIRECTORY, "testing_mimo_resized.xml")

print("Picked scene with path ",XML)

class MIMoRetargeting(MIMoEnv):
    """ MIMo learns about his own body.

    MIMo is tasked with touching a given part of his body using his right arm.
    Attributes and parameters are mostly identical to the base class, but there are two changes.
    The constructor takes two arguments less, ``goals_in_observation`` and ``done_active``, which are both permanently
    set to `True`.
    Finally there are two extra attributes for handling the goal state. The :attr:`.goal` attribute stores the target
    geom in a one hot encoding, while :attr:`.target_geom` and :attr:`.target_body` store the geom and its associated
    body as an index. For more information on geoms and bodies please see the MuJoCo documentation.

    Attributes:
        target_geom (int): The body part MIMo should try to touch, as a MuJoCo geom.
        target_body (str): The name of the kinematic body that the target geom is a part of.
        init_pose (numpy.ndarray): The initial position.
    """

    def __init__(self,
                 model_path=XML,
                 initial_qpos={},
                 n_substeps=1,
                 proprio_params=DEFAULT_PROPRIOCEPTION_PARAMS,
                 touch_params=DEFAULT_TOUCH_PARAMS,
                 vision_params=DEFAULT_VISION_PARAMS,
                 vestibular_params=DEFAULT_VESTIBULAR_PARAMS,
                **kwargs,
                 ):

        super().__init__(model_path=model_path,
                         initial_qpos=initial_qpos,
                        #  n_substeps=n_substeps,
                         proprio_params=proprio_params,
                         touch_params=touch_params,
                         vision_params=vision_params,
                         vestibular_params=vestibular_params,
                         goals_in_observation=False,
                         done_active=False,
                         **kwargs)
        self.init_pose = self.data.qpos.copy()

    def _step_callback(self):
        """ Manually reset position excluding arm each step.

        This restores the body to the sitting position if it deviated.
        Avoids some physics issues that would sometimes occur with welds.
        """
        # Manually set body to sitting position (except for the right arm joints)
        for body_name in RESET_POSITION:
            print("step", body_name)
            # env_utils.set_joint_qpos(self.model, self.data, body_name, RESET_POSITION[body_name])

    def reset_model(self):
        """ Reset to the initial sitting position.

        Returns:
            bool: `True`
        """
        # set qpos as new initial position and velocity as zero
        qpos = self.init_qpos
        qvel = np.zeros(self.data.qvel.shape)

        self.set_state(qpos, qvel)

        # perform 100 steps with no actions to stabilize initial position
        actions = np.zeros(self.action_space.shape)
        self._set_action(actions)
        mujoco.mj_step(self.model, self.data, nstep=100)
        return self._get_obs()
    

    def sample_goal(self):
        return np.zeros((0,))
    
    def get_achieved_goal(self):
        return np.zeros(self.goal.shape)
    
    def is_success(self, achieved_goal, desired_goal):
        return False

    def is_failure(self, achieved_goal, desired_goal):
        return False

    def is_truncated(self):
        return False
    
    def compute_reward(self, achieved_goal, desired_goal, info):
        quad_ctrl_cost = 0.01 * np.square(self.data.ctrl).sum()
        reward = achieved_goal - 0.2 - quad_ctrl_cost
        return np.sum(reward)
    
    def touch_setup(self, touch_params):
        """ Perform the setup and initialization of the touch system.

        This should be overridden if you want to use another implementation!

        Args:
            touch_params (dict): The parameter dictionary.
        """
        self.touch = TrimeshTouch(self, touch_params)
