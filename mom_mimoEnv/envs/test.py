
import os
import numpy as np
import mujoco

from mimoEnv.envs.mimo_env import MIMoEnv, DEFAULT_TOUCH_PARAMS, DEFAULT_VISION_PARAMS, DEFAULT_VESTIBULAR_PARAMS, \
    DEFAULT_PROPRIOCEPTION_PARAMS, DEFAULT_TOUCH_PARAMS, DEFAULT_TOUCH_PARAMS_V2
from mimoTouch.touch import TrimeshTouch
from mimoActuation.muscle import MuscleModel
from mimoActuation.actuation import SpringDamperModel
import mimoEnv.utils as env_utils

SCENE_DIRECTORY = os.path.abspath(os.path.join(__file__, "..", "..", "assets"))

XML = os.path.join(SCENE_DIRECTORY, "scene_adult_infant.xml")
print(XML)
""" Path to the test scene.

:meta hide-value:
"""
class MIMoTestEnv(MIMoEnv):
    def __init__(self,
                model_path=XML,
                proprio_params=DEFAULT_PROPRIOCEPTION_PARAMS,
                touch_params=None,
                vision_params=None,
                vestibular_params=DEFAULT_VESTIBULAR_PARAMS,
                done_active=False,
                goals_in_observation=False,
                **kwargs,
                ):

        super().__init__(model_path=model_path,
                         proprio_params=proprio_params,
                         touch_params=touch_params,
                         vision_params=vision_params,
                         vestibular_params=vestibular_params,
                         done_active=done_active,
                         goals_in_observation=goals_in_observation,
                         **kwargs,)
        
    def get_achieved_goal(self):
        return np.zeros(self.goal.shape)
    
    def is_success(self, achieved_goal, desired_goal):
        return False

    def is_failure(self, achieved_goal, desired_goal):
        return False

    def is_truncated(self):
        return False

    def sample_goal(self):
        return np.zeros((0,))
    
    def compute_reward(self, achieved_goal, desired_goal, info):
        quad_ctrl_cost = 0.01 * np.square(self.data.ctrl).sum()
        reward = achieved_goal - 0.2 - quad_ctrl_cost
        return reward
    def reset_model(self):
        self.set_state(self.init_qpos, self.init_qvel)
        mujoco.mj_forward(self.model, self.data)
        return self._get_obs()