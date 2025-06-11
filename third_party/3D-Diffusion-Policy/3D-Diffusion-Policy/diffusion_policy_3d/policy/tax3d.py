import torch
import numpy as np

from non_rigid.utils.script_utils import create_model
from diffusion_policy_3d.policy.base_policy import BasePolicy
import rpad.visualize_3d.plots as vpl

class TAX3D(BasePolicy):
    """
    This is simple TAX3D wrapper exclusively for policy rollouts.
    """
    def __init__(
            self,
            task_name,
    ):
        super().__init__()
        self.task_name = task_name
        
        # Initializing current goal position. This is set during policy reset.
        self.goal_position = None
        self.results_world = None

    def reset(self):
        """
        Since this is open loop, this function will set the goal position to None.
        """
        self.goal_position = None

    def clip_vec_mag(self, vec, mag):
        vec_norm = torch.linalg.vector_norm(vec)
        if vec_norm == 0:
            return vec
        else:
            return vec / vec_norm * torch.clamp(vec_norm, max=mag)

    def predict_action(self, goal_pc, results, obs_dict, deform_params, control_type='position'):
        """
        Predict the action.
        """
        # if goal_position is unset (after policy reset), predict the goal position.
        if self.goal_position == None:
            if self.task_name == "proccloth":
                goal1 = torch.as_tensor(goal_pc[[1], :], device=self.device)
                goal2 = torch.as_tensor(goal_pc[[0], :], device=self.device)
            elif self.task_name == "hangbag":
                # goal1 = pred_action[:, 209, :] + torch.tensor([0, -0.5, 1.0], device=self.device)
                # goal2 = pred_action[:, 297, :] + torch.tensor([0, -0.5, 1.0], device=self.device)
                raise NotImplementedError("Hangbag task is not implemented yet.")
            else:
                raise ValueError(f"Unknown task name: {self.eval_cfg.env_runner.task_name}")
            self.goal_position = torch.cat([goal1, goal2], dim=1)#.unsqueeze(0)
            self.results_world = [res for res in results]


        # TODO: for ee_position, goal_position should be converted to a noramlized delta position
        if control_type == 'position':
            action = self.goal_position.unsqueeze(0)
        elif control_type == 'ee_position':
            delta1 = self.goal_position[:, :3] - obs_dict['pc_action'][:, deform_params['node_density'] - 1, :]
            delta2 = self.goal_position[:, 3:] - obs_dict['pc_action'][:, 0, :]

            # clipping the delta values            
            delta1 = self.clip_vec_mag(delta1, 0.1)
            delta2 = self.clip_vec_mag(delta2, 0.1)

            action = torch.cat([delta1, delta2], dim=1).unsqueeze(0)

        action_dict = {
            'action': action,
        }

        return action_dict