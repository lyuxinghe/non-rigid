"""
DeformEnv class is the core class for loading and running various tasks.


Note: this code is for research i.e. quick experimentation; it has minimal
comments for now, but if we see further interest from the community -- we will
add further comments, unify the style, improve efficiency and add unittests.

@contactrika, @yonkshi

"""

import os
import time

from matplotlib import cm  # for colors
import numpy as np
import gym
import pybullet
import pybullet_utils.bullet_client as bclient

from ..utils.anchor_utils import (
    attach_anchor, command_anchor_velocity, command_anchor_position, create_anchor, create_anchor_geom,
    pin_fixed, change_anchor_color_gray)
from ..utils.init_utils import (
    load_deform_object, load_rigid_object, reset_bullet, load_deformable, 
    load_floor, get_preset_properties, apply_anchor_params)
from ..utils.mesh_utils import get_mesh_data
from ..utils.task_info import (
    DEFAULT_CAM_PROJECTION, DEFORM_INFO, SCENE_INFO, TASK_INFO,
    TOTE_MAJOR_VERSIONS, TOTE_VARS_PER_VERSION)
from ..utils.procedural_utils import (
    gen_procedural_hang_cloth, gen_procedural_button_cloth)
from ..utils.args import preset_override_util
from ..utils.process_camera import ProcessCamera, cameraConfig

from scipy.spatial.transform import Rotation as R

from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

import plotly.graph_objects as go
from PIL import Image
import copy
import pkgutil
import pybullet_data

def order_loop_vertices(vertices):
    """ Order the vertices of a loop in a clockwise manner. """
    top = []
    left = []
    right = []
    bottom = []
    # getting all top vertices
    for i in range(len(vertices)):
        top.append(vertices[i])
        if vertices[i + 1] - vertices[i] != 1:
            vertices = vertices[i+1:]
            break

    # getting all bottom vertices
    for i in range(1, len(vertices) + 1):
        bottom.append(vertices[-i])
        if vertices[-i] - vertices[-i - 1] != 1:
            vertices = vertices[:-i]
            break

    # getting left and right vertices
    while vertices:
        left.append(vertices[0])
        right.append(vertices[1])
        vertices = vertices[2:]
    
    # reverse left vertices
    left = left[::-1]
    return top + right + bottom + left


class HangBagTAX3D(gym.Env):
    MAX_OBS_VEL = 20.0  # max vel (in m/s) for the anchor observations
    MAX_ACT_VEL = 10.0  # max vel (in m/s) for the anchor actions
    WORKSPACE_BOX_SIZE = 20.0  # workspace box limits (needs to be >=1)
    STEPS_AFTER_DONE = 500     # steps after releasing anchors at the end
    FORCE_REWARD_MULT = 1e-4   # scaling for the force penalties
    FINAL_REWARD_MULT = 400    # multiply the final reward (for sparse rewards)
    SUCESS_REWARD_TRESHOLD = 2.5  # approx. threshold for task success/failure

    def __init__(self, args):
        self.args = args
        self.cam_on = args.cam_resolution > 0

        # this is hacky - lets a store anchor pcd for demos/configurations for now
        self.camera_config = None

        # # storing scene name
        # scene_name = self.args.task.lower()
        # if scene_name in ['hanggarment', 'bgarments', 'sewing','hangproccloth']:
        #    self.scene_name = 'hangcloth'  # same hanger for garments and cloths
        # elif scene_name.startswith('button'):
        #     self.scene_name = 'button'
        # elif scene_name.startswith('dress'):
        #     self.scene_name = 'dress'  # same human figure for dress and mask tasks

        # Initialize sim and load objects.
        # self.sim = bclient.BulletClient(
        #     connection_mode=pybullet.GUI if args.viz else pybullet.DIRECT)
        self.sim = bclient.BulletClient(connection_mode=pybullet.DIRECT)
        if self.args.viz:  # no rendering during load
            self.sim.configureDebugVisualizer(pybullet.COV_ENABLE_RENDERING, 0)

        # use egl for rendering
        self.sim.setAdditionalSearchPath(pybullet_data.getDataPath())
        egl = pkgutil.get_loader('eglRenderer')
        if (egl):
            pluginId = self.sim.loadPlugin(egl.get_filename(), '_eglRendererPlugin')
        else:
            pluginId = self.sim.loadPlugin('eglRendererPlugin')
        print('pluginId=', pluginId)
         
        reset_bullet(self.args, self.sim, debug=args.debug)

        # reset_bullet(args, self.sim, debug=args.debug)
        self.food_packing = self.args.env.startswith('FoodPacking')
        self.num_anchors = 1 if self.food_packing else 2
        res = self.load_objects(self.sim, self.args, debug=True)
        self.rigid_ids, self.deform_id, self.deform_obj, self.goal_pos, _ = res

        # Step 3: Load floor
        load_floor(self.sim, debug=args.debug)

        self.max_episode_len = self.args.max_episode_len
        # Define sizes of observation and action spaces.
        self.gripper_lims = np.tile(np.concatenate(
            [HangBagTAX3D.WORKSPACE_BOX_SIZE * np.ones(3),  # 3D pos
             np.ones(3)]), self.num_anchors)             # 3D linvel/MAX_OBS_VEL
        if args.cam_resolution <= 0:  # report gripper positions as low-dim obs
            self.observation_space = gym.spaces.Box(
                -1.0 * self.gripper_lims, self.gripper_lims)
        else:  # RGB WxHxC
            shape = (args.cam_resolution, args.cam_resolution, 3)
            if args.flat_obs:
                shape = (np.prod(shape),)
            self.observation_space = gym.spaces.Box(
                low=0, high=255 if args.uint8_pixels else 1.0,
                dtype=np.uint8 if args.uint8_pixels else np.float16,
                shape=shape)
        act_sz = 3  # 3D linear velocity for anchors
        self.action_space = gym.spaces.Box(  # [-1, 1]
            -2.0 * np.ones(self.num_anchors * act_sz),
            2.0 * np.ones(self.num_anchors * act_sz))
        if self.args.debug:
            print('Created HangBagTAX3D with obs', self.observation_space.shape,
                  'act', self.action_space.shape)

        # Point cloud observation initialization
        self.pcd_mode = args.pcd
        if args.pcd:
            self.camera_config = cameraConfig.from_file(args.cam_config_path)
            self.object_ids = res[0]
            self.object_ids.append(res[1])

            print(f"Starting object ids: {self.object_ids}")
            print(f"Deformable ID: {res[1]}")

        # Storing rollout video.
        self.rollout_vid = args.rollout_vid
        if args.rollout_vid:
            self.vid_frames = []
            self.color_key = None
            self.vid_width = 500
            self.vid_height = 500

    @staticmethod
    def unscale_vel(act, unscaled):
        if unscaled:
            return act
        return act*HangBagTAX3D.MAX_ACT_VEL

    @property
    def anchor_ids(self):
        return list(self.anchors.keys())

    @property
    def _cam_viewmat(self):
        dist, pitch, yaw, pos_x, pos_y, pos_z = self.args.cam_viewmat
        cam = {
            'distance': dist,
            'pitch': pitch,
            'yaw': yaw,
            'cameraTargetPosition': [pos_x, pos_y, pos_z],
            'upAxisIndex': 2,
            'roll': 0,
        }
        view_mat = self.sim.computeViewMatrixFromYawPitchRoll(**cam)
        return view_mat

    def get_texture_path(self, file_path):
        # Get either pre-specified texture file or a random one.
        if self.args.use_random_textures:
            parent = os.path.dirname(file_path)
            full_parent_path = os.path.join(self.args.data_path, parent)
            randfile = np.random.choice(list(os.listdir(full_parent_path)))
            file_path = os.path.join(parent, randfile)
        return file_path

    def load_objects(self, sim, args, debug, 
                     cloth_rot=None, 
                     rigid_trans=None, rigid_rot=None, 
                     deform_params={},
                     anchor_params={
                         'hanger_scale': 1.0,
                         'tallrod_scale': 1.0,
                     }):
        scene_name = self.args.task.lower()
        if scene_name in ['hanggarment', 'bgarments', 'sewing','hangproccloth']:
           scene_name = 'hangcloth'  # same hanger for garments and cloths
        elif scene_name.startswith('button'):
            scene_name = 'button'
        elif scene_name.startswith('dress'):
            scene_name = 'dress'  # same human figure for dress and mask tasks

        # Make v0 the random version
        if args.version == 0:
            args.use_random_textures = True

        data_path = os.path.join(os.path.split(__file__)[0], '..', 'data')
        args.data_path = data_path
        sim.setAdditionalSearchPath(data_path)

        # Adjust info in DEFORM_INFO and SCENE_INFO for the specific task,
        # then record the appropriate path to deformable into deform_obj.
        if args.override_deform_obj is not None:
            deform_obj = args.override_deform_obj
        else:
            assert (args.task in TASK_INFO)  # already checked in args
            assert (args.version <= len(TASK_INFO[args.task]))
            if args.version == 0:
                # if params not specified, pick a random bag
                if deform_params == {}:
                    tmp_id0 = np.random.randint(TOTE_MAJOR_VERSIONS)
                    tmp_id1 = np.random.randint(TOTE_VARS_PER_VERSION)
                else:
                    tmp_id0 = deform_params['id0']
                    tmp_id1 = deform_params['id1']
                    assert (tmp_id0 in range(TOTE_MAJOR_VERSIONS))
                    assert (tmp_id1 in range(TOTE_VARS_PER_VERSION))

                deform_obj = f'bags/totes/bag{tmp_id0:d}_{tmp_id1:d}.obj'
                # if its not in DEFORM INFO, add based on the major version
                if deform_obj not in DEFORM_INFO:
                    tmp_key = f'bags/totes/bag{tmp_id0:d}_0.obj'
                    assert (tmp_key in DEFORM_INFO)
                    DEFORM_INFO[deform_obj] = DEFORM_INFO[tmp_key].copy()
            else:
                deform_obj = TASK_INFO[args.task][args.version - 1]


            preset_override_util(args, DEFORM_INFO[deform_obj])
        if deform_obj in DEFORM_INFO:
            preset_override_util(args, DEFORM_INFO[deform_obj])

        # Load deformable object.
        texture_path = args.deform_texture_file
        #texture_path = "textures/deform/orange_pattern.png"
        #self.args.use_random_textures = False
        # Randomize textures for deformables (except YCB food objects).
        if not self.food_packing:
            texture_path = os.path.join(
                data_path, self.get_texture_path(args.deform_texture_file))
            
            # FIXING CLOTH TEXTURE FOR NOW
            texture_path = os.path.join(
                data_path, 'textures/deform/blue_bright.png'
            )
        

        # transform cloth orientation
        if cloth_rot is not None:
            # cloth_position, cloth_orientation = self.sim.multiplyTransforms(
            #     #positionA=R.from_euler('xyz', cloth_rot).apply(args.deform_init_pos),
            #     positionA=args.deform_init_pos,
            #     orientationA=self.sim.getQuaternionFromEuler(args.deform_init_ori),
            #     positionB=(0, 0, 0),
            #     orientationB=self.sim.getQuaternionFromEuler(cloth_rot),
            # )
            # cloth_orientation = self.sim.getEulerFromQuaternion(cloth_orientation)

            print(args.deform_init_ori)
            print(cloth_rot)

            init_r = R.from_euler('xyz', args.deform_init_ori).as_matrix()
            rot_r = R.from_euler('xyz', cloth_rot).as_matrix()
            cloth_position = args.deform_init_pos
            cloth_orientation = R.from_matrix(rot_r @ init_r).as_euler('xyz')

        else:
            cloth_position = args.deform_init_pos
            cloth_orientation = args.deform_init_ori


        deform_id = load_deform_object(
            sim, deform_obj, texture_path, args.deform_scale,
            # args.deform_init_pos, args.deform_init_ori,
            cloth_position, cloth_orientation,
            args.deform_bending_stiffness, args.deform_damping_stiffness,
            args.deform_elastic_stiffness, args.deform_friction_coeff,
            not args.disable_self_collision, debug)
        if scene_name == 'button':  # pin cloth edge for buttoning task
            assert ('deform_fixed_anchor_vertex_ids' in DEFORM_INFO[deform_obj])
            pin_fixed(sim, deform_id,
                      DEFORM_INFO[deform_obj]['deform_fixed_anchor_vertex_ids'])


        scene_info = copy.deepcopy(SCENE_INFO[scene_name])
        scene_info = apply_anchor_params(scene_name, scene_info, anchor_params)


        # TODO: load using baseposition and orientation first, save pc, and then transform
        # anchor, and anchor pc
        rigid_ids = []
        # for name, kwargs in SCENE_INFO[scene_name]['entities'].items():
        for name, kwargs in scene_info['entities'].items():
            # TODO: transform pose and orientation
            rgba_color = kwargs['rgbaColor'] if 'rgbaColor' in kwargs else None
            texture_file = None
            if 'useTexture' in kwargs and kwargs['useTexture']:
                #texture_file = self.get_texture_path(args.rigid_texture_file)
                texture_file = 'textures/rigid/darkbrownwood.jpg'

            id = load_rigid_object(
                sim, os.path.join(data_path, name), kwargs['globalScaling'],
                kwargs['basePosition'], kwargs['baseOrientation'],
                # rigid_position, rigid_orientation,
                kwargs.get('mass', 0.0), texture_file, rgba_color)
            rigid_ids.append(id)

        # storing rigid object point cloud, instead of rendering from RGBD throughout rollout
        if self.camera_config is not None:
            # first get anchor pcd
            _, pcd, ids = self.get_pcd_obs().values()
            pcd = pcd[ids > 0]

            # then, if rigid_rot and rigid_trans are specified, transform the rigid objects and point cloud
            if rigid_trans is not None:
                R_rigid = R.from_euler('xyz', rigid_rot)
                T_rigid = np.array(rigid_trans)

                for id, (name, kwargs) in zip(rigid_ids, scene_info['entities'].items()):
                    R_base = R.from_euler('xyz', kwargs['baseOrientation']).as_matrix()
                    T_base = np.array(kwargs['basePosition'])
                    # transforming base position and orientation
                    R_new = R.from_matrix(R_rigid.as_matrix() @ R_base).as_quat()
                    T_new = T_rigid + R_rigid.apply(T_base)
                    self.sim.resetBasePositionAndOrientation(id, T_new, R_new)

                pcd = R.from_euler('xyz', rigid_rot).apply(pcd) + np.array(rigid_trans)
            # store anchor pcd
            self.anchor_pcd = pcd

        # storing rigid pose TODO: need to update the interal representation of deform params, and object poses
        self.rigid_trans = [0, 0, 0] if rigid_trans is None else rigid_trans
        self.rigid_rot = [0, 0, 0] if rigid_rot is None else rigid_rot

        # Mark the goal and store intermediate info for reward computations.
        # goal_poses = SCENE_INFO[scene_name]['goal_pos']
        goal_poses = scene_info['goal_pos']
        if rigid_trans is not None:
            # TODO: update goal poses
            goal_poses = [R.from_euler('xyz', rigid_rot).apply(goal_pos) + rigid_trans
                          for goal_pos in goal_poses]

        if args.viz and debug:
            for i, goal_pos in enumerate(goal_poses):
                print(f'goal_pos{i}', goal_pos)
                alpha = 1 if i == 0 else 0.3  # primary vs secondary goal
                create_anchor_geom(sim, goal_pos, mass=0.0,
                                   rgba=(0, 1, 0, alpha), use_collision=False)
        if scene_name == 'foodpacking':
            # Save mesh info used for computing penalty for food packing task.
            _, vertices = get_mesh_data(sim, deform_id)
            vertices = np.array(vertices)
            relative_dist = np.linalg.norm(vertices - vertices[[0]], axis=1)
            self.deform_shape_sample_idx = np.random.choice(np.arange(
                vertices.shape[0]), 20, replace=False)
            self.deform_init_shape = relative_dist[self.deform_shape_sample_idx]

        return rigid_ids, deform_id, deform_obj, np.array(goal_poses), deform_params

    def seed(self, seed):
        np.random.seed(seed)

    def reset(self, cloth_rot=None, 
              rigid_trans=None, rigid_rot=None, 
              deform_params={},
              anchor_params={
                    'hanger_scale': 1.0,
                    'tallrod_scale': 1.0,
              }):
        self.stepnum = 0
        self.anchor_pcd = None
        self.episode_reward = 0.0
        self.anchors = {}
        self.vid_frames = []
        self.color_key = None
        self.anchor_params = anchor_params
        self.target_action = None

        if self.args.viz:  # no rendering during load
            self.sim.configureDebugVisualizer(pybullet.COV_ENABLE_RENDERING, 0)

        # Reset pybullet sim to clear out deformables and reload objects.
        # plane_texture_path = os.path.join(
        #     self.args.data_path,  self.get_texture_path(
        #         self.args.plane_texture_file))
        
        # FIXING THE PLANE TEXTURE FOR NOW
        plane_texture_path = os.path.join(
            self.args.data_path,
            'textures/plane/lightwood.jpg'
        )

        reset_bullet(self.args, self.sim, plane_texture=plane_texture_path)
        res = self.load_objects(self.sim, self.args, self.args.debug, cloth_rot, rigid_trans, rigid_rot, deform_params, anchor_params)
        self.rigid_ids, self.deform_id, self.deform_obj, self.goal_pos, self.deform_params = res
        load_floor(self.sim, plane_texture=plane_texture_path, debug=self.args.debug)

        # Special case for Procedural Cloth tasks that can have two holes:
        # reward is based on the closest hole.
        if self.args.env.startswith('HangProcCloth'):
            self.goal_pos = np.vstack((self.goal_pos, self.goal_pos))

        self.sim.stepSimulation()  # step once to get initial state

        debug_mrks = None
        if self.args.debug and self.args.viz:
           debug_mrks = self.debug_viz_true_loop()

        # Setup dynamic anchors.
        if not self.food_packing:
            self.make_anchors()

        # Set up viz.
        if self.args.viz:  # loading done, so enable debug rendering if needed
            time.sleep(0.1)  # wait for debug visualizer to catch up
            self.sim.stepSimulation()  # step once to get initial state
            self.sim.configureDebugVisualizer(pybullet.COV_ENABLE_RENDERING, 1)
            if debug_mrks is not None:
                input('Visualized true loops; press ENTER to continue')
                for mrk_id in debug_mrks:
                    # removeBody doesn't seem to work, so just make invisible
                    self.sim.changeVisualShape(mrk_id, -1,
                                               rgbaColor=[0, 0, 0, 0])

        # obs, _ = self.get_obs()
        obs = self.get_obs()

        # Updating rollout video
        if self.rollout_vid:
            self.vid_frames.append(
                self.render(mode='rgb_array', width=self.vid_width, height=self.vid_height)
            )
            action_pcd = obs['action_pcd']
            self.color_key = action_pcd[:, 0] + action_pcd[:, 2]
        return obs

    def make_anchors(self):
        preset_dynamic_anchor_vertices = get_preset_properties(
            DEFORM_INFO, self.deform_obj, 'deform_anchor_vertices')
        _, mesh = get_mesh_data(self.sim, self.deform_id)
        for i in range(self.num_anchors):  # make anchors
            anchor_init_pos = self.args.anchor_init_pos if (i % 2) == 0 else \
                self.args.other_anchor_init_pos
            anchor_id, anchor_pos, anchor_vertices = create_anchor(
                self.sim, anchor_init_pos, i,
                preset_dynamic_anchor_vertices, mesh)
            attach_anchor(self.sim, anchor_id, anchor_vertices, self.deform_id)
            self.anchors[anchor_id] = {'pos': anchor_pos,
                                       'vertices': anchor_vertices}

    def debug_viz_true_loop(self):
        # DEBUG visualize true loop vertices
        # Note: this function can be very slow when the number of ground truth
        # vertices marked is large, because it will create many visual elements.
        # So, use it sparingly (e.g. just only at trajectory start/end).
        if not hasattr(self.args, 'deform_true_loop_vertices'):
            return
        true_vs_lists = self.args.deform_true_loop_vertices
        _, vertex_positions = get_mesh_data(self.sim, self.deform_id)
        all_vs = np.array(vertex_positions)
        mrk_ids = []
        clrs = [cm.tab10(i) for i in range(len(true_vs_lists))]
        for l, loop_v_lst in enumerate(true_vs_lists):
            curr_vs = all_vs[loop_v_lst]
            for v in curr_vs:
                mrk_ids.append(create_anchor_geom(
                    self.sim, v, mass=0.0, radius=0.05,
                    rgba=clrs[l], use_collision=False))
        return mrk_ids

    def step(self, action, action_type='velocity', unscaled=False, tax3d=False):
        # TODO: this is hacky, maybe just remove the action_space.contains check
        if action_type == 'position':
            unscaled = True

        # print(action)
        if self.args.debug:
            print('action', action)
        if not unscaled:
            assert self.action_space.contains(action)
            # assert ((np.abs(action) <= 1.0).all()), 'action must be in [-1, 1]'
        action = action.reshape(self.num_anchors, -1)

        # Step through physics simulation.
        for sim_step in range(self.args.sim_steps_per_action):
            # self.do_action(action, unscaled)
            # self.do_action2(action, unscaled)
            if action_type == 'velocity':
                self.do_action_velocity(action, unscaled)
            elif action_type == 'position':
                self.do_action_position(action, unscaled, tax3d)
            else:
                raise ValueError(f'Unknown action type {action_type}')
            self.sim.stepSimulation()

        # Get next obs, reward, done.
        next_obs = self.get_obs()
        done = next_obs["done"]
        reward = self.get_reward()
        if done:  # if terminating early use reward from current step for rest
            reward *= (self.max_episode_len - self.stepnum)
        done = (done or self.stepnum >= self.max_episode_len)

        # Updating rollout video
        if self.rollout_vid:
            self.vid_frames.append(
                self.render(mode='rgb_array', width=self.vid_width, height=self.vid_height)
            )

        # Update episode info and call make_final_steps if needed.
        if done:
            if self.rollout_vid:
                pre_release_frame = self.render(mode='rgb_array', width=self.vid_width, height=self.vid_height)

            centroid_check, centroid_dist = self.check_centroid()
            info = self.make_final_steps()
            # polygon_check = self.check_polygon()
            centroid_check_post, _ = self.check_centroid()
            # print(centroid_check, polygon_check)
            # print(centroid_check, centroid_check_post)
            # success requires both checks to pass for at least one hole
            info['is_success'] = np.any(centroid_check * centroid_check_post)
            info['centroid_dist'] = np.mean(centroid_dist)

            last_rwd = self.get_reward() * HangBagTAX3D.FINAL_REWARD_MULT
            reward += last_rwd
            info['final_reward'] = reward

            # Returning rollout video
            if self.rollout_vid:
                info['vid_frames'] = self.vid_frames
                info['color_key'] = self.color_key
                info['viewmat'] = np.array(self._cam_viewmat)
                info['pre_release_frame'] = pre_release_frame

        else:
            info = {}

        self.episode_reward += reward  # update episode reward

        if self.args.debug and self.stepnum % 10 == 0:
            print(f'step {self.stepnum:d} reward {reward:0.4f}')
            if done:
                print(f'episode reward {self.episode_reward:0.4f}')
            
        self.stepnum += 1

        return next_obs, reward, done, info
    

    def do_action_velocity(self, action, unscaled):
        # Action is num_anchors x 3 for 3D velocity for anchors/grippers.
        # Assume action in [-1,1], convert to [-MAX_ACT_VEL, MAX_ACT_VEL].
        for i in range(self.num_anchors):
            command_anchor_velocity(
                self.sim, self.anchor_ids[i],
                HangBagTAX3D.unscale_vel(action[i], unscaled))

    def do_action_position(self, action, unscaled, tax3d):
        # uses basic proportional position control instead
        grip_obs = self.get_grip_obs()
        grip_obs = np.array([grip_obs[0:3], grip_obs[6:9]])

        for i in range(self.num_anchors):
            # clipping the magnitudes to prevent weird behavior
            p_delta = action[i] - grip_obs[i]
            delta_mag = np.linalg.norm(p_delta)
            clipped_action = grip_obs[i] + (p_delta / delta_mag) * min(delta_mag, 9)
            command_anchor_position(
                self.sim, self.anchor_ids[i],
                # action[i]
                clipped_action,
                tax3d=tax3d,
                task='hangbag'
            )
            # self.sim.addUserDebugPoints(
            #     [clipped_action], [[1, 0, 0]], pointSize=2
            # )
            # self.sim.addUserDebugPoints(
            #     [action[i]], [[0, 1, 0]], pointSize=4
            # )

    def make_final_steps(self):
        # We do no explicitly release the anchors, since this can create a jerk
        # and large forces.
        # release_anchor(self.sim, self.anchor_ids[0])
        # release_anchor(self.sim, self.anchor_ids[1])
        change_anchor_color_gray(self.sim, self.anchor_ids[0])
        change_anchor_color_gray(self.sim, self.anchor_ids[1])
        info = {'final_obs': []}
        for sim_step in range(HangBagTAX3D.STEPS_AFTER_DONE):
            # For lasso pull the string at the end to test lasso loop.
            # For other tasks noop action to let the anchors fall.
            if self.args.task.lower() == 'lasso':
                if sim_step % self.args.sim_steps_per_action == 0:
                    action = [10*HangBagTAX3D.MAX_ACT_VEL,
                              10*HangBagTAX3D.MAX_ACT_VEL, 0]
                    self.do_action(action, unscaled=True)
            self.sim.stepSimulation()
            if sim_step % self.args.sim_steps_per_action == 0:
                # next_obs, _ = self.get_obs()
                next_obs = self.get_obs()
                info['final_obs'].append(next_obs)

                # Updating rollout video
                if self.rollout_vid:
                    self.vid_frames.append(
                        self.render(mode='rgb_array', width=self.vid_width, height=self.vid_height)
                    )
        return info

    def end_episode(self, reward):
        """ Need to add this function so that point cloud observations can 
        be rendered before the call to make_final_steps in the step function."""
        info = self.make_final_steps()
        last_rwd = self.get_reward() * HangBagTAX3D.FINAL_REWARD_MULT
        # TODO: this threshold does not seem to be working well for all tasks.
        info['is_success'] = np.abs(last_rwd) < self.SUCESS_REWARD_TRESHOLD
        reward += last_rwd
        info['final_reward'] = reward
        # print(f'final_reward: {reward:.4f}')

        self.episode_reward += reward  # update episode reward
        self.stepnum += 1
        return info
    
    def end_episode_viz(self, width, height):
        change_anchor_color_gray(self.sim, self.anchor_ids[0])
        change_anchor_color_gray(self.sim, self.anchor_ids[1])
        frames = []
        for sim_step in range(HangBagTAX3D.STEPS_AFTER_DONE):
            # For lasso pull the string at the end to test lasso loop.
            # For other tasks noop action to let the anchors fall.
            self.sim.stepSimulation()
            if sim_step % self.args.sim_steps_per_action == 0:
                frames.append(Image.fromarray(self.render(mode='rgb_array', width=width, height=height)))
        return frames

    def get_pcd_obs(self):
        """ Grab Pointcloud observations based from the camera config. """

        # Grab pcd observation from the camera_config camera
        segmented_pcd, segmented_ids, img = ProcessCamera.render(
            self.sim, self.camera_config, width=self.args.cam_resolution,
            height=self.args.cam_resolution, object_ids=self.object_ids,
            return_rgb=True, retain_unknowns=True,
            debug=False)

        # Process the RGB image
        img = img[...,:-1] # drop the alpha
        if self.args.uint8_pixels:
            img = img.astype(np.uint8)  # already in [0,255]
        else:
            img = img.astype(np.float32)/255.0  # to [0,1]
            img = np.clip(img, 0, 1)
        if self.args.flat_obs:
            img = img.reshape(-1)
        atol = 0.0001
        if ((img < self.observation_space.low-atol).any() or
            (img > self.observation_space.high+atol).any()):
            print('img', img.shape, f'{np.min(img):e}, n{np.max(img):e}')
            assert self.observation_space.contains(img)

        # Package and return
        obs = {'img': img,
               'pcd': segmented_pcd,
               'ids': segmented_ids
              }

        return obs 
    

    def get_scene_pcd(self):
       # action pcd from ground truth mesh
       _, action_pcd = get_mesh_data(self.sim, self.deform_id)
       action_pcd = np.array(action_pcd)
       action_seg = np.ones(action_pcd.shape[0])

       # anchor pcd from scene camera render
       pcd_obs = self.get_pcd_obs()
       img, pcd, ids = pcd_obs.values()
       anchor_seg = ids[ids > 0]
       anchor_pcd = pcd[ids > 0]
       anchor_seg = anchor_seg * 0 # this is dumb, but keeps consistency with rest of demos

        # TODO: this should just return the point clouds, no need to seg here

       obs = {
            'action_pcd': np.array(action_pcd),
            'anchor_pcd': anchor_pcd,
            #'action_seg': action_seg,
            #'anchor_seg': anchor_seg,
        }
       return obs
    
    
    def get_obs(self):
        grip_obs = self.get_grip_obs()
        done = False
        grip_obs = np.nan_to_num(np.array(grip_obs))
        if (np.abs(grip_obs) > self.gripper_lims).any():  # at workspace lims
            if self.args.debug:
                print('clipping grip_obs', grip_obs)
            grip_obs = np.clip(
                grip_obs, -1.0*self.gripper_lims, self.gripper_lims)
            done = True
        # TODO: TAX3D environment should not need to return the image
        if self.args.cam_resolution <= 0:
            obs = grip_obs
        else:
            obs = self.render(mode='rgb_array', width=self.args.cam_resolution,
                              height=self.args.cam_resolution)
            if self.args.uint8_pixels:
                obs = obs.astype(np.uint8)  # already in [0,255]
            else:
                obs = obs.astype(np.float32)/255.0  # to [0,1]
                obs = np.clip(obs, 0, 1)
        if self.args.flat_obs:
            obs = obs.reshape(-1)
        atol = 0.0001
        if ((obs < self.observation_space.low-atol).any() or
            (obs > self.observation_space.high+atol).any()):
            print('obs', obs.shape, f'{np.min(obs):e}, n{np.max(obs):e}')
            assert self.observation_space.contains(obs)

        # --- Getting object-centric point clouds ---

        # action pcd from ground truth mesh
        _, action_pcd = get_mesh_data(self.sim, self.deform_id)
        action_pcd = np.array(action_pcd)
        anchor_pcd = self.anchor_pcd
  

        obs_dict = {
            'img': obs, # TODO: this could technically include gripper state - ignored either way
            'gripper_state': grip_obs,
            'done': done,
            'action_pcd': action_pcd,
            'anchor_pcd': anchor_pcd,
        }

        return obs_dict

    def get_grip_obs(self):
        anc_obs = []
        for i in range(self.num_anchors):
            pos, _ = self.sim.getBasePositionAndOrientation(
                self.anchor_ids[i])
            linvel, _ = self.sim.getBaseVelocity(self.anchor_ids[i])
            anc_obs.extend(pos)
            anc_obs.extend((np.array(linvel)/HangBagTAX3D.MAX_OBS_VEL))
        return anc_obs

    def get_reward(self, debug=False):
        _, vertex_positions = get_mesh_data(self.sim, self.deform_id)
        dist = []
        if not hasattr(self.args, 'deform_true_loop_vertices'):
            return 0.0  # no reward info without info about true loops
        # Compute distance from loop/hole to the corresponding target.
        num_holes_to_track = min(
            len(self.args.deform_true_loop_vertices), len(self.goal_pos))
        for i in range(num_holes_to_track):  # loop through goal vertices
            true_loop_vertices = self.args.deform_true_loop_vertices[i]
            goal_pos = self.goal_pos[i]
            pts = np.array(vertex_positions)
            cent_pts = pts[true_loop_vertices]
            cent_pts = cent_pts[~np.isnan(cent_pts).any(axis=1)]  # remove nans
            if len(cent_pts) == 0 or np.isnan(cent_pts).any():
                dist = HangBagTAX3D.WORKSPACE_BOX_SIZE*num_holes_to_track
                dist *= HangBagTAX3D.FINAL_REWARD_MULT
                # Save a screenshot for debugging.
                # obs = self.render(mode='rgb_array', width=300, height=300)
                # pth = f'nan_{self.args.env}_s{self.stepnum}.npy'
                # np.save(os.path.join(self.args.logdir, pth), obs)
                break
            cent_pos = cent_pts.mean(axis=0)
            if debug:
                print('cent_pos', cent_pos, 'goal_pos', goal_pos)
            dist.append(np.linalg.norm(cent_pos - goal_pos))

        if self.args.env.startswith('HangProcCloth'):
            dist = np.min(dist)
        else:
            dist = np.mean(dist)
        rwd = -1.0 * dist / HangBagTAX3D.WORKSPACE_BOX_SIZE
        return rwd
    

    def check_centroid(self):
        _, vertex_positions = get_mesh_data(self.sim, self.deform_id)
        centroid_checks = []
        centroid_dists = []
        num_holes_to_track = min(
            len(self.args.deform_true_loop_vertices), len(self.goal_pos)
        )
        for i in range(num_holes_to_track):
            true_loop_vertices = self.args.deform_true_loop_vertices[i]
            goal_pos = self.goal_pos[i]
            pts = np.array(vertex_positions)
            cent_pts = pts[true_loop_vertices]
            cent_pts = cent_pts[~np.isnan(cent_pts).any(axis=1)]
            cent_pos = cent_pts.mean(axis=0)
            dist = np.linalg.norm(cent_pos - goal_pos)
            print(dist)
            centroid_checks.append(dist < 1.4)
            centroid_dists.append(dist)
        return np.array(centroid_checks), np.array(centroid_dists)


    def check_polygon(self):
        _, vertex_positions = get_mesh_data(self.sim, self.deform_id)
        pts = np.array(vertex_positions)
        polygon_checks = []
        num_holes_to_track = min(
            len(self.args.deform_true_loop_vertices), len(self.goal_pos)
        )
        # first need to revert the rigid_rot (don't need to invert translation for this)
        R_rigid = R.from_euler('xyz', -np.array(self.rigid_rot))
        for i in range(num_holes_to_track):
            true_loop_vertices = self.args.deform_true_loop_vertices[i]
            # TODO: not sure if this ordering is correct
            # true_loop_vertices = order_loop_vertices(true_loop_vertices)
            goal_pos = self.goal_pos[i]

            cent_pts = pts[true_loop_vertices]
            polygon = Polygon(cent_pts[:, [0, 2]])
            convex_hull = polygon.convex_hull
            p_points = np.array(polygon.exterior.coords)
            c_points = np.array(convex_hull.exterior.coords)

            point = Point([goal_pos[0], goal_pos[2]])
            polygon_checks.append(convex_hull.contains(point))


            # from plotly.subplots import make_subplots
            # import plotly.graph_objects as go
            # fig = make_subplots(rows=1, cols=2)
            # poly = go.Scatter(
            #     x=p_points[:, 0],
            #     y=p_points[:, 1],
            #     mode='lines',
            #     fill='toself',
            # )
            # convex = go.Scatter(
            #     x=c_points[:, 0],
            #     y=c_points[:, 1],
            #     mode='lines',
            #     fill='toself',
            # )
            # goal_plot = go.Scatter(
            #     x=[point.x],
            #     y=[point.y],
            #     mode='markers',
            #     marker=dict(size=10, color='red')
            # )
            # fig.add_trace(poly, row=1, col=1)
            # fig.add_trace(convex, row=1, col=2)
            # fig.add_trace(goal_plot, row=1, col=2)
            # fig.show()

        return np.array(polygon_checks)



    def render(self, mode='rgb_array', width=300, height=300):
        assert (mode == 'rgb_array')
        w, h, rgba_px, _, _ = self.sim.getCameraImage(
            width=width, height=height,
            renderer=pybullet.ER_BULLET_HARDWARE_OPENGL,
            viewMatrix=self._cam_viewmat, **DEFAULT_CAM_PROJECTION)
        # If getCameraImage() returns a tuple instead of numpy array that
        # means that pybullet was installed without numpy being present.
        # Uninstall pybullet, then install numpy, then install pybullet.
        assert (isinstance(rgba_px, np.ndarray)), 'Install numpy, then pybullet'
        img = rgba_px[:, :, 0:3]
        return img
    
    def pseudo_expert_action(self, hole_id, speed_factor=1.0, rigid_rot=None, rigid_trans=None):
        """ Pseudo-expert action for demonstration generation. This is basic velocity control based on 
        vector from loop centroid to goal position."""
        # TODO: maybe switch entirely to position control for better demos, but this is low priority
        if self.target_action is not None:
            action = self.target_action
        else:
            # default goal pose
            # default_goal_pos = [0, 0.00, 8.2 * self.anchor_params['tallrod_scale']]
            default_goal_pos = [0.00, 1.28, 9]

            # getting goal position and loop centroid
            # goal_pos = self.goal_pos[hole_id]
            true_loop_vertices = self.args.deform_true_loop_vertices[hole_id]
            _, vertex_positions = get_mesh_data(self.sim, self.deform_id)
            vertex_positions = np.array(vertex_positions)
            centroid_points = vertex_positions[true_loop_vertices]
            centroid_points = centroid_points[~np.isnan(centroid_points).any(axis=1)]
            centroid = centroid_points.mean(axis=0)

            # getting the flow vector
            # flow = goal_pos - centroid

            # if rigid_rot and rigid_trans are provided, transform endpoints
            # otherwise, just use the default goal position


            flow = default_goal_pos - centroid
            # flow += np.array([0, -1.5, 0]) # grippers should go slightly past anchor
            flow += np.array([0, -0.5, 1.0])
            grip_obs = self.get_grip_obs()
            a1_pos = grip_obs[0:3]
            a2_pos = grip_obs[6:9]

            if rigid_rot is not None:
                # transforming default goal position
                R_default2goal = R.from_euler('xyz', rigid_rot)
                t_default2goal = np.array(rigid_trans)
                a1_act = R_default2goal.apply(a1_pos + flow) + t_default2goal
                a2_act = R_default2goal.apply(a2_pos + flow) + t_default2goal

            else:
                #a1_act = default_goal_pos - centroid
                #a2_act = default_goal_pos - centroid
                a1_act = a1_pos + flow
                a2_act = a2_pos + flow

            action = np.concatenate([a1_act, a2_act], axis=0).astype(np.float32)
            self.target_action = action

        # goal correction
        goal_pos = self.goal_pos[hole_id]
        true_loop_vertices = self.args.deform_true_loop_vertices[hole_id]
        _, vertex_positions = get_mesh_data(self.sim, self.deform_id)
        vertex_positions = np.array(vertex_positions)
        centroid_points = vertex_positions[true_loop_vertices]
        centroid_points = centroid_points[~np.isnan(centroid_points).any(axis=1)]
        centroid = centroid_points.mean(axis=0)

        correction = goal_pos - centroid
        correction /= max(np.linalg.norm(correction), 1.0)
        action = action + np.concatenate([correction, correction], axis=0).astype(np.float32)

        # gripper correction
        # TODO: do we need this?

        return action


        flow = goal_pos - centroid
        # hard-coded re-scaling
        a1_act = flow * 0.2 / 5 # 6.36
        a2_act = flow * 0.2 / 5 # 6.36
        # this is hacky way to increase success rate - slight upward velocity
        a1_act[2] += 0.01 / speed_factor
        a2_act[2] += 0.01 / speed_factor
        act = np.concatenate([a1_act, a2_act], axis=0).astype(np.float32)
        return act  * speed_factor
    
    def random_anchor_transform(self):
        z_rot = np.random.uniform(-np.pi / 3, np.pi / 3)
        rot = R.from_euler('z', 
            z_rot
        )#.as_matrix()
        transform = np.array([
            np.random.uniform() * 5 * np.power(-1, z_rot < 0),
            np.random.uniform() * -10,
            0.0
        ])
        return rot, transform
    
    def random_anchor_transform_ood(self):
        z_rot = np.random.uniform(-np.pi / 3, np.pi / 3)
        rot = R.from_euler('z', 
            z_rot
        )#.as_matrix()
        transform = np.array([
            np.random.uniform(5, 10) * np.power(-1, z_rot < 0),
            np.random.uniform() * -10,
            np.random.uniform(1, 5)
        ])
        return rot, transform