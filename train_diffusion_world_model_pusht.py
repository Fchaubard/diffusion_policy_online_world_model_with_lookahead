# train_diffusion_world_model_pusht.py
# This file trains a diffusion world model on the pusht environment to take in (St,At) -> (St+1 = the next image after taking action At from St, Rt+1 = the reward of the predicted St+1)  
# We use a U-Net with a cross attention layer At x St features. It will save off and overwrite checkpoints with a checkpoint config.txt with a unique id per run.

# TO RUN:
# Need to git clone https://github.com/real-stanford/diffusion_policy.git
# then need to conda activate robodiff
# then you can run this file with something like:

# python train_diffusion_world_model_pusht.py \
# --pred_horizon 16 \
# --obs_horizon 2 \
# --action_horizon 8 \
# --max_steps_pusht_environment 200 \
# --device cuda:1 \
# --unet_num_layers 2 \
# --base 64 \
# --heads 4 \
# --batch_size 32 \
# --num_diffusion_iters_action_policy 1000 \
# --beta_noise_start 1e-4 \
# --beta_noise_end 0.02 \
# --lr 0.0002 \
# --batches_per_episode 50 \
# --save_every 100




# diffusion policy imports

from typing import Tuple, Sequence, Dict, Union, Optional, Callable
import numpy as np
import math
import torch
import torch.nn as nn
import torchvision
import collections
import zarr

from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler
from tqdm.auto import tqdm

# env import
import gym
from gym import spaces
import pygame
import pymunk
import pymunk.pygame_util
from pymunk.space_debug_draw_options import SpaceDebugColor
from pymunk.vec2d import Vec2d
import shapely.geometry as sg
import cv2
import skimage.transform as st
from skvideo.io import vwrite
from IPython.display import Video
import gdown
import os


# **Environment**
positive_y_is_up: bool = False
"""Make increasing values of y point upwards.

When True::

    y
    ^
    |      . (3, 3)
    |
    |   . (2, 2)
    |
    +------ > x

When False::

    +------ > x
    |
    |   . (2, 2)
    |
    |      . (3, 3)
    v
    y

"""

def to_pygame(p: Tuple[float, float], surface: pygame.Surface) -> Tuple[int, int]:
    """Convenience method to convert pymunk coordinates to pygame surface
    local coordinates.

    Note that in case positive_y_is_up is False, this function wont actually do
    anything except converting the point to integers.
    """
    if positive_y_is_up:
        return round(p[0]), surface.get_height() - round(p[1])
    else:
        return round(p[0]), round(p[1])


def light_color(color: SpaceDebugColor):
    color = np.minimum(1.2 * np.float32([color.r, color.g, color.b, color.a]), np.float32([255]))
    color = SpaceDebugColor(r=color[0], g=color[1], b=color[2], a=color[3])
    return color

class DrawOptions(pymunk.SpaceDebugDrawOptions):
    def __init__(self, surface: pygame.Surface) -> None:
        """Draw a pymunk.Space on a pygame.Surface object.

        Typical usage::

        >>> import pymunk
        >>> surface = pygame.Surface((10,10))
        >>> space = pymunk.Space()
        >>> options = pymunk.pygame_util.DrawOptions(surface)
        >>> space.debug_draw(options)

        You can control the color of a shape by setting shape.color to the color
        you want it drawn in::

        >>> c = pymunk.Circle(None, 10)
        >>> c.color = pygame.Color("pink")

        See pygame_util.demo.py for a full example

        Since pygame uses a coordiante system where y points down (in contrast
        to many other cases), you either have to make the physics simulation
        with Pymunk also behave in that way, or flip everything when you draw.

        The easiest is probably to just make the simulation behave the same
        way as Pygame does. In that way all coordinates used are in the same
        orientation and easy to reason about::

        >>> space = pymunk.Space()
        >>> space.gravity = (0, -1000)
        >>> body = pymunk.Body()
        >>> body.position = (0, 0) # will be positioned in the top left corner
        >>> space.debug_draw(options)

        To flip the drawing its possible to set the module property
        :py:data:`positive_y_is_up` to True. Then the pygame drawing will flip
        the simulation upside down before drawing::

        >>> positive_y_is_up = True
        >>> body = pymunk.Body()
        >>> body.position = (0, 0)
        >>> # Body will be position in bottom left corner

        :Parameters:
                surface : pygame.Surface
                    Surface that the objects will be drawn on
        """
        self.surface = surface
        super(DrawOptions, self).__init__()

    def draw_circle(
        self,
        pos: Vec2d,
        angle: float,
        radius: float,
        outline_color: SpaceDebugColor,
        fill_color: SpaceDebugColor,
    ) -> None:
        p = to_pygame(pos, self.surface)

        pygame.draw.circle(self.surface, fill_color.as_int(), p, round(radius), 0)
        pygame.draw.circle(self.surface, light_color(fill_color).as_int(), p, round(radius-4), 0)

        circle_edge = pos + Vec2d(radius, 0).rotated(angle)
        p2 = to_pygame(circle_edge, self.surface)
        line_r = 2 if radius > 20 else 1
        # pygame.draw.lines(self.surface, outline_color.as_int(), False, [p, p2], line_r)

    def draw_segment(self, a: Vec2d, b: Vec2d, color: SpaceDebugColor) -> None:
        p1 = to_pygame(a, self.surface)
        p2 = to_pygame(b, self.surface)

        pygame.draw.aalines(self.surface, color.as_int(), False, [p1, p2])

    def draw_fat_segment(
        self,
        a: Tuple[float, float],
        b: Tuple[float, float],
        radius: float,
        outline_color: SpaceDebugColor,
        fill_color: SpaceDebugColor,
    ) -> None:
        p1 = to_pygame(a, self.surface)
        p2 = to_pygame(b, self.surface)

        r = round(max(1, radius * 2))
        pygame.draw.lines(self.surface, fill_color.as_int(), False, [p1, p2], r)
        if r > 2:
            orthog = [abs(p2[1] - p1[1]), abs(p2[0] - p1[0])]
            if orthog[0] == 0 and orthog[1] == 0:
                return
            scale = radius / (orthog[0] * orthog[0] + orthog[1] * orthog[1]) ** 0.5
            orthog[0] = round(orthog[0] * scale)
            orthog[1] = round(orthog[1] * scale)
            points = [
                (p1[0] - orthog[0], p1[1] - orthog[1]),
                (p1[0] + orthog[0], p1[1] + orthog[1]),
                (p2[0] + orthog[0], p2[1] + orthog[1]),
                (p2[0] - orthog[0], p2[1] - orthog[1]),
            ]
            pygame.draw.polygon(self.surface, fill_color.as_int(), points)
            pygame.draw.circle(
                self.surface,
                fill_color.as_int(),
                (round(p1[0]), round(p1[1])),
                round(radius),
            )
            pygame.draw.circle(
                self.surface,
                fill_color.as_int(),
                (round(p2[0]), round(p2[1])),
                round(radius),
            )

    def draw_polygon(
        self,
        verts: Sequence[Tuple[float, float]],
        radius: float,
        outline_color: SpaceDebugColor,
        fill_color: SpaceDebugColor,
    ) -> None:
        ps = [to_pygame(v, self.surface) for v in verts]
        ps += [ps[0]]

        radius = 2
        pygame.draw.polygon(self.surface, light_color(fill_color).as_int(), ps)

        if radius > 0:
            for i in range(len(verts)):
                a = verts[i]
                b = verts[(i + 1) % len(verts)]
                self.draw_fat_segment(a, b, radius, fill_color, fill_color)

    def draw_dot(
        self, size: float, pos: Tuple[float, float], color: SpaceDebugColor
    ) -> None:
        p = to_pygame(pos, self.surface)
        pygame.draw.circle(self.surface, color.as_int(), p, round(size), 0)


def pymunk_to_shapely(body, shapes):
    geoms = list()
    for shape in shapes:
        if isinstance(shape, pymunk.shapes.Poly):
            verts = [body.local_to_world(v) for v in shape.get_vertices()]
            verts += [verts[0]]
            geoms.append(sg.Polygon(verts))
        else:
            raise RuntimeError(f'Unsupported shape type {type(shape)}')
    geom = sg.MultiPolygon(geoms)
    return geom

# env
class PushTEnv(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 10}
    reward_range = (0., 1.)

    def __init__(self,
            legacy=False,
            block_cog=None, damping=None,
            render_action=True,
            render_size=96,
            reset_to_state=None
        ):
        self._seed = None
        self.seed()
        self.window_size = ws = 512  # The size of the PyGame window
        self.render_size = render_size
        self.sim_hz = 100
        # Local controller params.
        self.k_p, self.k_v = 100, 20    # PD control.z
        self.control_hz = self.metadata['video.frames_per_second']
        # legcay set_state for data compatiblity
        self.legacy = legacy

        # agent_pos, block_pos, block_angle
        self.observation_space = spaces.Box(
            low=np.array([0,0,0,0,0], dtype=np.float64),
            high=np.array([ws,ws,ws,ws,np.pi*2], dtype=np.float64),
            shape=(5,),
            dtype=np.float64
        )

        # positional goal for agent
        self.action_space = spaces.Box(
            low=np.array([0,0], dtype=np.float64),
            high=np.array([ws,ws], dtype=np.float64),
            shape=(2,),
            dtype=np.float64
        )

        self.block_cog = block_cog
        self.damping = damping
        self.render_action = render_action

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None
        self.screen = None

        self.space = None
        self.teleop = None
        self.render_buffer = None
        self.latest_action = None
        self.reset_to_state = reset_to_state

    def reset(self):
        seed = self._seed
        self._setup()
        if self.block_cog is not None:
            self.block.center_of_gravity = self.block_cog
        if self.damping is not None:
            self.space.damping = self.damping

        # use legacy RandomState for compatiblity
        state = self.reset_to_state
        if state is None:
            rs = np.random.RandomState(seed=seed)
            state = np.array([
                rs.randint(50, 450), rs.randint(50, 450),
                rs.randint(100, 400), rs.randint(100, 400),
                rs.randn() * 2 * np.pi - np.pi
                ])
        self._set_state(state)

        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def step(self, action):
        dt = 1.0 / self.sim_hz
        self.n_contact_points = 0
        n_steps = self.sim_hz // self.control_hz
        if action is not None:
            self.latest_action = action
            for i in range(n_steps):
                # Step PD control.
                # self.agent.velocity = self.k_p * (act - self.agent.position)    # P control works too.
                acceleration = self.k_p * (action - self.agent.position) + self.k_v * (Vec2d(0, 0) - self.agent.velocity)
                self.agent.velocity += acceleration * dt

                # Step physics.
                self.space.step(dt)

        # compute reward
        goal_body = self._get_goal_pose_body(self.goal_pose)
        goal_geom = pymunk_to_shapely(goal_body, self.block.shapes)
        block_geom = pymunk_to_shapely(self.block, self.block.shapes)

        intersection_area = goal_geom.intersection(block_geom).area
        goal_area = goal_geom.area
        coverage = intersection_area / goal_area
        reward = np.clip(coverage / self.success_threshold, 0, 1)
        done = coverage > self.success_threshold
        terminated = done
        truncated = done

        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def render(self, mode):
        return self._render_frame(mode)

    def teleop_agent(self):
        TeleopAgent = collections.namedtuple('TeleopAgent', ['act'])
        def act(obs):
            act = None
            mouse_position = pymunk.pygame_util.from_pygame(Vec2d(*pygame.mouse.get_pos()), self.screen)
            if self.teleop or (mouse_position - self.agent.position).length < 30:
                self.teleop = True
                act = mouse_position
            return act
        return TeleopAgent(act)

    def _get_obs(self):
        obs = np.array(
            tuple(self.agent.position) \
            + tuple(self.block.position) \
            + (self.block.angle % (2 * np.pi),))
        return obs

    def _get_goal_pose_body(self, pose):
        mass = 1
        inertia = pymunk.moment_for_box(mass, (50, 100))
        body = pymunk.Body(mass, inertia)
        # preserving the legacy assignment order for compatibility
        # the order here dosn't matter somehow, maybe because CoM is aligned with body origin
        body.position = pose[:2].tolist()
        body.angle = pose[2]
        return body

    def _get_info(self):
        n_steps = self.sim_hz // self.control_hz
        n_contact_points_per_step = int(np.ceil(self.n_contact_points / n_steps))
        info = {
            'pos_agent': np.array(self.agent.position),
            'vel_agent': np.array(self.agent.velocity),
            'block_pose': np.array(list(self.block.position) + [self.block.angle]),
            'goal_pose': self.goal_pose,
            'n_contacts': n_contact_points_per_step}
        return info

    def _render_frame(self, mode):

        if self.window is None and mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        self.screen = canvas

        draw_options = DrawOptions(canvas)

        # Draw goal pose.
        goal_body = self._get_goal_pose_body(self.goal_pose)
        for shape in self.block.shapes:
            goal_points = [pymunk.pygame_util.to_pygame(goal_body.local_to_world(v), draw_options.surface) for v in shape.get_vertices()]
            goal_points += [goal_points[0]]
            pygame.draw.polygon(canvas, self.goal_color, goal_points)

        # Draw agent and block.
        self.space.debug_draw(draw_options)

        if mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # the clock is aleady ticked during in step for "human"


        img = np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )
        img = cv2.resize(img, (self.render_size, self.render_size))
        if self.render_action:
            if self.render_action and (self.latest_action is not None):
                action = np.array(self.latest_action)
                coord = (action / 512 * 96).astype(np.int32)
                marker_size = int(8/96*self.render_size)
                thickness = int(1/96*self.render_size)
                cv2.drawMarker(img, coord,
                    color=(255,0,0), markerType=cv2.MARKER_CROSS,
                    markerSize=marker_size, thickness=thickness)
        return img


    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

    def seed(self, seed=None):
        if seed is None:
            seed = np.random.randint(0,25536)
        self._seed = seed
        self.np_random = np.random.default_rng(seed)

    # OLD API
    def _handle_collision(self, arbiter, space, data):
        self.n_contact_points += len(arbiter.contact_point_set.points)

    def _set_state(self, state):
        if isinstance(state, np.ndarray):
            state = state.tolist()
        pos_agent = state[:2]
        pos_block = state[2:4]
        rot_block = state[4]
        self.agent.position = pos_agent
        # setting angle rotates with respect to center of mass
        # therefore will modify the geometric position
        # if not the same as CoM
        # therefore should be modified first.
        if self.legacy:
            # for compatiblity with legacy data
            self.block.position = pos_block
            self.block.angle = rot_block
        else:
            self.block.angle = rot_block
            self.block.position = pos_block

        # Run physics to take effect
        self.space.step(1.0 / self.sim_hz)

    def _set_state_local(self, state_local):
        agent_pos_local = state_local[:2]
        block_pose_local = state_local[2:]
        tf_img_obj = st.AffineTransform(
            translation=self.goal_pose[:2],
            rotation=self.goal_pose[2])
        tf_obj_new = st.AffineTransform(
            translation=block_pose_local[:2],
            rotation=block_pose_local[2]
        )
        tf_img_new = st.AffineTransform(
            matrix=tf_img_obj.params @ tf_obj_new.params
        )
        agent_pos_new = tf_img_new(agent_pos_local)
        new_state = np.array(
            list(agent_pos_new[0]) + list(tf_img_new.translation) \
                + [tf_img_new.rotation])
        self._set_state(new_state)
        return new_state

    def _setup(self):
        self.space = pymunk.Space()
        self.space.gravity = 0, 0
        self.space.damping = 0
        self.teleop = False
        self.render_buffer = list()

        # Add walls.
        walls = [
            self._add_segment((5, 506), (5, 5), 2),
            self._add_segment((5, 5), (506, 5), 2),
            self._add_segment((506, 5), (506, 506), 2),
            self._add_segment((5, 506), (506, 506), 2)
        ]
        self.space.add(*walls)

        # Add agent, block, and goal zone.
        self.agent = self.add_circle((256, 400), 15)
        self.block = self.add_tee((256, 300), 0)
        self.goal_color = pygame.Color('LightGreen')
        self.goal_pose = np.array([256,256,np.pi/4])  # x, y, theta (in radians)

        # Add collision handeling
        # OLD API
        # self.collision_handeler = self.space.add_collision_handler(0, 0)
        # self.collision_handeler.post_solve = self._handle_collision
        # NEW API
        self.space.on_collision(0, 0, post_solve=self._handle_collision)
        
        self.n_contact_points = 0

        self.max_score = 50 * 100
        self.success_threshold = 0.95    # 95% coverage.

    def _add_segment(self, a, b, radius):
        shape = pymunk.Segment(self.space.static_body, a, b, radius)
        shape.color = pygame.Color('LightGray')    # https://htmlcolorcodes.com/color-names
        return shape

    def add_circle(self, position, radius):
        body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        body.position = position
        body.friction = 1
        shape = pymunk.Circle(body, radius)
        shape.color = pygame.Color('RoyalBlue')
        self.space.add(body, shape)
        return body

    def add_box(self, position, height, width):
        mass = 1
        inertia = pymunk.moment_for_box(mass, (height, width))
        body = pymunk.Body(mass, inertia)
        body.position = position
        shape = pymunk.Poly.create_box(body, (height, width))
        shape.color = pygame.Color('LightSlateGray')
        self.space.add(body, shape)
        return body

    def add_tee(self, position, angle, scale=30, color='LightSlateGray', mask=pymunk.ShapeFilter.ALL_MASKS()):
        mass = 1
        length = 4
        vertices1 = [(-length*scale/2, scale),
                                 ( length*scale/2, scale),
                                 ( length*scale/2, 0),
                                 (-length*scale/2, 0)]
        inertia1 = pymunk.moment_for_poly(mass, vertices=vertices1)
        vertices2 = [(-scale/2, scale),
                                 (-scale/2, length*scale),
                                 ( scale/2, length*scale),
                                 ( scale/2, scale)]
        inertia2 = pymunk.moment_for_poly(mass, vertices=vertices1)
        body = pymunk.Body(mass, inertia1 + inertia2)
        shape1 = pymunk.Poly(body, vertices1)
        shape2 = pymunk.Poly(body, vertices2)
        shape1.color = pygame.Color(color)
        shape2.color = pygame.Color(color)
        shape1.filter = pymunk.ShapeFilter(mask=mask)
        shape2.filter = pymunk.ShapeFilter(mask=mask)
        body.center_of_gravity = (shape1.center_of_gravity + shape2.center_of_gravity) / 2
        body.position = position
        body.angle = angle
        body.friction = 1
        self.space.add(body, shape1, shape2)
        return body


class PushTImageEnv(PushTEnv):
    metadata = {"render.modes": ["rgb_array"], "video.frames_per_second": 10}

    def __init__(self,
            legacy=False,
            block_cog=None,
            damping=None,
            render_size=96):
        super().__init__(
            legacy=legacy,
            block_cog=block_cog,
            damping=damping,
            render_size=render_size,
            render_action=False)
        ws = self.window_size
        self.observation_space = spaces.Dict({
            'image': spaces.Box(
                low=0,
                high=1,
                shape=(3,render_size,render_size),
                dtype=np.float32
            ),
            'agent_pos': spaces.Box(
                low=0,
                high=ws,
                shape=(2,),
                dtype=np.float32
            )
        })
        self.render_cache = None

    def _get_obs(self):
        img = super()._render_frame(mode='rgb_array')

        agent_pos = np.array(self.agent.position)
        img_obs = np.moveaxis(img.astype(np.float32) / 255, -1, 0)
        obs = {
            'image': img_obs,
            'agent_pos': agent_pos
        }

        # draw action
        if self.latest_action is not None:
            action = np.array(self.latest_action)
            coord = (action / 512 * 96).astype(np.int32)
            marker_size = int(8/96*self.render_size)
            thickness = int(1/96*self.render_size)
            cv2.drawMarker(img, coord,
                color=(255,0,0), markerType=cv2.MARKER_CROSS,
                markerSize=marker_size, thickness=thickness)
        self.render_cache = img

        return obs

    def render(self, mode):
        assert mode == 'rgb_array'

        if self.render_cache is None:
            self._get_obs()

        return self.render_cache













class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class Downsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)

class Upsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Conv1dBlock(nn.Module):
    '''
        Conv1d --> GroupNorm --> Mish
    '''

    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)


class ConditionalResidualBlock1D(nn.Module):
    def __init__(self,
            in_channels,
            out_channels,
            cond_dim,
            kernel_size=3,
            n_groups=8):
        super().__init__()

        self.blocks = nn.ModuleList([
            Conv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups),
            Conv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups),
        ])

        # FiLM modulation https://arxiv.org/abs/1709.07871
        # predicts per-channel scale and bias
        cond_channels = out_channels * 2
        self.out_channels = out_channels
        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, cond_channels),
            nn.Unflatten(-1, (-1, 1))
        )

        # make sure dimensions compatible
        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) \
            if in_channels != out_channels else nn.Identity()

    def forward(self, x, cond):
        '''
            x : [ batch_size x in_channels x horizon ]
            cond : [ batch_size x cond_dim]

            returns:
            out : [ batch_size x out_channels x horizon ]
        '''
        out = self.blocks[0](x)
        embed = self.cond_encoder(cond)

        embed = embed.reshape(
            embed.shape[0], 2, self.out_channels, 1)
        scale = embed[:,0,...]
        bias = embed[:,1,...]
        out = scale * out + bias

        out = self.blocks[1](out)
        out = out + self.residual_conv(x)
        return out


class ConditionalUnet1D(nn.Module):
    def __init__(self,
        input_dim,
        global_cond_dim,
        diffusion_step_embed_dim=256,
        down_dims=[256,512,1024],
        kernel_size=5,
        n_groups=8
        ):
        """
        input_dim: Dim of actions.
        global_cond_dim: Dim of global conditioning applied with FiLM
          in addition to diffusion step embedding. This is usually obs_horizon * obs_dim
        diffusion_step_embed_dim: Size of positional encoding for diffusion iteration k
        down_dims: Channel size for each UNet level.
          The length of this array determines numebr of levels.
        kernel_size: Conv kernel size
        n_groups: Number of groups for GroupNorm
        """

        super().__init__()
        all_dims = [input_dim] + list(down_dims)
        start_dim = down_dims[0]

        dsed = diffusion_step_embed_dim
        diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed),
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),
        )
        cond_dim = dsed + global_cond_dim

        in_out = list(zip(all_dims[:-1], all_dims[1:]))
        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList([
            ConditionalResidualBlock1D(
                mid_dim, mid_dim, cond_dim=cond_dim,
                kernel_size=kernel_size, n_groups=n_groups
            ),
            ConditionalResidualBlock1D(
                mid_dim, mid_dim, cond_dim=cond_dim,
                kernel_size=kernel_size, n_groups=n_groups
            ),
        ])

        down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            down_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(
                    dim_in, dim_out, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups),
                ConditionalResidualBlock1D(
                    dim_out, dim_out, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups),
                Downsample1d(dim_out) if not is_last else nn.Identity()
            ]))

        up_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            up_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(
                    dim_out*2, dim_in, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups),
                ConditionalResidualBlock1D(
                    dim_in, dim_in, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups),
                Upsample1d(dim_in) if not is_last else nn.Identity()
            ]))

        final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size=kernel_size),
            nn.Conv1d(start_dim, input_dim, 1),
        )

        self.diffusion_step_encoder = diffusion_step_encoder
        self.up_modules = up_modules
        self.down_modules = down_modules
        self.final_conv = final_conv

        print("number of parameters: {:e}".format(
            sum(p.numel() for p in self.parameters()))
        )

    def forward(self,
            sample: torch.Tensor,
            timestep: Union[torch.Tensor, float, int],
            global_cond=None):
        """
        x: (B,T,input_dim)
        timestep: (B,) or int, diffusion step
        global_cond: (B,global_cond_dim)
        output: (B,T,input_dim)
        """
        # (B,T,C)
        sample = sample.moveaxis(-1,-2)
        # (B,C,T)

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            timesteps = torch.tensor([timesteps], dtype=torch.float, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])

        global_feature = self.diffusion_step_encoder(timesteps)

        if global_cond is not None:
            global_feature = torch.cat([
                global_feature, global_cond
            ], axis=-1)

        x = sample
        h = []
        for idx, (resnet, resnet2, downsample) in enumerate(self.down_modules):
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            h.append(x)
            x = downsample(x)

        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)

        for idx, (resnet, resnet2, upsample) in enumerate(self.up_modules):
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            x = upsample(x)

        x = self.final_conv(x)

        # (B,C,T)
        x = x.moveaxis(-1,-2)
        # (B,T,C)
        return x
#@markdown ### **Vision Encoder**
#@markdown
#@markdown Defines helper functions:
#@markdown - `get_resnet` to initialize standard ResNet vision encoder
#@markdown - `replace_bn_with_gn` to replace all BatchNorm layers with GroupNorm

def get_resnet(name:str, weights=None, **kwargs) -> nn.Module:
    """
    name: resnet18, resnet34, resnet50
    weights: "IMAGENET1K_V1", None
    """
    # Use standard ResNet implementation from torchvision
    func = getattr(torchvision.models, name)
    resnet = func(weights=weights, **kwargs)

    # remove the final fully connected layer
    # for resnet18, the output dim should be 512
    resnet.fc = torch.nn.Identity()
    return resnet


def replace_submodules(
        root_module: nn.Module,
        predicate: Callable[[nn.Module], bool],
        func: Callable[[nn.Module], nn.Module]) -> nn.Module:
    """
    Replace all submodules selected by the predicate with
    the output of func.

    predicate: Return true if the module is to be replaced.
    func: Return new module to use.
    """
    if predicate(root_module):
        return func(root_module)

    bn_list = [k.split('.') for k, m
        in root_module.named_modules(remove_duplicate=True)
        if predicate(m)]
    for *parent, k in bn_list:
        parent_module = root_module
        if len(parent) > 0:
            parent_module = root_module.get_submodule('.'.join(parent))
        if isinstance(parent_module, nn.Sequential):
            src_module = parent_module[int(k)]
        else:
            src_module = getattr(parent_module, k)
        tgt_module = func(src_module)
        if isinstance(parent_module, nn.Sequential):
            parent_module[int(k)] = tgt_module
        else:
            setattr(parent_module, k, tgt_module)
    # verify that all modules are replaced
    bn_list = [k.split('.') for k, m
        in root_module.named_modules(remove_duplicate=True)
        if predicate(m)]
    assert len(bn_list) == 0
    return root_module

def replace_bn_with_gn(
    root_module: nn.Module,
    features_per_group: int=16) -> nn.Module:
    """
    Relace all BatchNorm layers with GroupNorm.
    """
    replace_submodules(
        root_module=root_module,
        predicate=lambda x: isinstance(x, nn.BatchNorm2d),
        func=lambda x: nn.GroupNorm(
            num_groups=x.num_features//features_per_group,
            num_channels=x.num_features)
    )
    return root_module
def normalize_data(data, stats):
    # nomalize to [0,1]
    ndata = (data - stats['min']) / (stats['max'] - stats['min'])
    # normalize to [-1, 1]
    ndata = ndata * 2 - 1
    return ndata

def unnormalize_data(ndata, stats):
    ndata = (ndata + 1) / 2
    data = ndata * (stats['max'] - stats['min']) + stats['min']
    return data





def get_resnet(name:str, weights=None, **kwargs) -> nn.Module:
    """
    name: resnet18, resnet34, resnet50
    weights: "IMAGENET1K_V1", None
    """
    # Use standard ResNet implementation from torchvision
    func = getattr(torchvision.models, name)
    resnet = func(weights=weights, **kwargs)

    # remove the final fully connected layer
    # for resnet18, the output dim should be 512
    resnet.fc = torch.nn.Identity()
    return resnet


def replace_submodules(
        root_module: nn.Module,
        predicate: Callable[[nn.Module], bool],
        func: Callable[[nn.Module], nn.Module]) -> nn.Module:
    """
    Replace all submodules selected by the predicate with
    the output of func.

    predicate: Return true if the module is to be replaced.
    func: Return new module to use.
    """
    if predicate(root_module):
        return func(root_module)

    bn_list = [k.split('.') for k, m
        in root_module.named_modules(remove_duplicate=True)
        if predicate(m)]
    for *parent, k in bn_list:
        parent_module = root_module
        if len(parent) > 0:
            parent_module = root_module.get_submodule('.'.join(parent))
        if isinstance(parent_module, nn.Sequential):
            src_module = parent_module[int(k)]
        else:
            src_module = getattr(parent_module, k)
        tgt_module = func(src_module)
        if isinstance(parent_module, nn.Sequential):
            parent_module[int(k)] = tgt_module
        else:
            setattr(parent_module, k, tgt_module)
    # verify that all modules are replaced
    bn_list = [k.split('.') for k, m
        in root_module.named_modules(remove_duplicate=True)
        if predicate(m)]
    assert len(bn_list) == 0
    return root_module

def replace_bn_with_gn(
    root_module: nn.Module,
    features_per_group: int=16) -> nn.Module:
    """
    Relace all BatchNorm layers with GroupNorm.
    """
    replace_submodules(
        root_module=root_module,
        predicate=lambda x: isinstance(x, nn.BatchNorm2d),
        func=lambda x: nn.GroupNorm(
            num_groups=x.num_features//features_per_group,
            num_channels=x.num_features)
    )
    return root_module


# STAGE 1: Generate the batch
# ──────────────────────────────────────────────────────────────────────────────
#  SingleStatePushTEpisodicDataset
#  Collects (image_t , action_t)  →  (image_{t+1}, reward_{t+1}) pairs
#  for PushTImageEnv using a diffusion-policy network with ε-greedy random
#  exploration.  Ready for use with torch.utils.data.DataLoader.
# ──────────────────────────────────────────────────────────────────────────────

import collections
import random
from typing import List, Tuple, Dict, Any

import numpy as np
import torch
from torch.utils.data import Dataset
# import tqdm


class SingleStatePushTEpisodicDataset(Dataset):
    """
    Parameters
    ----------
    env_instance : gym.Env
        Pre-constructed PushTImageEnv.
    policy_net : Dict[str, torch.nn.Module]
        Must contain 'vision_encoder' and 'noise_pred_net'.
    policy_scheduler : object
        Must implement .set_timesteps(T) and
        .step(model_output, timestep, sample) → object with .prev_sample.
    policy_config : dict
        Required keys:
            obs_horizon, pred_horizon, action_horizon,
            action_dim, num_diffusion_iters, stats (min/max dicts)
            Optional: max_steps (defaults to 200).
    num_episodes : int
        Number of episodes to sample.
    random_policy_percentage : float
        Probability ∈[0,1] of taking a random action at each macro-step.
    data_collection_device : str or torch.device
        Device on which the policy network is evaluated.
    """

    # ──────────────────────────────── init ──────────────────────────────── #
    def __init__(
        self,
        env_instance,
        policy_net: Dict[str, torch.nn.Module],
        policy_scheduler,
        policy_config: Dict[str, Any],
        num_episodes: int = 5,
        random_policy_percentage: float = 0.0, 
        data_collection_device="cuda:0",
    ):
        super().__init__()

        # ---------- store / move modules ---------------------------------- #
        self.device = torch.device(data_collection_device)
        self.env = env_instance
        self.policy_net = {k: v.to(self.device).eval() for k, v in policy_net.items()}
        self.scheduler = policy_scheduler

        # ---------- config ------------------------------------------------- #
        self.obs_horizon: int = policy_config["obs_horizon"]
        self.pred_horizon: int = policy_config["pred_horizon"]
        self.action_horizon: int = policy_config["action_horizon"]
        self.action_dim: int = policy_config["action_dim"]
        self.num_diffusion_iters: int = policy_config["num_diffusion_iters"]
        self.stats: Dict[str, Dict[str, np.ndarray]] = policy_config["stats"]
        self.max_steps: int = int(policy_config.get("max_steps", 200))

        self.random_policy_percentage = float(random_policy_percentage)
        self._py_rng = random.Random()              # python RNG (for coin-flip)
        self._np_rng = np.random.default_rng()      # numpy RNG (for vectors)

        # ---------- buffers ------------------------------------------------ #
        self._images_t:   List[torch.Tensor] = []   # state_t  image   (FloatTensor [3,96,96])
        self._actions:    List[torch.Tensor] = []   # action_t         (FloatTensor [action_dim])
        self._images_tp1: List[torch.Tensor] = []   # state_{t+1} image
        self._rewards:    List[float]        = []   # reward_{t+1}

        # ---------- collect data ------------------------------------------ #
        self._rollout(num_episodes)

    # ─────────────────────────── public API ─────────────────────────────── #
    def __len__(self) -> int:
        return len(self._images_t)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        image_t    : FloatTensor [3,96,96]  ∈ [0,1]
        action_t   : FloatTensor [action_dim]
        image_tp1  : FloatTensor [3,96,96]  ∈ [0,1]
        reward_tp1 : FloatTensor scalar
        """
        return (
            self._images_t[idx],
            self._actions[idx],
            self._images_tp1[idx],
            torch.tensor(self._rewards[idx], dtype=torch.float32),
        )

    # ────────────────────────── helper utils ────────────────────────────── #
    @staticmethod
    def _normalize(arr: np.ndarray, stats: Dict[str, np.ndarray]) -> np.ndarray:
        return (arr - stats["min"]) / (stats["max"] - stats["min"])

    @staticmethod
    def _unnormalize(arr: np.ndarray, stats: Dict[str, np.ndarray]) -> np.ndarray:
        return arr * (stats["max"] - stats["min"]) + stats["min"]

    def _sample_random_action(self) -> np.ndarray:
        """
        Uniform in the *unnormalised* action box.
        Returns ndarray shape [action_dim], dtype float32.
        """
        low  = self.stats["action"]["min"]
        high = self.stats["action"]["max"]
        return (low + self._np_rng.random(self.action_dim) * (high - low)).astype(np.float32)

    # # ───────────────────── diffusion-policy inference ───────────────────── #
    # @torch.no_grad()
    # def _infer_policy_actions(self, obs_deque: collections.deque) -> np.ndarray:
    #     """
    #     Returns ndarray (pred_horizon, action_dim) in *unnormalised* env scale.
    #     """
    #     # ---- gather history --------------------------------------------- #
    #     imgs  = np.stack([x["image"]      for x in obs_deque])  # (H,3,96,96)
    #     a_pos = np.stack([x["agent_pos"]  for x in obs_deque])  # (H,2)

    #     nimgs  = torch.from_numpy(imgs).to(self.device, dtype=torch.float32)
    #     nagpos = torch.from_numpy(self._normalize(a_pos, self.stats["agent_pos"])
    #                               ).to(self.device, dtype=torch.float32)

    #     # ---- encode obs -------------------------------------------------- #
    #     img_feat = self.policy_net["vision_encoder"](nimgs)      # (H,512)
    #     obs_feat = torch.cat([img_feat, nagpos], dim=-1)         # (H,514)
    #     obs_cond = obs_feat.unsqueeze(0).flatten(start_dim=1)    # (1, H*514)

    #     # ---- initialise noisy action trajectory ------------------------- #
    #     naction = torch.randn((1, self.pred_horizon, self.action_dim), device=self.device)

    #     # ---- denoising loop --------------------------------------------- #
    #     self.scheduler.set_timesteps(self.num_diffusion_iters)
    #     for t in self.scheduler.timesteps:
    #         eps_pred = self.policy_net["noise_pred_net"](sample=naction,
    #                                                      timestep=t,
    #                                                      global_cond=obs_cond)
    #         naction  = self.scheduler.step(model_output=eps_pred,
    #                                        timestep=t,
    #                                        sample=naction).prev_sample

    #     naction = naction[0].detach().cpu().numpy()              # (pred_horizon, action_dim)
    #     return self._unnormalize(naction, self.stats["action"])  # unnormalised

    @torch.no_grad()
    def _rollout(self, num_episodes: int) -> None:

        
        for ep in range(num_episodes):
            obs, _ = self.env.reset()
            obs_deque = collections.deque([obs] * self.obs_horizon, maxlen=self.obs_horizon)
            done, step = False, 0
                
            with tqdm(total=self.max_steps,
                       desc=f"Ep {ep+1}/{num_episodes}", leave=False) as bar:
                while not done and step < self.max_steps:

                    # ε-greedy choice
                    use_random = self._py_rng.random() < self.random_policy_percentage
    
                    if use_random:
                        action = np.stack(
                            [self._sample_random_action() for _ in range(self.action_horizon)]
                        )
                    else:
                        
        
                        # stack the last obs_horizon number of observations
                        images = np.stack([x['image'] for x in obs_deque])
                        agent_poses = np.stack([x['agent_pos'] for x in obs_deque])
                
                        # normalize observation
                        nagent_poses = normalize_data(agent_poses, stats=stats['agent_pos'])
                        # nagent_poses = agent_poses
                        # images are already normalized to [0,1]
                        nimages = images
                
                        # device transfer
                        nimages = torch.from_numpy(nimages).to(device, dtype=torch.float32)
                        # (2,3,96,96)
                        nagent_poses = torch.from_numpy(nagent_poses).to(device, dtype=torch.float32)
                        # (2,2)
                
                        # infer action
                        with torch.no_grad():
                            # get image features
                            image_features = ema_nets['vision_encoder'](nimages)
                            # (2,512)
                
                            # concat with low-dim observations
                            obs_features = torch.cat([image_features, nagent_poses], dim=-1)
                
                            # reshape observation to (B,obs_horizon*obs_dim)
                            obs_cond = obs_features.unsqueeze(0).flatten(start_dim=1)
                
                            # initialize action from Guassian noise
                            noisy_action = torch.randn(
                                (1, pred_horizon, action_dim), device=device)
                            naction = noisy_action
                
                            # init scheduler
                            noise_scheduler.set_timesteps(self.num_diffusion_iters)
                
                            for k in noise_scheduler.timesteps:
                                # predict noise
                                noise_pred = ema_nets['noise_pred_net'](
                                    sample=naction,
                                    timestep=k,
                                    global_cond=obs_cond
                                )
                
                                # inverse diffusion step (remove noise)
                                naction = noise_scheduler.step(
                                    model_output=noise_pred,
                                    timestep=k,
                                    sample=naction
                                ).prev_sample
                
                        # unnormalize action
                        naction = naction.detach().to('cpu').numpy()
                        # (B, pred_horizon, action_dim)
                        naction = naction[0]
                        # action_pred = naction
                        action_pred = unnormalize_data(naction, stats=stats['action'])
                
                        # only take action_horizon number of actions
                        start = obs_horizon - 1
                        end = start + action_horizon
                        action = action_pred[start:end,:]
                        # (action_horizon, action_dim)
            
                    # execute action_horizon number of steps
                    # without replanning
                    for i in range(len(action)):
                        # log inputs
                        self._images_t.append(obs["image"])
                        self._actions.append(action[i])
                        
                        # stepping env
                        obs, reward, done, _, info = env.step(action[i])
                        # save observations
                        obs_deque.append(obs)
                        # and reward/vis
                        # imgs.append(env.render(mode='rgb_array'))

                        # log outputs
                        self._images_tp1.append(obs["image"])
                        self._rewards.append(reward)
            
                        step += 1
                        bar.update(1)
                        bar.set_postfix(R=f"{reward:.3f}", rand=use_random)
    
                        if done or step >= self.max_steps:
                            break
                            

import math, torch, torch.nn as nn, torch.nn.functional as F

# ───────────────────────── helpers ──────────────────────────
def timestep_embedding(t, dim):
    half = dim // 2
    freqs = torch.exp(-math.log(10000) *
                      torch.arange(half, device=t.device) / (half - 1))
    ang = t[:, None].float() * freqs[None]
    return torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)

def best_group_divisor(c, gmax=8):
    for g in range(min(gmax, c), 0, -1):
        if c % g == 0:
            return g
    return 1

class ResidualBlock(nn.Module):
    def __init__(self, in_c, out_c, emb_c):
        super().__init__()
        self.n1 = nn.GroupNorm(best_group_divisor(in_c),  in_c)
        self.n2 = nn.GroupNorm(best_group_divisor(out_c), out_c)
        self.c1 = nn.Conv2d(in_c,  out_c, 3, 1, 1)
        self.c2 = nn.Conv2d(out_c, out_c, 3, 1, 1)
        self.emb = nn.Linear(emb_c, out_c * 2)
        self.skip = nn.Identity() if in_c == out_c else nn.Conv2d(in_c, out_c, 1)

    def forward(self, x, emb):
        h = self.c1(F.silu(self.n1(x)))
        scale, shift = self.emb(emb).chunk(2, dim=1)
        h = h * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        h = self.c2(F.silu(self.n2(h)))
        return h + self.skip(x)

class CrossAttentionBlock(nn.Module):
    def __init__(self, img_c, act_c, heads=4):
        super().__init__()
        self.h = heads
        self.norm_img = nn.GroupNorm(best_group_divisor(img_c), img_c)
        self.norm_act = nn.LayerNorm(act_c)
        self.q = nn.Linear(img_c, img_c)
        self.k = nn.Linear(act_c, img_c)
        self.v = nn.Linear(act_c, img_c)
        self.out = nn.Linear(img_c, img_c)

    def forward(self, img, act):
        B, C, H, W = img.shape
        q = self.q(self.norm_img(img).flatten(2).transpose(1, 2))         # B,N,C
        k = self.k(self.norm_act(act)).unsqueeze(1)                        # B,1,C
        v = self.v(self.norm_act(act)).unsqueeze(1)
        q = q.reshape(B, -1, self.h, C // self.h).transpose(1, 2)          # B,h,N,d
        k = k.reshape(B,  1, self.h, C // self.h).transpose(1, 2)
        v = v.reshape(B,  1, self.h, C // self.h).transpose(1, 2)
        attn = torch.softmax((q @ k.transpose(-2, -1)) /
                             math.sqrt(C // self.h), dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, -1, C)
        out = self.out(out).transpose(1, 2).view(B, C, H, W)
        return img + out

# ───────────────────── configurable U-Net ─────────────────────
class ActionConditionedUNet(nn.Module):
    def __init__(self,
                 img_c=3,
                 base=64,
                 act_dim=2,
                 emb_c=128,
                 heads=4,
                 num_layers=2):
        super().__init__()
        self.num_layers = num_layers

        # embeddings -----------------------------------------------------
        self.act_emb  = nn.Sequential(nn.Linear(act_dim, emb_c),
                                      nn.SiLU(),
                                      nn.Linear(emb_c, emb_c))
        self.time_mlp = nn.Sequential(nn.Linear(emb_c, emb_c*4),
                                      nn.SiLU(),
                                      nn.Linear(emb_c*4, emb_c))

        # channel sizes per level ---------------------------------------
        enc_chs = [base * (2 ** i) for i in range(num_layers)]   # [64,128,...]

        # encoder --------------------------------------------------------
        self.encoder = nn.ModuleList()
        in_c = img_c * 2                      # concat(noisy,img_t)
        for ch in enc_chs:
            self.encoder.append(ResidualBlock(in_c, ch, emb_c))
            in_c = ch
        self.pool = nn.AvgPool2d(2)

        # bottleneck -----------------------------------------------------
        mid_ch = enc_chs[-1]
        self.mid1 = ResidualBlock(mid_ch, mid_ch, emb_c)
        self.xatt = CrossAttentionBlock(mid_ch, emb_c, heads)
        self.mid2 = ResidualBlock(mid_ch, mid_ch, emb_c)

        # decoder --------------------------------------------------------
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.decoder = nn.ModuleList()
        curr_ch = mid_ch
        for skip_ch in reversed(enc_chs):      # iterate over skips
            in_c  = curr_ch + skip_ch          # upsample + skip cat
            out_c = skip_ch                    # mirror encoder
            self.decoder.append(ResidualBlock(in_c, out_c, emb_c))
            curr_ch = out_c                    # for next iteration

        # output heads ---------------------------------------------------
        self.out_eps = nn.Conv2d(curr_ch, img_c, 3, 1, 1)
        self.rew     = nn.Sequential(nn.Linear(mid_ch, 256),
                                     nn.SiLU(),
                                     nn.Linear(256, 1))

    # -------------------------------------------------------------------
    def forward(self, noisy_tp1, img_t, t, a):
        """
        noisy_tp1 : (B,3,H,W)  corrupted next-frame
        img_t     : (B,3,H,W)  current frame
        t         : (B,)       diffusion timestep
        a         : (B,act_dim) action
        """
        x   = torch.cat([noisy_tp1, img_t], dim=1)            # (B,6,H,W)
        emb = self.act_emb(a) + self.time_mlp(
              timestep_embedding(t, self.act_emb[0].out_features))

        skips = []
        h = x
        for enc in self.encoder:
            h = enc(h, emb)
            skips.append(h)
            h = self.pool(h)

        h = self.mid2(self.xatt(self.mid1(h, emb), emb), emb)
        bottleneck = h.mean([2, 3])

        for dec in self.decoder:
            h = self.up(h)
            h = torch.cat([h, skips.pop()], dim=1)            # add skip
            h = dec(h, emb)

        eps_pred = self.out_eps(h)
        rew_pred = self.rew(bottleneck).squeeze(1)
        return eps_pred, rew_pred


import itertools, math, torch, torch.nn.functional as F
from torch.utils.data import DataLoader

# STAGE 3: Write the DDPM style training loop code

# ───────────────────── 3. Diffusion-style trainer ───────────────────────
class DiffusionTrainer:
    def __init__(self, model, num_diffusion_iters_worldmodel=100, beta_noise_start=1e-4, beta_noise_end=0.02, lr=2e-4, device="cuda"):
        self.model  = model
        self.opt    = torch.optim.AdamW(model.parameters(), lr=lr)
        self.T      = num_diffusion_iters_worldmodel
        self.device = device
        self.beta_noise_start=beta_noise_start
        self.beta_noise_end=beta_noise_end
        self._init_noise_schedule()

    def _init_noise_schedule(self):
        beta  = torch.linspace(self.beta_noise_start, self.beta_noise_end, self.T, device=self.device)
        alpha = 1 - beta
        alphabar = torch.cumprod(alpha, 0)
        self.sqrt_ab     = alphabar.sqrt()            # √α̅_t
        self.sqrt_1mab   = (1 - alphabar).sqrt()      # √(1-α̅_t)

    @torch.no_grad()
    def add_noise(self, x0, t):
        ε = torch.randn_like(x0)
        x_t = self.sqrt_ab[t][:, None, None, None] * x0 + \
              self.sqrt_1mab[t][:, None, None, None] * ε
        return x_t, ε

    # ---------------------------------------------------------------
    def train_epoch(self, loader, epoch):
        self.model.train()
        tot, tot_img, tot_rew = 0, 0, 0
        N = len(loader.dataset)

        step = 0
        for img_t, act_t, img_tp1, rew_tp1 in loader:
            step+=1
            img_t   = img_t.to(self.device).float()
            img_tp1 = img_tp1.to(self.device).float()
            act_t   = act_t.to(self.device).float()
            rew_tp1 = rew_tp1.to(self.device).float()
        
            B   = img_tp1.size(0)
            t   = torch.randint(0, self.T, (B,), device=self.device)
            x_t, eps_gt = self.add_noise(img_tp1, t)
        
            eps_pred, r_pred = self.model(x_t, img_t, t, act_t)
        
            img_loss = F.mse_loss(eps_pred, eps_gt)
            rew_loss = F.mse_loss(r_pred, rew_tp1)
            loss     = img_loss + rew_loss

            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

            tot     += loss.item() * B
            tot_img += img_loss.item() * B
            tot_rew += rew_loss.item() * B

            if step % 20 == 0 or step == len(loader):
                print(f"ep {epoch:03d}  step {step:03d}/{len(loader)}  "
                      f"L {loss.item():.4f}  Img {img_loss.item():.4f}  "
                      f"Rew {rew_loss.item():.4f}")

        print(f"[epoch {epoch:03d}] mean losses  "
              f"tot {tot/N:.6f} | img {tot_img/N:.6f} | rew {tot_rew/N:.6f}")



import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train diffusion-based world model and action policy")

    # Horizons and environment
    parser.add_argument("--pred_horizon", type=int, default=16)
    parser.add_argument("--obs_horizon", type=int, default=2)
    parser.add_argument("--action_horizon", type=int, default=8)
    parser.add_argument("--max_steps_pusht_environment", type=int, default=200)
    parser.add_argument("--device", type=str, default="cuda:0")

    # U-Net config
    parser.add_argument("--unet_num_layers", type=int, default=2)
    parser.add_argument("--base", type=int, default=64)
    parser.add_argument("--heads", type=int, default=4)

    # Training hyperparams
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_diffusion_iters_worldmodel", type=int, default=100, help="Set manually or match action policy")
    parser.add_argument("--num_diffusion_iters_action_policy", type=int, default=100, help="Set manually or match world model")
    parser.add_argument("--beta_noise_start", type=float, default=1e-4)
    parser.add_argument("--beta_noise_end", type=float, default=0.02)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batches_per_episode", type=int, default=50)
    parser.add_argument("--save_every", type=int, default=100)

    args = parser.parse_args()

    # If not explicitly set, tie the iterations together
    if args.num_diffusion_iters_worldmodel is None and args.num_diffusion_iters_action_policy is not None:
        args.num_diffusion_iters_worldmodel = args.num_diffusion_iters_action_policy
    elif args.num_diffusion_iters_action_policy is None and args.num_diffusion_iters_worldmodel is not None:
        args.num_diffusion_iters_action_policy = args.num_diffusion_iters_worldmodel
    elif args.num_diffusion_iters_worldmodel is None and args.num_diffusion_iters_action_policy is None:
        raise ValueError("You must specify at least one of --num_diffusion_iters_worldmodel or --num_diffusion_iters_action_policy")





    for k, v in vars(args).items():
        exec(f"{k} = {repr(v)}")
        
    # ### CONFIGS:
    # pred_horizon = 16
    # obs_horizon = 2
    # action_horizon = 8
    # max_steps_pusht_environment = 200
    # device = "cuda:0"
    
    
    # unet_num_layers = 2
    # base = 64
    # heads = 4
    
    # batch_size = 32
    # num_diffusion_iters_worldmodel = num_diffusion_iters_action_policy
    # lr = 2e-4
    # batches_per_episode = 50          
    # SAVE_EVERY = 100


    # ResNet18 has output dim of 512
    vision_feature_dim = 512
    # agent_pos is 2 dimensional
    lowdim_obs_dim = 2
    # action is delta x and delta y
    action_dim = 2
    
    
    arg_str = ','.join([f"{k}={v}" for k, v in vars(args).items()])

    randdd = str(np.random.rand())[-5:]
    CKPT       = f"./checkpoint/model_{randdd}.ckpt"

    with open(f'./checkpoint/model_config_{randdd}.txt', 'w') as f:
        f.write(arg_str)

    print("CKPT name:",CKPT)

    env_seed=42

    stats = {'agent_pos': 
             {
              'min': np.array([13.456424, 32.938293]),
              'max': np.array([496.14618, 510.9579 ])
             }, 
             'action': 
             {
              'min': np.array([12., 25.]), 
              'max': np.array([511., 511.])
                       
             }
            }
    
    
    policy_config = dict(
        obs_horizon=obs_horizon,
        pred_horizon=pred_horizon,
        action_horizon=action_horizon,
        action_dim=action_dim,
        num_diffusion_iters=num_diffusion_iters_action_policy,
        stats=stats,                # the same dict you defined
        max_steps=max_steps_pusht_environment,
    )
    
    
    #@markdown ### **Network Demo**
    
    # construct ResNet18 encoder
    # if you have multiple camera views, use seperate encoder weights for each view.
    vision_encoder = get_resnet('resnet18')
    
    # IMPORTANT!
    # replace all BatchNorm with GroupNorm to work with EMA
    # performance will tank if you forget to do this!
    vision_encoder = replace_bn_with_gn(vision_encoder)
    
    # observation feature has 514 dims in total per step
    obs_dim = vision_feature_dim + lowdim_obs_dim
    
    
    # create network object
    noise_pred_net = ConditionalUnet1D(
        input_dim=action_dim,
        global_cond_dim=obs_dim*obs_horizon
    )
    
    # the final arch has 2 parts
    nets = nn.ModuleDict({
        'vision_encoder': vision_encoder,
        'noise_pred_net': noise_pred_net
    })
    
    
    
    # demo
    with torch.no_grad():
        # example inputs
        image = torch.zeros((1, obs_horizon,3,96,96))
        agent_pos = torch.zeros((1, obs_horizon, lowdim_obs_dim))
        # vision encoder
        image_features = nets['vision_encoder'](
            image.flatten(end_dim=1))
        # (2,512)
        image_features = image_features.reshape(*image.shape[:2],-1)
        # (1,2,512)
        obs = torch.cat([image_features, agent_pos],dim=-1)
        # (1,2,514)
    
        noised_action = torch.randn((1, pred_horizon, action_dim))
        diffusion_iter = torch.zeros((1,))
    
        # the noise prediction network
        # takes noisy action, diffusion iteration and observation as input
        # predicts the noise added to action
        noise = nets['noise_pred_net'](
            sample=noised_action,
            timestep=diffusion_iter,
            global_cond=obs.flatten(start_dim=1))
    
        # illustration of removing noise
        # the actual noise removal is performed by NoiseScheduler
        # and is dependent on the diffusion noise schedule
        denoised_action = noised_action - noise
    
    # for this demo, we use DDPMScheduler with 100 diffusion iterations
    num_diffusion_iters_action_policy = 100
    
    
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=num_diffusion_iters_action_policy,
        # the choise of beta schedule has big impact on performance
        # we found squared cosine works the best
        beta_schedule='squaredcos_cap_v2',
        # clip output to [-1,1] to improve stability
        clip_sample=True,
        # our network predicts noise (instead of denoised action)
        prediction_type='epsilon'
    )
    
    # device transfer
    _ = nets.to(device)
    
    
    load_pretrained = True
    if load_pretrained:
      ckpt_path = "pusht_vision_100ep.ckpt"
      if not os.path.isfile(ckpt_path):
          id = "1XKpfNSlwYMGaF5CncoFaLKCDTWoLAHf1&confirm=t"
          gdown.download(id=id, output=ckpt_path, quiet=False)
    
      state_dict = torch.load(ckpt_path, map_location='cuda')
      ema_nets = nets
      ema_nets.load_state_dict(state_dict)
      print('Pretrained weights loaded.')
    else:
      print("Skipped pretrained weight loading.")
    
    
    
    # 0. create env object
    env = PushTImageEnv()
    
    # 1. seed env for initial state.
    # Seed 0-200 are used for the demonstration dataset.
    env.seed(env_seed)
    
    # 2. must reset before use
    obs, info = env.reset()
    
    # Stage 3: Now lets test the train loop to overfit on one episode. SHOULD HIT 0 LOSS! 
    
    
    
    
    os.makedirs(os.path.dirname(CKPT), exist_ok=True)
    
    
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = ActionConditionedUNet(img_c=3, 
                                   base=base, 
                                   act_dim=action_dim, 
                                   num_layers=unet_num_layers, 
                                   heads = heads).to(device)
    
    trainer = DiffusionTrainer(model, 
                               num_diffusion_iters_worldmodel=num_diffusion_iters_worldmodel, 
                               beta_noise_start=beta_noise_start, 
                               beta_noise_end=beta_noise_end, 
                               lr=lr, 
                               device=device)
    print(policy_config)
    counterr=0
    while True:
        counterr+=1
        dataset = SingleStatePushTEpisodicDataset(
            env_instance=env,
            policy_net=ema_nets,          # {'vision_encoder': ..., 'noise_pred_net': ...}
            policy_scheduler=noise_scheduler,
            policy_config=policy_config,
            num_episodes=1,
            random_policy_percentage=0.0, #set to 100% if you want a really bad policy.. and in between for in between.
            data_collection_device=device,
        )
        for ep in range(1, batches_per_episode + 1):
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
            trainer.train_epoch(loader, ep)
        print("="*50)
        print(counterr)
        print(arg_str)
        print("="*50)    
        if counterr % save_every == 0 or counterr==1:
            torch.save({"model_state": model.state_dict()},
                       CKPT)
            
            print(f"[counterr {counterr}] checkpoint saved → {CKPT}")
