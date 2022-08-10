"""Script for building a hovering robot and collecting heightmap data."""

import argparse
import glob
import io
import os

import cv2
import gym
import imageio
import matplotlib.pyplot as plt
import numpy as np
from blind_walking.envs.env_modifiers.env_modifier import EnvModifier
from blind_walking.envs.env_modifiers.heightfield import HeightField
from blind_walking.envs.env_modifiers.stairs import Stairs, boxHalfLength, boxHalfWidth
from blind_walking.envs.env_wrappers import observation_dictionary_to_array_wrapper as obs_array_wrapper
from blind_walking.envs.sensors import environment_sensors
from blind_walking.envs.tasks.forward_task import ForwardTask
from enjoy import Logger
from gym.wrappers import Monitor
from joblib import Parallel, delayed
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scripts.plot_stats import Plotter, alphanum_key, stitch_videos

import utils.import_envs  # noqa: F401 pytype: disable=import-error


class MultipleTerrain(EnvModifier):
    def __init__(self):
        super().__init__()
        # Stairs parameters
        self.step_rise_levels = [0.02, 0.05, 0.07, 0.09, 0.11]
        self.num_steps = 10
        self.stair_gap = 1.0
        self.step_run = 0.3
        self.stair_length = (self.num_steps - 1) * self.step_run * 2 + boxHalfLength * 2 * 2
        self.sec_num_steps = 1
        self.sec_stair_length = (self.sec_num_steps - 1) * self.step_run * 2 + boxHalfLength * 2 * 2
        self.sec_trip_half_run = 0.03
        self.sec_trip_num_steps = 1
        self.sec_trip_stair_length = (
            self.sec_trip_num_steps - 1
        ) * self.sec_trip_half_run * 2 * 2 * 2 + self.sec_trip_half_run * 2 * 2
        # Heightfield parameters
        self.hf_length = 18

        self.hf = HeightField()
        self.stairs = []
        self.sec_stairs = []
        self.sec_trip_stairs = []
        for _ in range(len(self.step_rise_levels)):
            self.stairs.append(Stairs())
            self.sec_stairs.append(Stairs())
            self.sec_trip_stairs.append(Stairs())

        # Random boxes
        self.boxes_rise_levels = [0.03, 0.05, 0.07, 0.09, 0.10]
        self.boxes_pos_y = [
            -0.08,
            -0.14,
            0.04,
            0.14,
            0.08,
            0.10,
            0.06,
            0.0,
            -0.06,
            -0.10,
            -0.04,
            -0.08,
            0.08,
            0.0,
            0.04,
            0.12,
            0.06,
            -0.18,
            -0.06,
            -0.12,
        ]
        self.boxes_half_width = [
            0.03,
            0.04,
            0.05,
            0.06,
            0.07,
            0.04,
            0.05,
            0.06,
            0.07,
            0.03,
            0.05,
            0.06,
            0.07,
            0.03,
            0.04,
            0.06,
            0.07,
            0.03,
            0.04,
            0.05,
        ]
        self.boxes_half_length = [
            0.025,
            0.02,
            0.035,
            0.03,
            0.015,
            0.015,
            0.015,
            0.02,
            0.035,
            0.03,
            0.03,
            0.015,
            0.025,
            0.02,
            0.035,
            0.035,
            0.03,
            0.015,
            0.025,
            0.02,
        ]
        self.box_gap = 0.5
        self.boxes = [Stairs() for _ in range(30)]

    def _generate(self, env):
        # Generate stairs
        start_x = self.stair_gap
        for i in range(len(self.step_rise_levels)):
            self.sec_trip_stairs[i]._generate(
                env,
                start_x=start_x,
                num_steps=self.sec_trip_num_steps,
                step_rise=self.step_rise_levels[i],
                boxHalfLength=self.sec_trip_half_run,
            )
            start_x += self.sec_trip_stair_length + self.stair_gap
            self.sec_stairs[i]._generate(
                env, start_x=start_x, num_steps=self.sec_num_steps, step_rise=self.step_rise_levels[i], step_run=self.step_run
            )
            start_x += self.sec_stair_length + self.stair_gap
            self.stairs[i]._generate(
                env, start_x=start_x, num_steps=self.num_steps, step_rise=self.step_rise_levels[i], step_run=self.step_run
            )
            start_x += self.stair_length + self.stair_gap

        # Generate boxes
        for i in range(len(self.boxes)):
            self.boxes[i]._generate(
                env,
                start_x=start_x,
                num_steps=1,
                step_rise=self.boxes_rise_levels[i % len(self.boxes_rise_levels)],
                boxHalfLength=self.boxes_half_length[i % len(self.boxes_half_length)],
                boxHalfWidth=self.boxes_half_width[i % len(self.boxes_half_width)],
                pos_y=self.boxes_pos_y[i % len(self.boxes_pos_y)],
            )
            start_x += self.box_gap

        # Generate heightfield
        start_x += self.hf_length / 2
        self.hf._generate(env, start_x=start_x, heightPerturbationRange=0.08)

    def get_z_position(self, env, x, y):
        """Get z position for hovering robot at the xy-coord."""
        info = env.pybullet_client.rayTest([x, y, 2.0], [x, y, 0.0])
        hit_position = info[0][3]
        z_coordinate = hit_position[2]
        return z_coordinate


def get_video_save_path(env: Monitor):
    return os.path.join(
        env.directory,
        "{}.video.{}.video{:06}.mp4".format(env.file_prefix, env.file_infix, env.episode_id),
    )


def main():  # noqa: C901
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--no-hover", action="store_true", default=False, help="Do not generate heightmap data with hover robot"
    )
    parser.add_argument("--no-plot", action="store_true", default=False, help="Do not generate heightmap plots")
    parser.add_argument("--record", action="store_true", default=False, help="Record video")
    parser.add_argument("-n", "--n-timesteps", help="number of timesteps", default=1000, type=int)
    args = parser.parse_args()

    # Data parameters
    dx = 0.05
    dy = 0
    grid_sizes = [(12, 16)]
    grid_units = [(0.03, 0.03)]
    grid_transforms = [(0.10, 0)]
    ray_origins = ["head"]
    grid_names = ["depth"]
    num_timesteps = args.n_timesteps

    if not args.no_hover:
        # Environment parameters
        robot_sensor_list = []
        env_sensor_list = [
            environment_sensors.LocalTerrainDepthSensor(
                grid_size=grid_sizes[i],
                grid_unit=grid_units[i],
                transform=grid_transforms[i],
                ray_origin=ray_origins[i],
                noisy_reading=False,
                name=grid_names[i],
            )
            for i in range(len(grid_sizes))
        ]
        env_randomizer_list = []
        env_modifier_list = [MultipleTerrain()]
        obs_wrapper = obs_array_wrapper.ObservationDictionaryToArrayWrapper
        task = ForwardTask()

        # Create environment
        env = gym.make(
            "A1GymEnv-v0",
            robot_sensor_list=robot_sensor_list,
            env_sensor_list=env_sensor_list,
            env_randomizer_list=env_randomizer_list,
            env_modifier_list=env_modifier_list,
            obs_wrapper=obs_wrapper,
            task=task,
        )
        if args.record:
            video_folder = os.path.dirname(__file__)
            env = Monitor(env, video_folder, force=True)
            replay_video_path = get_video_save_path(env)
        env.reset()

        # Create logger
        hm_logger = Logger("heightmap")

        # Move robot across terrain and collect heightmap data
        default_orientation = env.pybullet_client.getQuaternionFromEuler([0, 0, 0])
        default_position = env.robot._GetDefaultInitPosition()
        zero_action = np.zeros(12)
        position = default_position.copy()
        for i in range(num_timesteps):
            # Update position
            position[0] += dx
            position[1] += dy
            # Calculate z pos 5 timestep faster to avoid legs hitting the stairs
            z_pos = env_modifier_list[0].get_z_position(env, position[0] + 5 * dx, position[1] + 5 * dy)
            position[2] = default_position[2] + z_pos
            env.pybullet_client.resetBasePositionAndOrientation(env.robot.quadruped, position, default_orientation)
            obs, _, _, _ = env.step(zero_action)
            # Record heightmap data
            hm_logger.update([obs])
            if i % 100 == 0:
                print(f"timesteps {i}")

        # Close environment and save data
        env.close()
        hm_logger.save(os.path.dirname(__file__))
        print("Collected data")

    if not args.no_plot:
        datalim = (0.2, 1.0)
        dirpath = os.path.dirname(__file__)
        datapath = os.path.join(dirpath, "heightmap.npy")
        data = np.load(datapath)

        # Plot one data point of the heightmap
        plotter = Plotter(datapath, "hm_single")
        plotter.plot(columns=[-1], ylim=datalim, savedir=dirpath)

        # Generate GIF of heightmap over time
        hmobs_startindex = 0
        grid_end_indices = [np.prod(s) for s in grid_sizes]
        grid_end_indices = np.cumsum(grid_end_indices) + hmobs_startindex
        subplot_size = "2" + str(int(np.ceil(len(grid_sizes) / 2)))

        def get_pic_of_timestep(t):
            fig = plt.figure()
            for i in range(len(grid_sizes)):
                start_index = hmobs_startindex if i == 0 else grid_end_indices[i - 1]
                data = plotter.data[t][start_index : grid_end_indices[i]]
                if any(np.array(grid_sizes[i]) == 1):
                    # 2d bar plot
                    ax = fig.add_subplot(int(subplot_size + str(i + 1)))
                    ax.bar(x=np.arange(len(data)), height=data)
                    ax.set_xticklabels([])
                    ax.set_ylim(datalim)
                    ax.yaxis.tick_right()
                    ax.set_title(grid_names[i])
                else:
                    # 3d bar plot
                    ax = fig.add_subplot(int(subplot_size + str(i + 1)), projection="3d")
                    x = np.arange(grid_sizes[i][0])
                    y = np.arange(grid_sizes[i][1])
                    xx, yy = np.meshgrid(x, y)
                    x, y = xx.ravel(), yy.ravel()
                    z = np.zeros(len(x))
                    dx = dy = 1
                    dz = data
                    ax.set_zlim(datalim)
                    ax.set_xticklabels([])
                    ax.set_yticklabels([])
                    ax.bar3d(x, y, z, dx, dy, dz, shade=True)
                    ax.set_title(grid_names[i])
            plt.savefig(os.path.join(dirpath, f"tmp{t}"))
            plt.close()

        Parallel(n_jobs=2)(delayed(get_pic_of_timestep)(t) for t in range(num_timesteps))
        # for t in range(num_timesteps): get_pic_of_timestep(t)  # No parallelism
        print("Generated images for video")
        # build gif
        files = glob.glob(os.path.join(dirpath, "tmp*.png"))
        files.sort(key=alphanum_key)
        heightmap_video_path = os.path.join(dirpath, "hm.mp4")
        with imageio.get_writer(heightmap_video_path, mode="I", fps=30) as writer:
            for f in files:
                image = imageio.imread(f)
                writer.append_data(image)
        print("Created heightmap video")
        # remove images
        for f in files:
            os.remove(f)
        print("Removed unnessary image files")

        # stitch both videos together
        if args.record:
            stitch_videos(
                in_path1=replay_video_path,
                in_path2=heightmap_video_path,
                out_path=os.path.join(dirpath, "replay_and_hm.mp4"),
                verbose=0,
            )


if __name__ == "__main__":
    main()
