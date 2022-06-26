import glob
import os
import sys

import random
import numpy as np
import math
import time

import matplotlib.pyplot as plt
import cv2
import weakref

import torch

IM_WIDTH = 480
IM_HEIGHT = 360

BEV_DISTANCE = 20

N_ACTIONS = 9

RESET_SLEEP_TIME = 1

# ==============================================================================
# -- Find CARLA module ---------------------------------------------------------
# ==============================================================================
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

# ==============================================================================
# -- Add PythonAPI for release mode --------------------------------------------
# ==============================================================================
try:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/carla')
except IndexError:
    pass

import carla

class Environment:

    def __init__(self, world, host='localhost', port=2000, s_width=IM_WIDTH, s_height=IM_HEIGHT, cam_height=BEV_DISTANCE, cam_rotation=-90, cam_zoom=110, random_spawn=True):
        self.client = carla.Client(host, port)            #Connect to server
        self.client.set_timeout(30.0)

        # traffic_manager = self.client.get_trafficmanager(port)
        # traffic_manager.set_global_distance_to_leading_vehicle(2.5)
        # traffic_manager.set_respawn_dormant_vehicles(True)
        # traffic_manager.set_synchronous_mode(True)

        self.random_spawn = random_spawn
        if not world == None: self.world = self.client.load_world(world)
        else: self.world = self.client.load_world("Town01_Opt")

        self.bp_lib = self.world.get_blueprint_library()
        self.map = self.world.get_map()
        self.spawn_points = self.map.get_spawn_points()
        self.s_width = s_width
        self.s_height = s_height
        self.cam_height = cam_height
        self.cam_rotation = cam_rotation
        self.cam_zoom = cam_zoom

        self.actor_list = []
        self.IM_WIDTH = IM_WIDTH
        self.IM_HEIGHT = IM_HEIGHT
        weather = carla.WeatherParameters(
            cloudiness=0.0,
            precipitation=0.0,
            precipitation_deposits= 0.0,
            wind_intensity=0.0,
            fog_density=0.0,
            wetness=0.0,
            sun_altitude_angle=70.0)

        self.world.set_weather(weather) 

        #Enable synchronous mode between server and client
        # self.settings = self.world.get_settings()
        # self.settings.synchronous_mode = True # Enables synchronous mode
        # self.world.apply_settings(self.settings)

    def init_ego(self):
        self.vehicle_bp = self.bp_lib.find('vehicle.tesla.model3')
        self.ss_camera_bp = self.bp_lib.find('sensor.camera.rgb')
        self.col_sensor_bp = self.bp_lib.find('sensor.other.collision')

        # Configure sensors
        self.ss_camera_bp.set_attribute('image_size_x', f'{self.s_width}')
        self.ss_camera_bp.set_attribute('image_size_y', f'{self.s_height}')
        self.ss_camera_bp.set_attribute('fov', str(self.cam_zoom))

        self.ss_cam_location = carla.Location(0,0,self.cam_height)
        self.ss_cam_rotation = carla.Rotation(self.cam_rotation,0,0)
        self.ss_cam_transform = carla.Transform(self.ss_cam_location, self.ss_cam_rotation)


    def reset(self):
        # for actor in self.actor_list:
        #     actor.destroy()

        self.actor_list = []

        # Spawn vehicle
        if self.random_spawn: transform = random.choice(self.spawn_points)
        else: transform = self.spawn_points[0]
        self.vehicle = self.world.spawn_actor(self.vehicle_bp, transform)
        self.vehicle.set_autopilot(True)
        self.actor_list.append(self.vehicle)

        # Attach and listen to image sensor (RGB)
        self.ss_cam = self.world.spawn_actor(self.ss_camera_bp, self.ss_cam_transform, attach_to=self.vehicle, attachment_type=carla.AttachmentType.Rigid)
        weak_self = weakref.ref(self)
        self.actor_list.append(self.ss_cam)
        self.ss_cam.listen(lambda data: self.__process_sensor_data(data, weak_self))

        time.sleep(RESET_SLEEP_TIME)   # sleep to get things started and to not detect a collision when the car spawns/falls from sky.


        self.episode_start = time.time()
        return self.get_observation()

    def step(self, action):
        # Easy actions: Steer left, center, right (0, 1, 2)
        if action == 0:
            self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=0))
        elif action == 1:
            self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=-1))
        elif action == 2:
            self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=1))
        elif action == 3:
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.5, steer=0))
        elif action == 4:
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.5, steer=-1))
        elif action == 5:
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.5, steer=1))
        elif action == 6:
            self.vehicle.apply_control(carla.VehicleControl(throttle=0, steer=0))
        elif action == 7:
            self.vehicle.apply_control(carla.VehicleControl(throttle=0, steer=-1))
        elif action == 8:
            self.vehicle.apply_control(carla.VehicleControl(throttle=0, steer=1))


        return self.get_observation()

    def get_observation(self):
        """ Observations in PyTorch format BCHW """
        frame = self.observation
        frame = frame.astype(np.float32) / 255
        # image = torch.from_numpy(image)
        return frame

    def __process_sensor_data(self, image, weak_self):
        """ Observations directly viewable with OpenCV in CHW format """
        # image.convert(carla.ColorConverter.CityScapesPalette)
        self = weak_self()
        i = np.array(image.raw_data)
        i2 = i.reshape((self.s_height, self.s_width, 4))
        i3 = i2[:, :, :3]
        self.observation = i3

    def deleteEnv(self):
        self.client.apply_batch([carla.command.DestroyActor(x) for x in self.actor_list])
        pass
