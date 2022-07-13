import glob
import os
import sys
import csv

import random
from tkinter import W
import numpy as np
import math
import time

import matplotlib.pyplot as plt
import matplotlib.image as mpllimg
import cv2

import torch

from env_carla import Environment

IM_WIDTH = 480
IM_HEIGHT = 360
CAM_HEIGHT = 20
ROTATION = -90
ZOOM = 110
ROOT_STORAGE_PATH = "/disk/vanishing_data/is789/anomaly_samples/"
MAP_SET = ["Town01_Opt", "Town02_Opt", "Town03_Opt", "Town04_Opt","Town05_Opt"]
# MAP_SET = ["Town01_Opt"]

class Sampler:

    def __init__(self, s_width=IM_WIDTH, s_height=IM_HEIGHT, cam_height=CAM_HEIGHT, cam_rotation=ROTATION, cam_zoom=ZOOM, host="localhost"):
        self.s_width = s_width
        self.s_height = s_height
        self.cam_height = cam_height
        self.cam_rotation = cam_rotation
        self.cam_zoom = cam_zoom
        self.host = host

    
    def sample(self, world_model=None, random_spawn=True):
        if world_model == None: world_model = MAP_SET[random.randrange(0,len(MAP_SET))]

        env = Environment(world=world_model, s_width=self.s_width, s_height=self.s_height, cam_height=self.cam_height, cam_rotation=self.cam_rotation, cam_zoom=self.cam_zoom, host=self.host, random_spawn=random_spawn)
        env.init_ego()
        image = env.reset()
        env.deleteEnv()
        return self.arrange_colorchannels(image)

    def sample_Ride(self, world_model=None, random_spawn=True, num_of_snaps=100, tick_rate=1):
        if world_model == None: world_model = MAP_SET[random.randrange(0,len(MAP_SET))]

        env = Environment(world=world_model, s_width=self.s_width, s_height=self.s_height, cam_height=self.cam_height, cam_rotation=self.cam_rotation, cam_zoom=self.cam_zoom, host=self.host, random_spawn=random_spawn)
        env.init_ego()
        env.reset()
        images = []
        for x in range(num_of_snaps):
            image = env.get_observation()
            image = self.arrange_colorchannels(image)
            images.append(image)
            time.sleep(tick_rate)
        env.deleteEnv()
        return images

    def show_Example(self, random_spawn=True):
        image = self.sample(random_spawn=random_spawn)
        plt.imshow(image)
        plt.show()


    def collect_Samples(self, sample_size, tick_rate=1):
        print(f"Starting to collect {sample_size} frames out of {len(MAP_SET)} worlds...")
        storagePath = self.create_Storage()
        samplesPerEnv = int(sample_size / len(MAP_SET))
        
        image_index = 0
        for x in range(len(MAP_SET)):
            images = self.sample_Ride(world_model=MAP_SET[x], random_spawn=True, num_of_snaps=samplesPerEnv, tick_rate=tick_rate)
            images = np.array(images)
            images = (images * 255).astype("int")
            print(f"finished world! {x}")
            for k in range(len(images)):
                cv2.imwrite(storagePath + f"snap_{image_index}.png", images[k]) 
                # plt.imsave(storagePath + f"snap_{image_index}.png",images[k], format="png")
                image_index = image_index + 1

        print(f"Finished | Collected: {str(samplesPerEnv * len(MAP_SET))} samples.")


# ==============================================================================
# -- Collect huge amounts of samples (> 20k) --> save after each frame ---------
# ==============================================================================
    def collect_huge_Samples(self, sample_size, tick_rate=1):
        print(f"Starting to collect {sample_size} frames out of {len(MAP_SET)} worlds...")
        storagePath = self.create_Storage()
        samplesPerEnv = int(sample_size / len(MAP_SET))
        
        image_index = 0
        for x in range(len(MAP_SET)):
            image_index = self.sample_save_Ride(world_model=MAP_SET[x], random_spawn=True, num_of_snaps=samplesPerEnv, tick_rate=tick_rate, save_index=image_index, storagePath=storagePath)
            print(f"finished world! {x}")
        
        print(f"Finished | Collected: {str(samplesPerEnv * len(MAP_SET))} samples.")

    def sample_save_Ride(self, save_index, storagePath, world_model=None, random_spawn=True, num_of_snaps=100, tick_rate=1):
        if world_model == None: world_model = MAP_SET[random.randrange(0,len(MAP_SET))]

        env = Environment(world=world_model, s_width=self.s_width, s_height=self.s_height, cam_height=self.cam_height, cam_rotation=self.cam_rotation, cam_zoom=self.cam_zoom, host=self.host, random_spawn=random_spawn)
        env.init_ego()
        env.reset()
        for x in range(num_of_snaps):
            image = env.get_observation()
            image = self.arrange_colorchannels(image)
            image = (image * 255).astype("int")
            cv2.imwrite(storagePath + f"snap_{save_index}.png", image) 
            save_index += 1
            time.sleep(tick_rate)
        env.deleteEnv()
        return save_index
# ==============================================================================
# -- End of huge sample code ---------------------------------------------------
# ==============================================================================

    # changes order of color channels. Silly but works...
    def arrange_colorchannels(self, image):
        mock = image.transpose(2,1,0)
        tmp = []
        tmp.append(mock[2])
        tmp.append(mock[1])
        tmp.append(mock[0])
        tmp = np.array(tmp)
        tmp = tmp.transpose(2,1,0)
        return tmp

    # create Storage and return the path pointing towards it
    def create_Storage(self):
        if not os.path.isdir(ROOT_STORAGE_PATH):
            os.mkdir(ROOT_STORAGE_PATH)

        timestr = time.strftime("%Y-%m-%d_%H:%M:%S")
        pathToStorage = ROOT_STORAGE_PATH + "Samples_" + timestr + "/"

        if not os.path.isdir(pathToStorage):
            os.mkdir(pathToStorage)
        
        return pathToStorage

# ==============================================================================
# -- Video section -------------------------------------------------------------
# ==============================================================================
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 
    def add_model_prediction(self, model, device, true_image):
        true_image = np.transpose(true_image, (2,1,0))
        img = np.array([true_image])
        img = torch.as_tensor(img)
        img = img.to(device)
        out = model(img)
        out = out[0].detach().cpu().numpy()
        
        seperator = np.zeros((3,20,512))
        
        final_img = np.concatenate((true_image, seperator, out), axis=1)
        final_img = np.transpose(final_img, (2,1,0))
        return final_img

    # 10 fps rendering
    def sample_canonicaly(self, model, device, seconds):
        
        images = self.sample_Ride(world_model="Town01_Opt", num_of_snaps=seconds*10, tick_rate=0.1)
        storagePath = "/disk/vanishing_data/is789/anomaly_samples/video_images/"
        path_list = Sampler.get_image_paths(storagePath)
        for path in path_list: #remove former runs
            os.remove(path)
        if not os.path.isdir(storagePath):
            os.mkdir(storagePath)
            

        tmp = images
        images = []
        for image in tmp:
            model_predict = self.add_model_prediction(model, device, image)
            images.append(model_predict)
            
        image_index = 0
        images = np.array(images)
        images = (images * 255).astype("int")
        for k in range(len(images)):
            fill_index = image_index
            if image_index < 10:
                fill_index = "00"+str(image_index)
            elif image_index < 100:
                fill_index = "0"+str(image_index)
            cv2.imwrite(storagePath + f"snap_{fill_index}.png", images[k])
            # plt.imsave(storagePath + f"snap_{image_index}.png",images[k], format="png")
            image_index = image_index + 1
        
        return storagePath
    
    def create_model_video(self, model, device, seconds=14):
        storagePath = self.sample_canonicaly(model, device, seconds)
        path_list = sorted(Sampler.get_image_paths(storagePath))
        video = cv2.VideoWriter("example_ride.avi", 0, 10, (1044,512))
        for path in path_list:
            video.write(cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB))
        cv2.destroyAllWindows()
        return video.release()



# ==============================================================================
# -- Static methods ------------------------------------------------------------
# ==============================================================================


    # loads the set of images
    @staticmethod
    def load_Images(dir_path, size=9999999999):
        if size <= 0: size = 9999999999
        path_list = Sampler.get_image_paths(dir_path)
        img_list = []

        for x in range(len(path_list)):
            if x >= size: break
            path = path_list[x]
            img = cv2.imread(path)
            img_list.append(img)
        
        img_list = np.array(img_list)
        img_list = img_list[:,:,:,:3] # clear alpha channel
        print(f"Loaded {str(len(img_list))} images | width = {len(img_list[0])}, height = {len(img_list[0][0])}, channels = {len(img_list[0][0][0])}")
        return img_list


    # loads the set of images
    @staticmethod
    def sample_from_Set(img_list):
        size = len(img_list)
        index = random.randrange(0,size-1)


        plt.imshow(img_list[index])

        print(f"width = {len(img_list[index])}, height = {len(img_list[index][0])}, channels = {len(img_list[index][0][0])}")


    @staticmethod
    def get_image_paths(path):
        path_list = []

        for root, dirs, files in os.walk(os.path.abspath(path)):
            for file in files:
                path_list.append(os.path.join(root, file))
        return path_list



if __name__ == "__main__":
    sampler = Sampler(s_width=512, s_height=512, cam_height=4, cam_zoom=50, cam_rotation=-18)
    # sampler.collect_Samples(sample_size=10, tick_rate=5)
    sampler.collect_huge_Samples(sample_size=11000, tick_rate=5)