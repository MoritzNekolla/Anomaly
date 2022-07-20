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
from matplotlib.pyplot import figure
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

    def add_model_prediction(self, model, device, true_image):
        true_image = np.transpose(true_image, (2,1,0))
        img = np.array([true_image])
        img = torch.as_tensor(img)
        img = img.to(device)
        out = model(img)
        out = out[0].detach().cpu().numpy()
        
        seperator = np.zeros((3,15,512))
     
        prediction_img = np.concatenate((true_image, seperator, out), axis=1)
        prediction_img = np.transpose(prediction_img, (2,1,0))

        return prediction_img, np.transpose(out, (2,1,0))

    def add_errormap(self, input, output):
        #1 grayscale
        g_input = np.dot(input[...,:3], [0.2989, 0.5870, 0.1140])
        g_output = np.dot(output[...,:3], [0.2989, 0.5870, 0.1140])
        errormap = abs(g_input-g_output)
        errorMatrix = abs(input-output)
        
        #2 heatmap
        hm = np.zeros((512,512,3))
        hm = hm.astype("float32")
        for x in range(len(hm)):
            for y in range(len(hm[x])):
                if errormap[x][y] < 0.5:
                    hm[x][y][2] = 1
                    hm[x][y][1] = 2*errormap[x][y]
                    hm[x][y][0] = 2*errormap[x][y]
                else:
                    hm[x][y][0] = 1
                    hm[x][y][1] = 1 - 2*(errormap[x][y] - 0.5)
                    hm[x][y][2] = 1 - 2*(errormap[x][y] - 0.5)

        #3 colorbar
        bar = np.zeros((450,25,3))
        ll = np.linspace(0,1,450)
        for y in range(len(bar)):
            if ll[y] < 0.5:
                bar[y,:,2] = 1
                bar[y,:,1] = 2*ll[y]
                bar[y,:,0] = 2*ll[y]
            else:
                bar[y,:,2] = 1 - 2*(ll[y] - 0.5)
                bar[y,:,1] = 1 - 2*(ll[y] - 0.5)
                bar[y,:,0] = 1

        padd = np.zeros((31,25,3)) + 1.0
        bar = np.vstack((padd, bar, padd))

                

        #4 vstack
        padding = np.zeros((512,15,3)).astype("float32")
        # padding = padding + 1.0
        padding1 = np.zeros((512,30,3)).astype("float32")
        padding1 = padding1 + 1.0
        padding2 = np.zeros((512,31,3)).astype("float32")
        padding2 = padding2 + 1.0
        heatmap = np.hstack((hm,padding, padding1, bar,padding2))

        return heatmap

    def add_errorPlot(self, input, output, errorScores, samples):
        errorMatrix = abs(input-output)
        errorAvg = np.sum(errorMatrix) / (errorMatrix.shape[0] * errorMatrix.shape[1] * errorMatrix.shape[2])
        errorAvg = int(errorAvg * 100000) / 100000.0
        errorScores.append(errorAvg)

        figure(figsize=(4.27, 5.12), dpi=100)
        plt.xlim(-5, 140)
        plt.ylim(-0.01, 0.3)
        plt.plot(errorScores, color="black", lw=3)
        # get image as np.array
        canvas = plt.gca().figure.canvas
        canvas.draw()
        data = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
        errorPlot = data.reshape(canvas.get_width_height()[::-1] + (3,))

        
        plt.close()
        return errorPlot, errorScores

    # 10 fps rendering
    def sample_canonicaly(self, model, device, seconds):
        seconds = seconds * 10
        images = self.sample_Ride(world_model="Town01_Opt", num_of_snaps=seconds, tick_rate=0.1)
        storagePath = "/disk/vanishing_data/is789/anomaly_samples/video_images/"
        path_list = Sampler.get_image_paths(storagePath)
        for path in path_list: #remove former runs
            os.remove(path)
        if not os.path.isdir(storagePath):
            os.mkdir(storagePath)
            

        tmp = images
        images = []
        errorScores = [0]
        for image in tmp:
            model_predict, output = self.add_model_prediction(model, device, image)
            heatmap = self.add_errormap(image, output)
            errorPlot, errorScores = self.add_errorPlot(image, output, errorScores, seconds)
            seperator_v = np.zeros((15,1039,3))
            secondLine = np.hstack((heatmap, errorPlot))

            final_img = np.vstack((model_predict, seperator_v, secondLine))
            images.append(final_img)
            
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
        video = cv2.VideoWriter("example_ride.avi", 0, 10, (1039,1039))
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