import cv2
import numpy as np
import os, sys
import random
import math
import re
import time
import matplotlib.patches as patches
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import copy

class MLmodel:    
    
    def __init__(self):
        """
            test everything
        """
        print("Creating ML model")

        # Root directory of the project
        ROOT_DIR = os.path.abspath("..\\")
        print("\n"+os.path.abspath(".\\")+"\n")
        # Import Mask RCNN
        sys.path.append(ROOT_DIR)  # To find local version of the library
        from mrcnn import utils
        import mrcnn.model as modellib
        import blood
        #from mrcnn import visualize
        # Import COCO config
        sys.path.append(os.path.join(ROOT_DIR, "Mask_RCNN-master\\samples\\coco\\"))  # To find local version    
        #import coco

        #ROOT_DIR = os.getcwd()
        #MODEL_DIR = os.path.join(ROOT_DIR, "logs")
        #sys.path.append(os.path.join(ROOT_DIR,"samples/coco"))
        #import coco
        #COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
        #if not os.path.exists(COCO_MODEL_PATH):
        #    utils.download_trained_weights(COCO_MODEL_PATH)

        # Directory to save logs and trained model
        MODEL_DIR = os.path.join(ROOT_DIR, ".\\Mask_RCNN-master\\logs")

        # Local path to trained weights file
        COCO_MODEL_PATH = os.path.join(ROOT_DIR, ".\\Mask_RCNN-master\\logs\\mask_rcnn_blood and nonviolence_0010.h5")#change
        # Download COCO trained weights from Releases if needed
        if not os.path.exists(COCO_MODEL_PATH):
            utils.download_trained_weights(COCO_MODEL_PATH)

        config = blood.BloodConfig()

        class InferenceConfig(config.__class__):
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1

        config = InferenceConfig()
        #config.display()

        dataset = blood.BloodDataset()
        # Must call before using the dataset
        dataset.prepare()

        self.model = modellib.MaskRCNN(
            mode="inference", model_dir=MODEL_DIR, config=config
        )
        self.model.load_weights(COCO_MODEL_PATH, by_name=True)
        self.class_names = [
            'BG', 'blood', 'non'
        ]

        print("Creted Model")       

        #fps=5
        #vidcap = cv2.VideoCapture('E:/blender/New Folder/Final/Final/Teacher_Online.mp4');
        #E:/blender/New Folder/Final/Final/Teacher_Online.mp4
        #width  = vidcap.get(3)  # float
        #height = vidcap.get(4) # float
        #size=(width, height)
        #print(size)
        #out = cv2.VideoWriter('project_PR100.avi',cv2.VideoWriter_fourcc(*'DIVX'), fps, (int(vidcap.get(3)), int(vidcap.get(4))));
        
    def getFrame(self, image): 
        #vidcap.set(cv2.CAP_PROP_POS_MSEC,sec*1000) 
        #hasFrames,image = vidcap.read()
        #image1=image  
        print(type(image))    
        if os.path.exists(".\\media\\images\\"+image): 
            #cv2.imwrite('/usr/img/kang'+"{0:0=6d}".format(i)+'.jpg',image)      
            # save frame as JPG file  
            image_=cv2.imread(".\\media\\images\\"+image)
            print(image_.dtype)
            results = self.model.detect([image_], verbose=0)        
            r = results[0]  
            image_ = self.display_instances(
            image_, r['rois'], r['masks'], r['class_ids'], self.class_names, r['scores'])      
            #out.write(image) 
            #cv2.imshow('frame', image_)
            cv2.imwrite(".\\media\\WithMask\\"+image, image_)
            print("\n Image Save T WithMask \n")

            return image_
        return

        #sec = 0 
        #i = 0
        #frameRate = (1/fps) #it will capture image in each 0.5 second 
        #success = getFrame(image)
        #while success: 
        #    sec = sec + frameRate 
        #    sec = round(sec, 2)
        #    i=i+1 
        #    success = getFrame(sec,i)    

        #print(i)
        #vidcap.release()
        #out.release()
        #cv2.destroyAllWindows()

    def getFrame_vid(self, video):
        fps=0.25
        print("Video")
        if os.path.exists(".\\media\\images\\"+video): 
            print("Path Exist")
            vidcap = cv2.VideoCapture(".\\media\\images\\"+video);
            width = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            length = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
            #size=(width, height)
            print("size")
            out = cv2.VideoWriter(".\\media\\WithMask\\"+video, cv2.VideoWriter_fourcc(*"X264"), fps, (width, height))
            def getFrame(sec,i): 
                vidcap.set(cv2.CAP_PROP_POS_MSEC,sec*1000) 
                hasFrames,image = vidcap.read() 
                if hasFrames: 
                    #cv2.imwrite('/usr/img/kang'+"{0:0=6d}".format(i)+'.jpg',image)      # save frame as JPG file  
                    results = self.model.detect([image], verbose=0)        
                    r = results[0]  
                    image = self.display_instances(
                    image, r['rois'], r['masks'], r['class_ids'], self.class_names, r['scores'])      
                    out.write(image) 
                return hasFrames 
            sec = 0 
            i = 0
            frameRate = (1/fps) #it will capture image in each 0.5 second 
            success = getFrame(sec,i) 
            while success: 
                sec = sec + frameRate 
                sec = round(sec, 2)
                i=i+1 
                success = getFrame(sec,i)    

            print(i)
            vidcap.release()
            out.release()
            cv2.destroyAllWindows()
            return video
        return

    def random_colors(self, N):
        np.random.seed(1)
        colors = [tuple(255 * np.random.rand(3)) for _ in range(N)]
        return colors


    def apply_mask(self, image, mask, color, alpha=0.5):
        """apply mask to image"""
        for n, c in enumerate(color):
            image[:, :, n] = np.where(
                mask == 1,
                image[:, :, n] * (1 - alpha) + alpha * c,
                image[:, :, n]
            )
        return image


    def display_instances(self, image, boxes, masks, ids, names, scores):
        """
            take the image and results and apply the mask, box, and Label
        """
        n_instances = boxes.shape[0]
        colors = self.random_colors(n_instances)

        if not n_instances:
            print('NO INSTANCES TO DISPLAY')
        else:
            assert boxes.shape[0] == masks.shape[-1] == ids.shape[0]

        for i, color in enumerate(colors):
            if not np.any(boxes[i]):
                continue

            y1, x1, y2, x2 = boxes[i]
            label = names[ids[i]]
            score = scores[i] if scores is not None else None
            caption = '{} {:.2f}'.format(label, score) if score else label
            mask = masks[:, :, i]

            image = self.apply_mask(image, mask, color)
            image = cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            image = cv2.putText(
                image, caption, (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 0.7, color, 2
            )

        return image

    def runModel(self, image):
        self.getFrame(image)

    def runModel_video(self, video):
        self.getFrame_vid(video)    