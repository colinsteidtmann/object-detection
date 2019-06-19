import tensorflow as tf

from models import resnet_model
from models import rpn_model
from PIL import Image, ImageDraw, ImageEnhance
import json
import numpy as np
import csv
import sys
import h5py
#tf.config.experimental.set_memory_growth(tf.compat.v1.config.experimental.get_visible_devices()[0], True)

IMAGE_BASE_PATH = "images/"
TRAIN_CSV = "train.csv"
VALIDATION_CSV = "validation.csv"
NUM_CLASSES = 90
IMAGE_WIDTH = 600
IMAGE_HEIGHT = 1000
number_of_sections = 5
num_anchors=9

class faster_r_cnn:
    def __init__(self):
        hdf5_store = h5py.File("./cache.hdf5", "a")
        validation_images = hdf5_store["validation_resized_images"]
        validation_bounding_boxes = np.load("validation_npAnnotations.npy")

        print(validation_bounding_boxes[:5].shape)
        #self.show_image_with_annotations(validation_images[855], validation_bounding_boxes[855])
        
        rpn = self.create_rpn_model()
        print(rpn.anchor_boxes_over_image.shape)
        rpn.train_model(inputs=validation_images[:5], ground_truth_bounding_boxes=validation_bounding_boxes[:5])
        #predictions = model_rpn.predict(validation_images[855:856])
        #print(predictions[0][:,:,:,0,:])


    
    

    def show_image_with_annotations(self, numpy_image_array, numpy_bounding_boxes):
        img = Image.fromarray(numpy_image_array.astype(np.uint8()), 'RGB')
        draw = ImageDraw.Draw(img)
        for bounding_box in numpy_bounding_boxes[:,:4]:
            center_x, center_y, width, height = bounding_box
            x1 = center_x - (width / 2)
            x2 = center_x + (width / 2)
            y1 = center_y - (height / 2)
            y2 = center_y + (height /2 )
            draw.rectangle([(x1,y1), (x2,y2)])
        img.save("my_image.png")
        img.show()

    def create_rpn_model(self):
        #========= Feature extractor, resnet ============
        resnet = resnet_model.Resnet(tf=tf, image_height=IMAGE_HEIGHT, image_width=IMAGE_WIDTH, image_channels=3, number_of_sections=number_of_sections)
        res_mod = resnet.get_model()
        resnet_model_feature_map = res_mod.output
        resnet_model_height_dims = resnet_model_feature_map.get_shape().as_list()[1]
        resnet_model_output_dims = resnet_model_feature_map.get_shape().as_list()[-1]
        
        # #========= RPN ============
        rpn = rpn_model.Rpn(tf=tf, resnet_model=res_mod, image_height=IMAGE_HEIGHT, image_width=IMAGE_WIDTH, in_channels=resnet_model_output_dims, mid_channels=resnet_model_output_dims, base_anchor_size=128, anchor_ratios=[[1,1],[1,2],[2,1]], anchor_scales=[0.5,1,2,4], subsample_rate=(IMAGE_HEIGHT//resnet_model_height_dims))
        
        #outputs = model_rpn.predict(np.random.rand(1,image_height,image_width,3))
        return rpn
        #tf.summary.FileWriter("tf_graphs", sess.graph)
        #tf.keras.utils.plot_model(model=model_rpn, to_file="rpn_model.png", show_shapes=True)

if (__name__ == "__main__"):
    faster_r_cnn()