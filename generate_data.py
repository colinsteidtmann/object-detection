from PIL import Image, ImageDraw, ImageEnhance
import h5py
import sys
import json
import numpy as np
import csv
import argparse


class DataGenerator():
    def __init__(self, csv_file, image_base_path, image_height, image_width):
        self.image_dataset_name, self.image_annotations_list = self.generate_numpy_data(csv_file, image_base_path, image_height, image_width)
        print("saving data to numpy array files")
        self.save_numpy_data(csv_file, self.image_dataset_name, self.image_annotations_list)
    
    def save_numpy_data(self, csv_file, image_dataset_name, numpy_annotations):
        print("saved {} file image data as h5py dataset in {} with dataset name as {}".format(csv_file, "./cache.hdf5", image_dataset_name))
        print("open it with the commands\ncommand 1: {}\ncommand 2: {}".format('hdf5_store = h5py.File("./cache.hdf5", "a")', 'image_data = hdf5_store["{}"]'.format(image_dataset_name)))
        print("saving {} file annotation data as {}".format(csv_file, csv_file[:csv_file.index(".csv")] + "_npAnnotations.npy"))
        np.save(csv_file[:csv_file.index(".csv")] + "_npAnnotations", numpy_annotations)
        print("saved {} as numpy data".format(csv_file))

    def generate_numpy_data(self, csv_file, IMAGE_BASE_PATH, IMAGE_HEIGHT, IMAGE_WIDTH):
        IMAGE_HEIGHT = int(IMAGE_HEIGHT)
        IMAGE_WIDTH = int(IMAGE_WIDTH)
        with open(csv_file, "r") as file:
            """ Create empty arrays to store data """
            reader = csv.reader(file, delimiter=",")
            num_images = sum(1 for line in open(csv_file))
            hdf5_store = h5py.File("./cache.hdf5", "a")
            image_dataset_name = csv_file[:csv_file.index(".csv")] + "_resized_images"
            if image_dataset_name in hdf5_store: del hdf5_store[image_dataset_name] 
            resized_images = hdf5_store.create_dataset(image_dataset_name, (num_images, IMAGE_HEIGHT, IMAGE_WIDTH, 3), compression="gzip")
            tmp_update_time = 500
            resized_images_tmp_np_array = np.zeros((tmp_update_time, IMAGE_HEIGHT, IMAGE_WIDTH, 3))
            image_annotations_list = []
            
            for index, row in enumerate(reader):
                print("On line {}/{} of {} ...".format(index,num_images, csv_file), end="\r")
                """ Get parts of data line from csv file """
                image_path = str(IMAGE_BASE_PATH+row[0])
                image_height = int(row[1])
                image_width = int(row[2])
                image_annotations = json.loads(row[4])
                image_id = int(row[3])

                y_scale = IMAGE_HEIGHT / image_height
                x_scale = IMAGE_WIDTH / image_width
                
                """ Find image, resize it, and turn it into a numpy array """
                with Image.open(image_path) as img:
                    img = img.resize((IMAGE_WIDTH, IMAGE_HEIGHT))
                    img = img.convert('RGB')
                    img = np.array(img, dtype=np.float32)
                    resized_images_tmp_np_array[index%tmp_update_time] = img
                    """ Transfer tmp numpy array to h5py cache """
                    if ((index + 1) % tmp_update_time) == 0:
                        resized_images[((index + 1) - tmp_update_time):(index + 1)] = resized_images_tmp_np_array
                        resized_images_tmp_np_array = np.zeros((tmp_update_time, IMAGE_HEIGHT, IMAGE_WIDTH, 3))

                """ 
                    Find annotations and turn them into a numpy array, each in the formate 
                    (bbox_x_center, bbox_y_center, bbox_scaled_width, bbox_scaled_height, annotation_class) 
                """  
                annotations = np.zeros((len(image_annotations), 5))  
                for idx,image_annotation in enumerate(image_annotations):
                    annotation_image_id = image_annotation[2]
                    bbox = image_annotation[3]
                    top_left_x, top_left_y, width, height = bbox
                    bbox[0] = int(np.round(int(top_left_x)*x_scale))
                    bbox[1] = int(np.round(int(top_left_y)*y_scale))
                    bbox[2] = int(np.round(int(width)*x_scale))
                    bbox[3] = int(np.round(int(height)*y_scale))
                    bbox[0] += (bbox[2]//2)
                    bbox[1] += (bbox[3]//2)

                    annotation_class = [image_annotation[4]]
                    annotations[idx] = bbox+annotation_class
                    if (annotation_image_id != image_id):
                        print("image and annotation id's don't match, something got out of order\nImage id: {}\nAnnotation id: {}".format(image_id, annotation_image_id))
                        sys.exit()
                image_annotations_list.append(annotations)
            
            image_annotations_list = np.array(image_annotations_list)
            return image_dataset_name, image_annotations_list


def main(dataset_folder, train_file_name, validation_file_name, split_ratio):
    DATASET_FOLDER = dataset_folder
    TRAIN_OUTPUT_FILE = train_file_name
    VALIDATION_OUTPUT_FILE = validation_file_name
    SPLIT_RATIO = split_ratio

    with open("annotations/instances_val2017.json", 'r') as f:
        data = json.load(f)
        print("Collecting images and annotations...")
        data_list = []
        for image in data.get('images'):
            file_name = image.get('file_name')
            height = image.get('height')
            width = image.get('width')
            image_id = image.get('id')
            annotations_for_image = [[val for key,val in list(annotation.items()) if key != 'segmentation']  for annotation in data.get('annotations') if annotation.get('image_id') == image_id]
            data_list.append([file_name, height, width, image_id, annotations_for_image])
        
        print("Writing annotations to train and validation csv files")
        with open(TRAIN_OUTPUT_FILE, "w") as train, open(VALIDATION_OUTPUT_FILE, "w") as validate:
            writer = csv.writer(train, delimiter=",")
            writer2 = csv.writer(validate, delimiter=",")

            for date_train_row in data_list[:int(len(data_list)*SPLIT_RATIO)]:
                writer.writerow(date_train_row)
            
            for data_validate_row in data_list[int(len(data_list)*SPLIT_RATIO):]:
                writer2.writerow(data_validate_row)
        
        print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_height")
    parser.add_argument("--image_width")
    args = parser.parse_args()

    DATASET_FOLDER = "images/"
    TRAIN_OUTPUT_FILE = "train.csv"
    VALIDATION_OUTPUT_FILE = "validation.csv"
    SPLIT_RATIO = 0.8
    
    main(DATASET_FOLDER, TRAIN_OUTPUT_FILE, VALIDATION_OUTPUT_FILE, SPLIT_RATIO)
    print("csv files made, now converting them to numpy and h5py files")
    #DataGenerator(TRAIN_OUTPUT_FILE, image_base_path=DATASET_FOLDER, image_height=args.image_height, image_width=args.image_width)
    DataGenerator(VALIDATION_OUTPUT_FILE, image_base_path=DATASET_FOLDER, image_height=args.image_height, image_width=args.image_width)



