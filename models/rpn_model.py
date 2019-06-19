import numpy as np
import sys
from PIL import Image, ImageDraw, ImageEnhance, ImageColor

class Rpn:
    def __init__(self, tf, resnet_model, image_height, image_width, in_channels, mid_channels, base_anchor_size, anchor_ratios, anchor_scales, subsample_rate):
        """
            in_channels for resnet: 512
            mid_channels: 512
            base_anchor_size: 128
            anchor_ratios: [[1,1],[1,2],[2,1]]
            anchor_scales: [1,2,3,4]
            subsample_rate: (image_size//feature_map_size), ex. (1000//32) --> 31
        """
        self.tf = tf
        self.resnet_model = resnet_model
        self.resnet_model_inputs = self.resnet_model.input
        self.resnet_model_feature_map = resnet_model.output
        self.image_height = image_height
        self.image_width = image_width
        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.base_anchor_size = base_anchor_size
        self.anchor_ratios = anchor_ratios
        self.anchor_scales = anchor_scales
        self.subsample_rate = subsample_rate


        self.anchor_boxes = self.generate_anchor_boxes(self.base_anchor_size, self.anchor_ratios, self.anchor_scales)
        self.anchor_boxes_over_image = self.tf.convert_to_tensor(self.generate_anchor_boxes_over_image(self.anchor_boxes, self.image_height, self.image_width, self.subsample_rate), dtype=self.tf.float64)
        self.total_number_of_anchor_boxes = self.tf.size(self.anchor_boxes_over_image)

        self.model = self.create_model()

    def get_model(self):
        return self.model

    def create_model(self):
        self.tf.keras.backend.set_floatx('float64')
        rpn_conv = self.tf.keras.layers.SeparableConv2D(filters=512, kernel_size=[3, 3], strides=[1, 1], padding="same", data_format="channels_last", activation="relu")(self.resnet_model.output)
        rpn_class_score = self.tf.keras.layers.SeparableConv2D(filters=24, kernel_size=[1, 1], strides=[1, 1], padding="valid", data_format="channels_last", activation="softmax")(rpn_conv)
        rpn_class_score_shape = rpn_class_score.get_shape().as_list()
        rpn_class_score = self.tf.keras.layers.Reshape(rpn_class_score_shape[1:3] + [rpn_class_score_shape[3] // 2, 2])(rpn_class_score)
        rpn_class_score = self.tf.keras.backend.cast(rpn_class_score, dtype='float64')
        
        rpn_bbox_pred = self.tf.keras.layers.SeparableConv2D(filters=48, kernel_size=[1, 1], strides=[1, 1], padding="valid", data_format="channels_last")(rpn_conv)
        rpn_bbox_pred_shape = rpn_bbox_pred.get_shape().as_list()
        rpn_bbox_pred = self.tf.keras.layers.Reshape(rpn_bbox_pred_shape[1:3] + [rpn_bbox_pred_shape[3] // 4, 4])(rpn_bbox_pred)
        rpn_bbox_pred = self.tf.keras.backend.cast(rpn_bbox_pred, dtype='float64')

        rpn_model = self.tf.keras.Model(inputs=self.resnet_model.input, outputs=[rpn_class_score, rpn_bbox_pred])

        self.optimizer = self.tf.keras.optimizers.Nadam(0.001)
        return rpn_model

    def get_iou(self, anchor_box_predictions, ground_truth_bounding_box, giou=False):
        ground_truth_bounding_box = self.tf.convert_to_tensor(ground_truth_bounding_box, dtype=self.tf.float64)
        """ Predicted and ground truth bounding box coordinates """
        anchor_box_predictions_center_x, anchor_box_predictions_center_y, anchor_box_predictions_width, anchor_box_predictions_height = self.tf.slice(anchor_box_predictions, [0, 0, 0, 0, 0], [-1, -1, -1, -1, 1]), self.tf.slice(anchor_box_predictions, [0, 0, 0, 0, 1], [-1, -1, -1, -1, 1]), self.tf.slice(anchor_box_predictions, [0, 0, 0, 0, 2], [-1, -1, -1, -1, 1]), self.tf.slice(anchor_box_predictions, [0, 0, 0, 0, 3], [-1, -1, -1, -1, 1])
        anchor_box_predictions_x1 = anchor_box_predictions_center_x - (anchor_box_predictions_width / 2)
        anchor_box_predictions_x2 = anchor_box_predictions_center_x + (anchor_box_predictions_width / 2)
        anchor_box_predictions_y1 = anchor_box_predictions_center_y - (anchor_box_predictions_height / 2)
        anchor_box_predictions_y2 = anchor_box_predictions_center_y + (anchor_box_predictions_height / 2)
        
        """ For the predicted box ensure x2>x1 and y2>y1: """
        anchor_box_predictions_x1, anchor_box_predictions_x2 = self.tf.minimum(anchor_box_predictions_x1, anchor_box_predictions_x2), self.tf.maximum(anchor_box_predictions_x1, anchor_box_predictions_x2)
        anchor_box_predictions_y1, anchor_box_predictions_y2 = self.tf.minimum(anchor_box_predictions_y1, anchor_box_predictions_y2), self.tf.maximum(anchor_box_predictions_y1, anchor_box_predictions_y2)
        
        """ ground truth bounding box coordinates """
        ground_truth_bounding_boxes_center_x, ground_truth_bounding_boxes_center_y, ground_truth_bounding_boxes_width, ground_truth_bounding_boxes_height = ground_truth_bounding_box[:,0:1], ground_truth_bounding_box[:,1:2], ground_truth_bounding_box[:,2:3], ground_truth_bounding_box[:,3:4]

        ground_truth_bounding_boxes_x1 = ground_truth_bounding_boxes_center_x - (ground_truth_bounding_boxes_width / 2)
        ground_truth_bounding_boxes_x2 = ground_truth_bounding_boxes_center_x + (ground_truth_bounding_boxes_width / 2)
        ground_truth_bounding_boxes_y1 = ground_truth_bounding_boxes_center_y - (ground_truth_bounding_boxes_height / 2)
        ground_truth_bounding_boxes_y2 = ground_truth_bounding_boxes_center_y + (ground_truth_bounding_boxes_height / 2)
        ground_truth_bounding_boxes_x1, ground_truth_bounding_boxes_x2, ground_truth_bounding_boxes_y1, ground_truth_bounding_boxes_y2 = self.tf.reshape(ground_truth_bounding_boxes_x1, [1,1,1,1,-1]), self.tf.reshape(ground_truth_bounding_boxes_x2, [1,1,1,1,-1]), self.tf.reshape(ground_truth_bounding_boxes_y1, [1,1,1,1,-1]), self.tf.reshape(ground_truth_bounding_boxes_y2, [1,1,1,1,-1])


        """ Get areas of boxes """
        anchor_box_predictions_area = (anchor_box_predictions_x2 - anchor_box_predictions_x1) * (anchor_box_predictions_y2 - anchor_box_predictions_y1)
        ground_truth_bounding_boxes_area = (ground_truth_bounding_boxes_x2 - ground_truth_bounding_boxes_x1) * (ground_truth_bounding_boxes_y2 - ground_truth_bounding_boxes_y1)

        """ Calculate intersection between prediction boxes and ground truth boxes """
        
        x1_intersection, x2_intersection = self.tf.maximum(anchor_box_predictions_x1, ground_truth_bounding_boxes_x1), self.tf.minimum(anchor_box_predictions_x2, ground_truth_bounding_boxes_x2)
        y1_intersection, y2_intersection = self.tf.maximum(anchor_box_predictions_y1, ground_truth_bounding_boxes_y1), self.tf.minimum(anchor_box_predictions_y2, ground_truth_bounding_boxes_y2)


        """ Calculate intersection area """
        intersection = self.tf.where(self.tf.logical_and(x2_intersection > x1_intersection, y2_intersection > y1_intersection), (x2_intersection - x1_intersection) * (y2_intersection - y1_intersection), 0)

        """ Find the coordinates of smallest enclosing box (union) """
        x1_coord, x2_coord = self.tf.minimum(anchor_box_predictions_x1, ground_truth_bounding_boxes_x1), self.tf.maximum(anchor_box_predictions_x2, ground_truth_bounding_boxes_x2)
        y1_coord, y2_coord = self.tf.minimum(anchor_box_predictions_y1, ground_truth_bounding_boxes_y1), self.tf.maximum(anchor_box_predictions_y2, ground_truth_bounding_boxes_y2)

        """ Get area of union box """
        union_area = (x2_coord - x1_coord) * (y2_coord - y1_coord)
        
        """ Intersection over union """
        iou = intersection / (anchor_box_predictions_area + ground_truth_bounding_boxes_area - intersection)
        Giou = iou - ((union_area - (anchor_box_predictions_area + ground_truth_bounding_boxes_area - intersection)) / union_area)
        
        """ get max iou and giou and bounding box for max iou/Giou"""
        if giou is True:
            bounding_box_true = self.tf.slice(self.tf.gather(ground_truth_bounding_box,self.tf.argmax(Giou, axis=4)),[0,0,0,0,0],[-1,-1,-1,-1,4])
        else:
            bounding_box_true = self.tf.slice(self.tf.gather(ground_truth_bounding_box,self.tf.argmax(iou, axis=4)),[0,0,0,0,0],[-1,-1,-1,-1,4])
        iou = self.tf.reduce_max(iou, axis=4)
        Giou = self.tf.reduce_max(Giou, axis=4)
        return (Giou,bounding_box_true) if giou else (iou, bounding_box_true)

    def model_loss(self, y_true, y_pred):
        mini_batch_sample_size = 256
        object_predictions = y_pred[0]
        anchor_box_change_predictions = y_pred[1]
        anchor_box_predictions = self.anchor_boxes_over_image + anchor_box_change_predictions
        anchor_boxes_giou, anchor_boxes_nearest_bounding_box = self.get_iou(anchor_box_predictions, y_true, giou=True)
        object_prediction_labels = self.tf.one_hot(self.tf.where(anchor_boxes_giou > 0.5, 1, 0),depth=2, on_value=1, off_value=0)

        
        object_predictions_flat = self.tf.reshape(object_predictions,[-1,2])
        object_prediction_labels_flat = self.tf.reshape(object_prediction_labels, [-1, 2])
        anchor_boxes_flat = self.tf.reshape(self.anchor_boxes_over_image,[-1, 4])
        anchor_box_predictions_flat = self.tf.reshape(anchor_box_predictions,[-1, 4])
        anchor_boxes_nearest_bounding_box_flat = self.tf.reshape(anchor_boxes_nearest_bounding_box, [-1, 4])

        """Shuffle flattened y_pred and y_true all in the same order"""
        random_indices = self.tf.random.shuffle(self.tf.range(self.tf.gather(self.tf.shape(object_predictions_flat),0)))
        object_predictions_flat = self.tf.gather(object_predictions_flat, random_indices)
        object_prediction_labels_flat = self.tf.gather(object_prediction_labels_flat, random_indices)
        anchor_boxes_flat = self.tf.gather(anchor_boxes_flat, random_indices)
        anchor_box_predictions_flat = self.tf.gather(anchor_box_predictions_flat, random_indices)
        anchor_boxes_nearest_bounding_box_flat = self.tf.gather(anchor_boxes_nearest_bounding_box_flat, random_indices)

        """ Sort in ascending order from background to foreground """
        ind_sorted = self.tf.argsort(self.tf.argmax(object_prediction_labels_flat, axis=1),axis=0)
        object_predictions_flat = self.tf.gather(object_predictions_flat, ind_sorted)
        object_prediction_labels_flat = self.tf.gather(object_prediction_labels_flat, ind_sorted)
        anchor_boxes_flat = self.tf.gather(anchor_boxes_flat, ind_sorted)
        anchor_box_predictions_flat = self.tf.gather(anchor_box_predictions_flat, ind_sorted)
        anchor_boxes_nearest_bounding_box_flat = self.tf.gather(anchor_boxes_nearest_bounding_box_flat, ind_sorted)

        """ Get 128 background anchors and 128 foreground anchors and merge them into a single batch """
        split_amount = mini_batch_sample_size // 2
        #Background
        background_object_predictions_flat = self.tf.slice(object_predictions_flat, [0,0],[split_amount,-1])
        background_object_prediction_labels_flat = self.tf.slice(object_prediction_labels_flat, [0,0], [split_amount,-1])
        background_anchor_boxes_flat = self.tf.slice(anchor_boxes_flat, [0, 0], [split_amount, -1])
        background_anchor_box_predictions_flat = self.tf.slice(anchor_box_predictions_flat, [0, 0], [split_amount, -1])
        background_anchor_boxes_nearest_bounding_box_flat = self.tf.slice(anchor_boxes_nearest_bounding_box_flat, [0,0], [split_amount,-1])

        #Foreground
        foreground_object_predictions_flat = self.tf.slice(object_predictions_flat, [split_amount,0], [-1,-1])
        foreground_object_prediction_labels_flat = self.tf.slice(object_prediction_labels_flat, [split_amount,0], [-1,-1])
        foreground_anchor_boxes_flat = self.tf.slice(anchor_boxes_flat, [split_amount,0], [-1,-1])
        foreground_anchor_box_predictions_flat = self.tf.slice(anchor_box_predictions_flat, [split_amount,0], [-1,-1])
        foreground_anchor_boxes_nearest_bounding_box_flat = self.tf.slice(anchor_boxes_nearest_bounding_box_flat, [split_amount,0], [-1,-1])
        

        #Merge
        final_object_predictions_flat = self.tf.concat((background_object_predictions_flat,foreground_object_predictions_flat), axis=0)
        final_object_prediction_labels_flat = self.tf.concat((background_object_prediction_labels_flat, foreground_object_prediction_labels_flat), axis=0)
        all_anchor_boxes_flat = self.tf.concat((background_anchor_boxes_flat, foreground_anchor_boxes_flat), axis=0)
        all_anchor_box_predictions_flat = self.tf.concat((background_anchor_box_predictions_flat, foreground_anchor_box_predictions_flat), axis=0)
        all_anchor_boxes_nearest_bounding_box_flat = self.tf.concat((background_anchor_boxes_nearest_bounding_box_flat, foreground_anchor_boxes_nearest_bounding_box_flat), axis=0)
        #Take only foreground anchor and the closest ground truth boxes from the mini batch for bounding box regression
        final_anchor_boxes_flat = self.tf.gather(foreground_anchor_boxes_flat, self.tf.squeeze(self.tf.where(self.tf.squeeze(self.tf.equal(self.tf.slice(foreground_object_prediction_labels_flat,[0,1],[-1,1]),1),-1)),-1))
        final_anchor_box_predictions_flat = self.tf.gather(foreground_anchor_box_predictions_flat, self.tf.squeeze(self.tf.where(self.tf.squeeze(self.tf.equal(self.tf.slice(foreground_object_prediction_labels_flat,[0,1],[-1,1]),1),-1)),-1))
        final_anchor_boxes_nearest_bounding_box_flat = self.tf.gather(foreground_anchor_boxes_nearest_bounding_box_flat, self.tf.squeeze(self.tf.where(self.tf.squeeze(self.tf.equal(self.tf.slice(foreground_object_prediction_labels_flat,[0,1],[-1,1]),1),-1)),-1))

        #losses
        anchor_boxes_loss = self.anchors_loss(final_object_prediction_labels_flat, final_object_predictions_flat, all_anchor_boxes_nearest_bounding_box_flat, all_anchor_box_predictions_flat)
        regression_loss = self.bounding_box_regression_loss(final_anchor_boxes_nearest_bounding_box_flat, final_anchor_box_predictions_flat, final_anchor_boxes_flat)
        sum_anchor_boxes_loss = self.tf.reduce_sum(anchor_boxes_loss)
        sum_regression_loss = self.tf.reduce_sum(regression_loss)
        return sum_anchor_boxes_loss + sum_regression_loss

    def train_model(self, inputs, ground_truth_bounding_boxes):
        print([x.shape for x in self.model.output])
        mini_batch_sample_size = 256
        for image, bb_true in zip(list(inputs), ground_truth_bounding_boxes):
            image = image.reshape(1, self.image_height, self.image_width, 3)
            predictions = self.model.predict(image)
            loss = self.model_loss(bb_true, predictions)
            print(loss)
            gradients = self.optimizer.get_gradients(loss, self.model.trainable_variables)


    def anchors_loss(self, classification_true, classification_pred, anchor_box_true, anchor_box_pred, reg_gamma=10):
        mini_batch_size = self.tf.gather(self.tf.shape(classification_pred),0)
        classification_loss = self.logloss(classification_true, classification_pred)
        true_objects = self.tf.cast(self.tf.argmax(classification_true, -1),dtype=self.tf.float64)
        reg_loss = true_objects*self.regloss(anchor_box_true, anchor_box_pred)
        anchors_loss = (1 / mini_batch_size) * classification_loss + reg_gamma * (1 / self.total_number_of_anchor_boxes) * reg_loss
        return anchors_loss

    def regloss(self, box_true, box_predictions, alpha=1):
        error = box_predictions - box_true 
        smooth_l1 = self.tf.reduce_sum(self.tf.where(self.tf.abs(error) < 1, 0.5 * self.tf.square(error), self.tf.abs(error) - 0.5), axis=1)
        return smooth_l1
        
    
    def logloss(self, true_label, predicted, eps=1e-15):
        p = self.tf.clip_by_value(predicted, eps, 1 - eps)
        log_loss = self.tf.reduce_sum(self.tf.where(self.tf.equal(true_label, 1), -self.tf.math.log(p), -self.tf.math.log(1 - p)), axis=1)
        return log_loss
    
    def bounding_box_regression_loss(self, true_box, predicted_box, anchor_box):
        p_x, p_y, p_w, p_h = self.tf.slice(predicted_box, [0,0],[-1,1]), self.tf.slice(predicted_box, [0,1],[-1,1]), self.tf.slice(predicted_box, [0,2],[-1,1]), self.tf.slice(predicted_box, [0,3],[-1,1])
        t_x, t_y, t_w, t_h = self.tf.slice(true_box, [0,0],[-1,1]), self.tf.slice(true_box, [0,1],[-1,1]), self.tf.slice(true_box, [0,2],[-1,1]), self.tf.slice(true_box, [0,3],[-1,1])
        a_x, a_y, a_w, a_h = self.tf.slice(anchor_box, [0, 0], [-1, 1]), self.tf.slice(anchor_box, [0, 1], [-1, 1]), self.tf.slice(anchor_box, [0, 2], [-1, 1]), self.tf.slice(anchor_box, [0, 3], [-1, 1])
        
        p_x = (p_x - a_x) / a_w
        p_y = (p_y - a_y) / a_h
        p_w = self.tf.math.log(p_w / a_w)
        p_h = self.tf.math.log(p_h / a_h)
        
        t_x = (t_x - a_x) / a_w
        t_y = (t_y - a_y) / a_h
        t_w = self.tf.math.log(t_w / a_w)
        t_h = self.tf.math.log(t_h / a_h)
        
        predicted = self.tf.concat((p_x, p_y, p_w, p_h), axis=1)
        truth = self.tf.concat((t_x, t_y, t_w, t_h), axis=1)
        
        regression_loss = self.tf.reduce_sum(self.tf.square(predicted-truth),axis=1)
        return regression_loss

    def generate_anchor_boxes(self, base_anchor_size, anchor_ratios, anchor_scales):
        """ 
            Every anchor box is different 
            Number of anchor boxes: len(anchor_ratios)*len(anchor_scales)
        """

        anchor_boxes = []
        for scale in anchor_scales:
            for aspect_ratio in anchor_ratios:
                height = base_anchor_size*scale*aspect_ratio[0]
                width = base_anchor_size*scale*aspect_ratio[1]
                anchor_box = [width, height]
                anchor_boxes.append(anchor_box)

        return anchor_boxes
    
    def generate_anchor_boxes_over_image(self, anchor_boxes, image_height, image_width, subsample_rate):
        """ 
            Returns an array of all the anchor boxes, each with shape [anchor_box_center_x, anchor_box_center_y, width, height]
        """
        anchor_boxes_over_image = np.zeros(( image_height//subsample_rate, image_width//subsample_rate, len(anchor_boxes), 4))
        for row in range(anchor_boxes_over_image.shape[0]):
            for col in range(anchor_boxes_over_image.shape[1]):
                for anchor_idx in range(anchor_boxes_over_image.shape[2]):
                    anchor_boxes_over_image[row,col,anchor_idx] = [(row+1)*subsample_rate, (col+1)*subsample_rate, anchor_boxes[anchor_idx][0], anchor_boxes[anchor_idx][1]]
        return anchor_boxes_over_image
    
    def show_boxes_over_black_screen(self, ab_x1, ab_x2, ab_y1, ab_y2, grounding_box_truths, iou_scores):
        for image in range(ab_x1.shape[0]):
            gbt_x1, gbt_x2, gbt_y1, gbt_y2 = (grounding_box_truths[image][:,0]-(grounding_box_truths[image][:,2]/2)), grounding_box_truths[image][:,0]+(grounding_box_truths[image][:,2]/2), grounding_box_truths[image][:,1]-(grounding_box_truths[image][:,3]/2), grounding_box_truths[image][:,1]+(grounding_box_truths[image][:,3]/2)
            img = Image.new('RGB', (self.image_height, self.image_width), color='black')
            draw = ImageDraw.Draw(img)

            """ Draw anchor boxes """
            for row in range(ab_x1.shape[1]):
                for col in range(ab_x1.shape[2]):
                    for anchor in range(ab_x1.shape[3]):
                        draw.rectangle([(ab_x1[image, row, col, anchor], ab_y1[image, row, col, anchor]), (ab_x2[image, row, col, anchor], ab_y2[image, row, col, anchor])], outline='white', fill='black')
                        
            """ Draw ground truth bounding boxes """
            for true_bounding_box in range(grounding_box_truths[image].shape[0]):
                draw.rectangle([(gbt_x1[true_bounding_box], gbt_y1[true_bounding_box]), (gbt_x2[true_bounding_box], gbt_y2[true_bounding_box])], outline='yellow', fill='blue')

            """ Draw iou scores """
            for row in range(ab_x1.shape[1]):
                for col in range(ab_x1.shape[2]):
                    for anchor in range(ab_x1.shape[3]):
                        if (iou_scores[image, row, col, anchor] > 0.5):
                            draw.rectangle([(ab_x1[image, row, col, anchor], ab_y1[image, row, col, anchor]), (ab_x2[image, row, col, anchor], ab_y2[image, row, col, anchor])], outline='white', fill='orange')
                            draw.text((ab_x1[image, row, col, anchor], ab_y1[image, row, col, anchor]), "{}".format((iou_scores[image,row,col,anchor].astype(str))[:3]), fill="black")

        img.save("my_image.png")
        img.show()
    
    def plot_anchor_box_points_over_black_screen(self, anchor_boxes):
        img = Image.new('RGB', (self.image_width, self.image_height), color='black')
        draw = ImageDraw.Draw(img)
        for row in range(anchor_boxes.shape[0]):
            for col in range(anchor_boxes.shape[1]):
                for anchor in range(anchor_boxes.shape[2]):
                    draw.point([anchor_boxes[row,col,anchor,1], anchor_boxes[row,col,anchor,0]], fill='white')
        img.save("my_image.png")
        img.show()

    
            