import tensorflow as tf

class Resnet:
    """ 
        Builds a model up to the final convolutional layer
        34 layer resnet
        number_of_sections = 5
        out_width & out_height == image_size*((0.5)**number_of_sections)
        returns a conv layer of shape (batch_size, out_width, out_height, 512)

        the 34 layer resnet archetecture can be defined in the paper here: https://arxiv.org/pdf/1512.03385.pdf
    """
    def __init__(self, tf, image_height, image_width, image_channels, number_of_sections=5):
        self.tf = tf
        self.IMAGE_HEIGHT = image_height
        self.IMAGE_WIDTH = image_width
        self.IMAGE_CHANNELS = image_channels
        self.number_of_sections = number_of_sections
        self.batch_norm_momentum = 0.997
        self.batch_norm_eps = 1e-5
        
        self.model = self.create_model()
        
    def get_model(self):
        return self.model

    def create_residual_block(self, inputs, filter_size, num_filters, projection_shortcut=False):
        """ 
            A projection_shortcut occurs when the input and output dimensions are changing
            Order of operations if shortcut block: batch_norm(inputs) --> relu() --> shortcut() --> batch_norm() --> relu() --> conv() --> batch_norm() --> relu() --> conv()
            Order of operations if regular block: batch_norm(inputs) --> relu() --> conv() --> batch_norm() --> relu() --> conv()
        """
        if projection_shortcut is False:
            shortcut = inputs
        else:
            """ Reduce dimensions for shortcut addition """
            shortcut = self.tf.keras.layers.SeparableConv2D(filters=num_filters, kernel_size=[1, 1], strides=[2, 2], padding="valid", data_format="channels_last")(inputs)
            shortcut = self.tf.keras.layers.BatchNormalization(axis=-1, momentum=self.batch_norm_momentum, epsilon=self.batch_norm_eps)(shortcut)

        
        conv1 = self.tf.keras.layers.SeparableConv2D(filters=num_filters, kernel_size=[filter_size, filter_size], strides=[1, 1], padding="same", data_format="channels_last")(shortcut)
        norm_conv1 = self.tf.keras.layers.BatchNormalization(axis=-1, momentum=self.batch_norm_momentum, epsilon=self.batch_norm_eps)(conv1)
        relu_conv1 = self.tf.keras.layers.Activation("relu")(norm_conv1)

        conv2 = self.tf.keras.layers.SeparableConv2D(filters=num_filters, kernel_size=[filter_size, filter_size], strides=[1, 1], padding="same", data_format="channels_last")(relu_conv1)
        norm_conv2 = self.tf.keras.layers.BatchNormalization(axis=-1, momentum=self.batch_norm_momentum, epsilon=self.batch_norm_eps)(conv2)
        
        shortcut_connected_conv2 = self.tf.keras.layers.Add()([norm_conv2, shortcut])
        relu_conv2 = self.tf.keras.layers.Activation("relu")(shortcut_connected_conv2)

        return relu_conv2
        

    def create_model(self):
        orig_inputs = self.tf.keras.Input(shape=(self.IMAGE_HEIGHT, self.IMAGE_WIDTH, self.IMAGE_CHANNELS))
        #===============================================================
        #Conv1
        #===============================================================
        """ First reduce dimensions """
        inputs = self.tf.keras.layers.SeparableConv2D(filters=64, kernel_size=[1, 1], strides=[2, 2], padding="valid", data_format="channels_last")(orig_inputs)
        inputs = self.tf.keras.layers.BatchNormalization(axis=-1, momentum=self.batch_norm_momentum, epsilon=self.batch_norm_eps)(inputs)
        inputs = self.tf.keras.layers.Activation("relu")(inputs)
        """ Next perform first actual conv """
        conv1 = self.tf.keras.layers.SeparableConv2D(filters=64, kernel_size=[7, 7], strides=[1, 1], padding="same", data_format="channels_last")(inputs)
        norm_conv1 = self.tf.keras.layers.BatchNormalization(axis=-1, momentum=self.batch_norm_momentum, epsilon=self.batch_norm_eps)(conv1)
        relu_conv1 = relu_conv2 = self.tf.keras.layers.Activation("relu")(norm_conv1)
        conv1_pool = self.tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=[2, 2], padding="valid", data_format="channels_last")(relu_conv1)
        
        #===============================================================
        #Conv2
        #===============================================================
        conv2_1 = self.create_residual_block(inputs=conv1_pool, filter_size=3, num_filters=64)
        conv2_2 = self.create_residual_block(inputs=conv2_1, filter_size=3, num_filters=64)
        conv2_3 = self.create_residual_block(inputs=conv2_2, filter_size=3, num_filters=64)
        
        if (self.number_of_sections == 2):
            model = self.tf.keras.Model(inputs=orig_inputs, outputs=conv2_3)
            return model
        #===============================================================
        #Conv3
        #===============================================================
        conv3_1 = self.create_residual_block(inputs=conv2_3, filter_size=3, num_filters=128, projection_shortcut=True)
        conv3_2 = self.create_residual_block(inputs=conv3_1, filter_size=3, num_filters=128)
        conv3_3 = self.create_residual_block(inputs=conv3_2, filter_size=3, num_filters=128)
        conv3_4 = self.create_residual_block(inputs=conv3_3, filter_size=3, num_filters=128)

        if (self.number_of_sections == 3):
            model = self.tf.keras.Model(inputs=orig_inputs, outputs=conv3_4)
            return model
        #===============================================================
        #Conv4
        #===============================================================
        conv4_1 = self.create_residual_block(inputs=conv3_4, filter_size=3, num_filters=256, projection_shortcut=True)
        conv4_2 = self.create_residual_block(inputs=conv4_1, filter_size=3, num_filters=256)
        conv4_3 = self.create_residual_block(inputs=conv4_2, filter_size=3, num_filters=256)
        conv4_4 = self.create_residual_block(inputs=conv4_3, filter_size=3, num_filters=256)
        conv4_5 = self.create_residual_block(inputs=conv4_4, filter_size=3, num_filters=256)
        conv4_6 = self.create_residual_block(inputs=conv4_5, filter_size=3, num_filters=256)

        if (self.number_of_sections == 4):
            model = self.tf.keras.Model(inputs=orig_inputs, outputs=conv4_6)
            return model
        #===============================================================
        #Conv5
        #===============================================================
        conv5_1 = self.create_residual_block(inputs=conv4_6, filter_size=3, num_filters=512, projection_shortcut=True)
        conv5_2 = self.create_residual_block(inputs=conv5_1, filter_size=3, num_filters=512)
        conv5_3 = self.create_residual_block(inputs=conv5_2, filter_size=3, num_filters=512)

        if (self.number_of_sections == 5):
            model = self.tf.keras.Model(inputs=orig_inputs, outputs=conv5_3)
            return model
        else:
            return None
    
    