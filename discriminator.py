from tensorflow.keras import Input, Model
from tensorflow.keras.layers import BatchNormalization, LeakyReLU, Dense, Activation, Flatten, Conv2D


class Discriminator:

    def __init__(self, the_shape):
        self.the_shape = the_shape
        self.input_layer = Input(the_shape)

    @staticmethod
    def __define_residual_block(previous_block, current_kernel_size, current_number_of_features_maps,
                                current_strides):
        conv2d = Conv2D(filters=current_number_of_features_maps, kernel_size=current_kernel_size, strides=current_strides, padding="same")(previous_block)
        batch_normalization = BatchNormalization(momentum=0.5)(conv2d)
        leaky_relu = LeakyReLU(alpha=0.2)(batch_normalization)
        return leaky_relu

    def __define_first_block(self):
        conv2d = Conv2D(filters=64, kernel_size=3, padding="same", strides=1)(self.input_layer)
        leaky_relu = LeakyReLU(alpha=0.2)(conv2d)
        return leaky_relu

    def __define_residual_blocks(self):
        residual_blocks_kernel_size = [3, 3, 3, 3, 3, 3]
        residual_blocks_number_of_features_maps = [128, 128, 256, 256, 512, 512]
        residual_blocks_strides = [1, 2, 1, 2, 1, 2]

        stacked_residual_block = self.__define_residual_block(Discriminator.__define_first_block(self), 3, 64, 2)

        for x in range(6):
            current_kernel_size = residual_blocks_kernel_size[x]
            current_number_of_features_maps = residual_blocks_number_of_features_maps[x]
            current_strides = residual_blocks_strides[x]
            stacked_residual_block = self.__define_residual_block(stacked_residual_block, current_kernel_size,
                                                                  current_number_of_features_maps,
                                                                  current_strides)

        return stacked_residual_block

    def __last_block(self):
        flatten = Flatten()(Discriminator.__define_residual_blocks(self))
        dense = Dense(1024)(flatten)
        leaky_relu = LeakyReLU(alpha=0.2)(dense)
        dense = Dense(1)(leaky_relu)
        return dense

    def define_model(self):
        dense = Discriminator.__last_block(self)
        output = Activation('sigmoid')(dense)
        model = Model(inputs=self.input_layer, outputs=output)
        return model
