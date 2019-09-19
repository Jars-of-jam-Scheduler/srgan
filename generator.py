from tensorflow.keras import Input, Model
from tensorflow.keras.layers import LeakyReLU, BatchNormalization, PReLU, UpSampling2D, Activation, Conv2D
from tensorflow.keras.layers import add


class Generator:

    def __init__(self, the_shape):
        self.the_shape = the_shape
        self.input_layer = Input(the_shape)

    @staticmethod
    def __residual_block(previous_block):
        conv2d = Conv2D(filters=64, kernel_size=3, padding='same', strides=1)(previous_block)
        batch_normalization = BatchNormalization(momentum=0.5)(conv2d)
        prelu = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[1,2])(batch_normalization)
        conv2d = Conv2D(filters=64, kernel_size=3, padding='same', strides=1)(prelu)
        batch_normalization = BatchNormalization(momentum=0.5)(conv2d)
        added = add([previous_block, batch_normalization])
        return added

    @staticmethod
    def __repeat_residual_blocks_computation(result_of_first_block):
        last_value = Generator.__residual_block(result_of_first_block)
        for _ in range(15):
            last_value = Generator.__residual_block(last_value)
        return last_value

    @staticmethod
    def __first_last_block(result_of_first_block):
        blocks = Generator.__repeat_residual_blocks_computation(result_of_first_block)
        conv2d = Conv2D(filters=64, kernel_size=3, padding="same", strides=1)(blocks)
        batch_normalization = BatchNormalization(momentum = 0.5)(conv2d)
        added = add([result_of_first_block, batch_normalization])
        return added

    @staticmethod
    def __pre_last_block(last_value):
        conv2d = Conv2D(filters=256, kernel_size=3, padding="same", strides=1)(last_value)
        up_sampling_2d = UpSampling2D(size=2)(conv2d)
        leaky_relu = LeakyReLU(alpha=0.2)(up_sampling_2d)
        last_value = leaky_relu
        return last_value

    def __first_block(self):
        conv2d = Conv2D(filters=64, kernel_size=9, padding="same", strides=1)(self.input_layer)
        prelu = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[1,2])(conv2d)
        return prelu

    def __last_block(self):
        conv2d = Conv2D(filters=3, kernel_size=9, padding="same", strides=1)(Generator.__pre_last_block(
            Generator.__pre_last_block(
                Generator.__first_last_block(
                    self.__first_block()
                )
            )
        ))
        return conv2d

    def define_model(self):
        conv2d = Generator.__last_block(self)
        output = Activation('tanh')(conv2d)
        model = Model(inputs=self.input_layer, outputs=output)
        return model
