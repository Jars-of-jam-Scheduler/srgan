from tensorflow.keras import Model
from tensorflow.keras.applications import VGG19
from tensorflow.keras.backend import mean, square


class Vgg19Loss:
    def __init__(self):
        pass

    @staticmethod
    def define_the_loss(high_resolution_shape, ground_truth_image, predicted_image):
        model_vgg19 = VGG19(include_top=False, weights='imagenet', input_shape=high_resolution_shape)
        model_vgg19.trainable = False
        for l in model_vgg19.layers:
            l.trainable = False
        loss_model = Model(inputs=model_vgg19.input, outputs=model_vgg19.get_layer('block5_conv4').output)
        loss_model.trainable = False
        return mean(square(loss_model(ground_truth_image) - loss_model(predicted_image)))
