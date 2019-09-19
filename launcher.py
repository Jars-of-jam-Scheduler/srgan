import numpy as np
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Input, Model

from Scaler import Scaler
from util import Util
from vgg19_loss import Vgg19Loss

from discriminator import Discriminator
from generator import Generator

from my_plotter import MyPlotter

np.random.seed(10)


# <!--- THE OPTIMIZER (COMMON TO ALL THE MODELS) --->
the_optimizer = Adam(1E-4, 0.9, 0.999, 1e-08)
# <!--- /THE OPTIMIZER (COMMON TO ALL THE MODELS) --->


# <!--- COST FUNCTION --->
def build_vgg19_loss_network(ground_truth_image, predicted_image):
    the_loss = Vgg19Loss.define_the_loss(Scaler.hr_images_dimensions, ground_truth_image, predicted_image)
    return the_loss
# <!--- /COST FUNCTION --->


# <!--- MAIN MODELS -->
def build_generator():
    shape_width = Scaler.hr_images_dimensions[0] // Scaler.down_scale_factor
    shape_height = Scaler.hr_images_dimensions[1] // Scaler.down_scale_factor
    shape_dimensions = Scaler.hr_images_dimensions[2]
    generator_shape = (shape_width, shape_height, shape_dimensions)
    generator_model = Generator(generator_shape)
    generator_model = generator_model.define_model()
    generator_model.compile(optimizer=the_optimizer, loss=build_vgg19_loss_network)
    return generator_model


def build_discriminator():
    shape_width = Scaler.hr_images_dimensions[0]
    shape_height = Scaler.hr_images_dimensions[1]
    shape_dimensions = Scaler.hr_images_dimensions[2]
    discriminator_shape = (shape_width, shape_height, shape_dimensions)
    discriminator_model = Discriminator(discriminator_shape)
    discriminator_model = discriminator_model.define_model()
    discriminator_model.compile(optimizer=the_optimizer, loss='binary_crossentropy')
    return discriminator_model


def build_adversarial_network(discriminator_model, generator_model):
    discriminator_model.trainable = False

    shape_width_low_resolution = Scaler.hr_images_dimensions[0] // Scaler.down_scale_factor
    shape_height_low_resolution = Scaler.hr_images_dimensions[1] // Scaler.down_scale_factor
    shape_dimensions_low_resolution = Scaler.hr_images_dimensions[2]
    adversarial_network_shape = (shape_width_low_resolution, shape_height_low_resolution, shape_dimensions_low_resolution)
    gan_input = Input(adversarial_network_shape)
    x = generator_model(gan_input)
    gan_output = discriminator_model(x)
    gan_model = Model(gan_input, [x, gan_output])
    gan_model.compile   (
                            loss=[build_vgg19_loss_network, "binary_crossentropy"],
                            loss_weights=[1., 1e-3],
                            optimizer=the_optimizer
                        )
    return gan_model
# <!--- /MAIN MODELS -->


# <!--- TRAININGS -->
def train(number_of_epochs, batch_size):
    discriminator_model = build_discriminator()
    generator_model = build_generator()
    gan_model = build_adversarial_network(discriminator_model, generator_model)

    print("Loading data...")
    x_resized_original_hr = Util.fetch_then_resize_high_resolution_images()
    x_train_original_hr = x_resized_original_hr[:1]
    x_train_hr = Util.fetch_training_set_resized_high_resolution_images(x_train_original_hr)
    x_train_lr = Util.fetch_training_set_resized_low_resolution_images(x_train_original_hr)
    normalized_x_train_hr = Util.normalize(x_train_hr)
    normalized_x_train_lr = Util.normalize(x_train_lr)
    print("Data is now loaded.")

    my_plotter = MyPlotter()

    batch_count = int(normalized_x_train_hr.shape[0] / batch_size)
    print("Batch size = " + str(batch_size))
    print("Number of training images = " + str(normalized_x_train_hr.shape[0]))
    print("Batches count = " + str(batch_count))
    for epoch in range(0, number_of_epochs):
        print('-' * 15, 'Epoch %d' % epoch, '-' * 15)
        for _ in range(batch_count):
            train_discriminator(batch_size, discriminator_model, generator_model, normalized_x_train_hr, normalized_x_train_lr)
            train_gan(batch_size, gan_model, discriminator_model, normalized_x_train_hr, normalized_x_train_lr)

        if epoch == 1 or epoch % 5 == 0:
            my_plotter.plot_images_predicted_for_testing(epoch, generator_model, x_resized_original_hr)
            generator_model.save(
                '/content/drive/My Drive/Informatique/Projets_Informatiques/Projets_Python/srgan/output/srgan.h5',
                include_optimizer=False)

        if epoch == number_of_epochs - 1:
            my_plotter.plot_images_predicted_for_testing(epoch, generator_model, x_resized_original_hr)
            generator_model.save(
                '/content/drive/My Drive/Informatique/Projets_Informatiques/Projets_Python/srgan/output/srgan.h5', include_optimizer=False)

def train_discriminator(batch_size, discriminator_model, generator_model, x_train_hr, x_train_lr):
    batch_images_hr_and_lr_tuple = Util.get_random_batch_images_hr_and_lr(batch_size, x_train_hr, x_train_lr)
    batch_image_hr = batch_images_hr_and_lr_tuple[0]
    batch_image_lr = batch_images_hr_and_lr_tuple[1]
    generated_hr_images = generator_model.predict(batch_image_lr)
    real_data_y = np.ones(batch_size) - np.random.random_sample(batch_size) * 0.2
    fake_data_y = np.random.random_sample(batch_size) * 0.2
    discriminator_model.trainable = True
    d_loss_real = discriminator_model.train_on_batch(batch_image_hr, real_data_y)
    d_loss_fake = discriminator_model.train_on_batch(generated_hr_images, fake_data_y)
    return d_loss_real, d_loss_fake

def train_gan(batch_size, gan_model, discriminator_model, x_train_hr, x_train_lr):
    batch_images_hr_and_lr_tuple = Util.get_random_batch_images_hr_and_lr(batch_size, x_train_hr, x_train_lr)
    batch_image_hr = batch_images_hr_and_lr_tuple[0]
    batch_image_lr = batch_images_hr_and_lr_tuple[1]
    gan_y = np.ones(batch_size) - np.random.random_sample(batch_size) * 0.2
    discriminator_model.trainable = False
    loss_gan = gan_model.train_on_batch(batch_image_lr, [batch_image_hr, gan_y])
    return loss_gan
# <!--- /TRAININGS -->


# <!--- ENTRY POINT -->
train(10000, 1)
# <!--- /ENTRY POINT -->
