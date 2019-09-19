import numpy as np
from util import Util
import matplotlib.pyplot as plt
plt.switch_backend('agg')

class MyPlotter:
    def __init__(self):
        pass

    def plot_images_predicted_for_testing(self, epoch, generator, x_resized_original_hr, examples=3, dim=(1, 3),
                                          figure_size=(15, 5)):
        x_test_original_hr = x_resized_original_hr[:1]
        x_test_hr = Util.fetch_testing_set_resized_high_resolution_images(x_test_original_hr)
        x_test_lr = Util.fetch_testing_set_resized_low_resolution_images(x_test_original_hr)
        normalized_x_test_lr = Util.normalize(x_test_lr)

        rand_nums = np.random.randint(0, x_test_hr.shape[0], size=examples)
        test_normalized_image_batch_lr = normalized_x_test_lr[rand_nums]
        gen_img = generator.predict(test_normalized_image_batch_lr)

        denormalized_test_image_batch_lr = Util.denormalize(test_normalized_image_batch_lr)
        test_image_batch_hr = x_test_hr[rand_nums]
        denormalized_generated_image = Util.denormalize(gen_img)

        figure = plt.figure(None, figure_size)

        plt.subplot(dim[0], dim[1], 1)
        plt.imshow(denormalized_test_image_batch_lr[1], interpolation='nearest')
        plt.axis('off')

        plt.subplot(dim[0], dim[1], 2)
        plt.imshow(denormalized_generated_image[1], interpolation='nearest')
        plt.axis('off')

        plt.subplot(dim[0], dim[1], 3)
        plt.imshow(test_image_batch_hr[1], interpolation='nearest')
        plt.axis('off')

        plt.tight_layout()
        plt.savefig(
            '/content/drive/My Drive/Informatique/Projets_Informatiques/Projets_Python/srgan/output/generated_image_epoch_%d.png' % epoch)
        plt.close(figure)