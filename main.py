import os
import numpy as np

from model import DCGAN
from utils import pp, visualize, to_json

import tensorflow as tf

tf.app.flags.DEFINE_string("devices", "gpu:0", "Which gpu to be used")

tf.app.flags.DEFINE_string("dataset", "celebA", "The name of dataset [celebA, mnist, lsun]")
tf.app.flags.DEFINE_integer("image_size", 64, "The size of the output images to produce [64]")
tf.app.flags.DEFINE_integer("center_crop_size", 108, "The width of the images presented to the model, 0 for auto")
tf.app.flags.DEFINE_boolean("is_crop", True, "True for training, False for testing [False]")
tf.app.flags.DEFINE_integer("train_size", np.inf, "The size of train images [np.inf]")

tf.app.flags.DEFINE_string("dir_tag", "z100_d1_g1_new", "dir_tag for sample_dir and checkpoint_dir")
tf.app.flags.DEFINE_string("result_dir", "./result/", "Where to save the checkpoint and sample")
tf.app.flags.DEFINE_boolean("is_train", False, "True for training, False for testing [False]")
tf.app.flags.DEFINE_boolean("visualize", False, "True for visualizing, False for nothing [False]")
tf.app.flags.DEFINE_boolean("b_loadcheckpoint", False, "b_loadcheckpoint")

tf.app.flags.DEFINE_integer("epoch", 20, "Epoch to train [25]")
tf.app.flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
tf.app.flags.DEFINE_float("d_learning_rate", 0.0002, "Learning rate of for adam")
tf.app.flags.DEFINE_float("g_learning_rate", 0.0002, "Learning rate of for adam")

tf.app.flags.DEFINE_integer("batch_size", 64, "The size of batch images")
tf.app.flags.DEFINE_integer("gf_dim", 128, "gf_dim")
tf.app.flags.DEFINE_integer("df_dim", 64, "df_dim")
tf.app.flags.DEFINE_integer("dfc_dim", 1024, "df_dim")
tf.app.flags.DEFINE_integer("gfc_dim", 1024, "df_dim")
tf.app.flags.DEFINE_integer("z_dim", 100, "z_dim")
tf.app.flags.DEFINE_integer("c_dim", 3, "c_dim")

tf.app.flags.DEFINE_integer("K_for_Dtrain", 1, "K_for_Dtrain")
tf.app.flags.DEFINE_integer("K_for_Gtrain", 1, "K_for_Gtrain") # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)

tf.app.flags.DEFINE_integer("num_classes", 1, "num_classes")
tf.app.flags.DEFINE_float("class_loss_weight", 0.0, "The weight of class loss, in discriminator")
tf.app.flags.DEFINE_float("d_label_smooth", 0.0, "The amount label smooth")
tf.app.flags.DEFINE_float("generator_target_prob", 1.0, "The generator target prob")
tf.app.flags.DEFINE_float("d_noise_image_mean", 0.0, "d_noise_image_mean")
tf.app.flags.DEFINE_float("d_noise_image_var", 0.1, "d_noise_image_var")
tf.app.flags.DEFINE_float("out_init_b", -0.45, "out_init_b")
tf.app.flags.DEFINE_float("out_stddev", 0.075, "out_stddev")
tf.app.flags.DEFINE_boolean("use_vbn", False, "True for use_vbn")
tf.app.flags.DEFINE_boolean("minibacth", False, "True for minibacth")
tf.app.flags.DEFINE_boolean("add_hz", False, "True for add random z in each hidden layer, in generator")

tf.app.flags.DEFINE_integer("test_image_idx", -1, "test_image_idx")
tf.app.flags.DEFINE_boolean("random_z", True, "test random z")
tf.app.flags.DEFINE_integer("number_of_test_images", 64, "number_of_test_images")
tf.app.flags.DEFINE_float("smooth", 0.8, "smooth")

FLAGS = tf.app.flags.FLAGS

def main(_):

    pp.pprint(FLAGS.__flags)

    FLAGS.is_grayscale = (FLAGS.c_dim == 1)
    FLAGS.sample_dir = FLAGS.result_dir + 'samples/' + FLAGS.dataset + '_' + FLAGS.dir_tag
    FLAGS.checkpoint_dir = FLAGS.result_dir + 'checkpoint/' + FLAGS.dataset + '_' + FLAGS.dir_tag

    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)
    if not os.path.exists(FLAGS.sample_dir):
        os.makedirs(FLAGS.sample_dir)

    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:

        if FLAGS.dataset == 'mnist':
            FLAGS.image_size = 32
            FLAGS.c_dim = 1

        with tf.device(FLAGS.devices):

            dcgan = DCGAN(sess, config=FLAGS)

            if FLAGS.is_train:
                dcgan.train(FLAGS)
            else:
                if dcgan.load(FLAGS):
                    print " [*] Load SUCCESS"
                    if FLAGS.random_z:
                        print " [*] Test RANDOM Z"
                        dcgan.test_fix(FLAGS)
                    else:
                        print " [*] Test Z"
                        dcgan.test_z(FLAGS)
                else:
                    print " [!] Load failed..."

        #if FLAGS.visualize:
        #    OPTION = 2
        #   visualize(sess, dcgan, FLAGS, OPTION)

if __name__ == '__main__':
    tf.app.run()
