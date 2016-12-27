from __future__ import division

import os
import time
import cPickle
import scipy.misc
from glob import glob

from ops import *
from utils import *

class DataProvider(object):
    def __init__(self, config):

        if config.dataset == 'mnist':
            self.data_X, self.data_y = self.load_mnist()
            self.len = len(self.data_X)
        elif config.dataset == 'imagenet':
            self.data = glob(os.path.join("./data/imagenet/ILSVRC2012/ILSVRC2012_img_train_t3/n*/*.JPEG"))
            self.len = len(self.data)
            print 'data len:', self.len
        elif config.dataset == 'celebA':
            self.data = glob(os.path.join("./data/celebA/img_align_celeba/*.jpg"))
            self.len = len(self.data)
            print 'data len:', self.len
        else:
            self.data = glob(os.path.join("./data/", config.dataset, "*.jpg"))
            self.len = len(self.data)
            print 'data len:', self.len

    def load_data(self, config, idx):

        if idx == 0:
            if config.dataset == 'mnist':
                seed = np.random.randint(10000)
                np.random.seed(seed)
                np.random.shuffle(self.data_X)
                np.random.seed(seed)
                np.random.shuffle(self.data_y)
            else:
                np.random.shuffle(self.data)

        if config.dataset == 'mnist':
            batch_images = self.data_X[idx * config.batch_size:(idx + 1) * config.batch_size]
            batch_labels = self.data_y[idx * config.batch_size:(idx + 1) * config.batch_size]
        else:
            batch_files = self.data[idx * config.batch_size:(idx + 1) * config.batch_size]
            batch_images = [get_image(batch_file, config.center_crop_size, is_crop=config.is_crop, resize_w=config.image_size, is_grayscale=config.is_grayscale) for batch_file in
                            batch_files]

            for dir_idx, dir_file in enumerate(batch_files):
                img = scipy.misc.imread(dir_file)
                scipy.misc.imsave('/home/yuncao/Documents/result/test/images_aftercvt/%d.png'%dir_idx, img)
            with open('/home/yuncao/Documents/result/test/sample.pkl','w') as outfile:
                cPickle.dump((batch_files, batch_images),outfile)
            
            if (config.is_grayscale):
                batch_images = np.array(batch_images).astype(np.float32)[:, :, :, None]
            else:
                batch_images = np.array(batch_images).astype(np.float32)
            #batch_labels = np.zeros([config.batch_size, config.y_dim])

        return batch_images  # , batch_labels

    def load_mnist(self, loadTest = False, useX32 = True):

        data_dir = './data/mnist/'  # os.path.join("./data", config.dataset)

        fd = open(os.path.join(data_dir, 'train-images-idx3-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        trX = loaded[16:].reshape((60000, 28, 28, 1)).astype(np.float)

        fd = open(os.path.join(data_dir, 'train-labels-idx1-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        trY = loaded[8:].reshape((60000)).astype(np.float)

        if loadTest:
            fd = open(os.path.join(data_dir, 't10k-images-idx3-ubyte'))
            loaded = np.fromfile(file=fd, dtype=np.uint8)
            teX = loaded[16:].reshape((10000, 28, 28, 1)).astype(np.float)

            fd = open(os.path.join(data_dir, 't10k-labels-idx1-ubyte'))
            loaded = np.fromfile(file=fd, dtype=np.uint8)
            teY = loaded[8:].reshape((10000)).astype(np.float)

            trY = np.asarray(trY)
            teY = np.asarray(teY)

            X = np.concatenate((trX, teX), axis=0)
            Y = np.concatenate((trY, teY), axis=0)
        else:
            X = trX
            Y = trY

        seed = 547
        np.random.seed(seed)
        np.random.shuffle(X)
        np.random.seed(seed)
        np.random.shuffle(Y)

        y_vec = np.zeros((len(Y), 10), dtype=np.float)
        for i, label in enumerate(Y):
            y_vec[i, int(Y[i])] = 1.0

        if useX32:
            X32 = np.zeros([len(X), 32, 32, 1])
            X32[:,2:30,2:30] = X
            X = X32

        return X / 127.5 - 1.0, y_vec

    def read_and_decode(self, filename_queue):

        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
        features = tf.parse_single_example(serialized_example, features={'image_raw': tf.FixedLenFeature([], tf.string), })

        image = tf.decode_raw(features['image_raw'], tf.uint8)
        image.set_shape(128 * 128 * 3)
        image = tf.reshape(image, [128, 128, 3])

        image = tf.cast(image, tf.float32) * (2. / 255) - 1.

        return image

    def read_and_decode_with_labels(self, filename_queue):

        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
        features = tf.parse_single_example(serialized_example, features={'image_raw': tf.FixedLenFeature([], tf.string), 'label': tf.FixedLenFeature([], tf.int64)})

        image = tf.decode_raw(features['image_raw'], tf.uint8)
        image.set_shape(128 * 128 * 3)
        image = tf.reshape(image, [128, 128, 3])

        image = tf.cast(image, tf.float32) * (2. / 255) - 1.

        label = tf.cast(features['label'], tf.int32)

        return image, label


class DCGAN(object):
    def __init__(self, sess, config=None):

        self.sess = sess
        self.config = config

        """
        self.d_bn1 = batch_norm1(name='d_bn1')
        self.d_bn2 = batch_norm1(name='d_bn2')
        self.d_bn3 = batch_norm1(name='d_bn3')
        self.g_bn0 = batch_norm1(name='g_bn0')
        self.g_bn1 = batch_norm1(name='g_bn1')
        self.g_bn2 = batch_norm1(name='g_bn2')
        self.g_bn3 = batch_norm1(name='g_bn3')
        """
        self.build_model(config)

    def build_model(self, config):

        self.z = tf.placeholder(tf.float32, [config.batch_size, config.z_dim], name='z')
        self.images = tf.placeholder(tf.float32, [config.batch_size] + [config.image_size, config.image_size, config.c_dim], name='real_images')
        #self.images_Y = tf.slice(self.images, [0,0,0,0], [-1, -1,-1,1])
        self.images_Y, self.images_U, self.images_V = tf.split(3, 3, self.images)
        print 'Y shape after split', self.images_Y.get_shape()

        #self.generate_image = self.generator(self.z, config=config) #old
        self.generate_image_UV = self.generator_colorization(self.z, self.images_Y, config=config) #check if direct slice correct
        self.generate_image = tf.concat(3, [self.images_Y, self.generate_image_UV])
        #self.round_Y = tf.concat(3, [self.images_Y[1:], self.images_Y[0]])
        #self.madefake_image = tf.concat(3, [self.images_Y, self.generate_image_UV])

        self.probs_real, self.logits_real = self.discriminator(self.images, config=config)
        self.probs_fake, self.logits_fake = self.discriminator(self.generate_image, reuse=True, config=config)
        #self.probs_madefake, self.logits_madefake = self.discriminator(self.madefake_image, reuse=True, config=config)

        self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.logits_fake, tf.ones([config.batch_size,1])))
        self.d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.logits_fake, tf.zeros([config.batch_size,1])))
        #self.d_loss_madefake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.logits_madefake, tf.zeros([config.batch_size, 1])))
        self.d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.logits_real, tf.ones([config.batch_size,1])))
        self.d_loss = self.d_loss_real + self.d_loss_fake

        self.total_loss = self.d_loss + self.g_loss
        self.avg_prob_real = tf.reduce_mean(self.probs_real)
        self.avg_prob_fake = tf.reduce_mean(self.probs_fake)

        # tensorfolw summary:

        tf.image_summary("generate_image", self.generate_image, 1000)

        tf.histogram_summary("prob_real", self.probs_real)
        tf.histogram_summary("prob_fake", self.probs_fake)

        tf.scalar_summary("avg_probs_fake", self.avg_prob_fake)
        tf.scalar_summary("avg_probs_real", self.avg_prob_real)
        tf.scalar_summary("d_loss_real", self.d_loss_real)
        tf.scalar_summary("d_loss_fake", self.d_loss_fake)

        tf.scalar_summary("d_loss", self.d_loss)
        tf.scalar_summary("g_loss", self.g_loss)
        tf.scalar_summary("total_loss", self.total_loss)

        self.merged_summary = tf.merge_all_summaries()

        t_vars = tf.trainable_variables()

        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]

        self.saver = tf.train.Saver()

    def train(self, config):

        d_optim = tf.train.AdamOptimizer(config.d_learning_rate, beta1=config.beta1).minimize(self.d_loss, var_list=self.d_vars)
        g_optim = tf.train.AdamOptimizer(config.g_learning_rate, beta1=config.beta1).minimize(self.g_loss, var_list=self.g_vars)

        #tf.global_variables_initializer().run()
        tf.initialize_all_variables().run()
        self.writer = tf.train.SummaryWriter(config.result_dir + 'log/' + config.dataset + '_' + config.dir_tag, self.sess.graph)

        data = DataProvider(config)

        sample_images = data.load_data(config, 0)
        sample_z = np.random.uniform(-1, 1, size=(config.batch_size, config.z_dim))

        save_size = int(math.sqrt(config.batch_size))
        save_images(sample_images[:save_size * save_size], [save_size, save_size], '{}/train_{:02d}_{:04d}.png'.format(config.sample_dir, 0, 0))

        raw_input('pause')

        if config.b_loadcheckpoint:
            if self.load(config):
                print(" [*] Load SUCCESS")
            else:
                print(" [!] Load failed...")

        counter = 1
        start_time = time.time()

        for epoch in xrange(config.epoch):

            batch_idxs = min(data.len, config.train_size) // config.batch_size
            for idx in xrange(0, batch_idxs):

                batch_images = data.load_data(config, idx)
                batch_z = np.random.uniform(-1, 1, [config.batch_size, config.z_dim]).astype(np.float32)

                # Update D network
                for k_d in xrange(0, config.K_for_Dtrain):
                    _, summary_str, _loss, _prob_real, _prob_fake = self.sess.run([d_optim, self.merged_summary, self.total_loss, self.avg_prob_real, self.avg_prob_fake], feed_dict={self.z: batch_z, self.images: batch_images})
                    self.writer.add_summary(summary_str, counter)
                    # when running d_optim, basically the whole graph will be executed, so, we get summary and log info from here.

                # Update G network
                for k_g in xrange(0, config.K_for_Gtrain):
                    self.sess.run([g_optim], feed_dict={self.z: batch_z, self.images: batch_images})

                counter += 1

                print("Epoch: [%2d] [%4d/%4d] time: %4.4f, loss: %.8f, prob_real: %.8f, prob_fake: %.8f" % (
                    epoch, idx, batch_idxs, time.time() - start_time, _loss, _prob_real, _prob_fake))

                if np.mod(counter, 100) == 1:
                    _generate_image, _loss, _prob_real, _prob_fake = self.sess.run([self.generate_image, self.total_loss, self.avg_prob_real, self.avg_prob_fake], feed_dict={self.z: sample_z, self.images: sample_images})

                    save_size = int(math.sqrt(config.batch_size))
                    save_images(_generate_image[:save_size * save_size], [save_size, save_size], '{}/train_{:02d}_{:04d}.png'.format(config.sample_dir, epoch, idx))
                    print("[Sample] loss: %.8f, prob_real: %.8f, prob_fake: %.8f" % (_loss, _prob_real, _prob_fake))

                if np.mod(counter, 500) == 2:
                    self.save(config, counter)

    def discriminator(self, image, y=None, reuse=False, config=None):
        if reuse:
            tf.get_variable_scope().reuse_variables()

        h0 = lrelu(conv2d(image, config.df_dim, name='d_h0_conv'))
        h1 = lrelu(batch_norm(conv2d(h0, config.df_dim * 2, name='d_h1_conv'), 'd_bn1'))
        h2 = lrelu(batch_norm(conv2d(h1, config.df_dim * 4, name='d_h2_conv'), 'd_bn2'))
        h3 = lrelu(batch_norm(conv2d(h2, config.df_dim * 8, name='d_h3_conv'), 'd_bn3'))
        h4 = linear(tf.reshape(h3, [config.batch_size, -1]), 1, 'd_h3_lin')

        return tf.nn.sigmoid(h4), h4

    def generator(self, z, config=None):

        print('Generator shape')
        print('z : ', z.get_shape())

        curfdim = config.gf_dim #128
        cursize = config.image_size // 2 #32
        while (cursize % 2 == 0 and cursize >= 8):
            cursize = cursize // 2
            curfdim *= 2

        #1024*4*4
        zr = linear(z, curfdim * cursize * cursize, 'g_h0_lin', with_w=False)
        h0 = tf.reshape(zr, [-1, cursize, cursize, curfdim])
        h0 = tf.nn.relu(batch_norm(h0, "g_vbn_0"))
        print('h0: ', h0.get_shape())

        idx = 1
        while config.image_size // 2 != cursize:

            idx += 1
            cursize *= 2
            curfdim //= 2

            h0 = deconv2d(h0, [config.batch_size, cursize, cursize, curfdim], name='g_h' + str(idx), with_w=False)
            h0 = tf.nn.relu(batch_norm(h0, "g_vbn_" + str(idx)))
            print('h' + str(idx) + ': ', h0.get_shape())

        h0 = deconv2d(h0, [config.batch_size, config.image_size, config.image_size, config.c_dim], name='g_h' + str(idx+1), with_w=False)
        out = tf.nn.tanh(h0)
        print('out:', out.get_shape())

        return out

    def generator_colorization(self, z, image_Y, config=None):

        print 'Init Colorization Generator with z:', z.get_shape()
        s = config.image_size #64

        # project z
        h0 = linear(z, s * s, 'g_h0_lin', with_w=False)
        # reshape 
        h0 = tf.reshape(h0, [-1, s, s, 1])
        # concat with Y
        h0 = tf.concat(3, [image_Y, h0])
        print 'h0 shape after concat:', h0.get_shape()
        h0 = tf.nn.relu(batch_norm(h0, 'g_bn0'))

        h1 = conv2d(h0, 32, d_h = 1, d_w = 1, name = 'g_h1_conv')
        h1 = tf.nn.relu(batch_norm(h1, 'g_bn1'))

        h2 = conv2d(h1, 64, d_h = 1, d_w = 1, name = 'g_h2_conv')
        h2 = tf.nn.relu(batch_norm(h2, 'g_bn2'))

        h3 = conv2d(h2, 64, d_h = 1, d_w = 1, name = 'g_h3_conv')
        h3 = tf.nn.relu(batch_norm(h3, 'g_bn3'))

        h4 = conv2d(h3, 32, d_h = 1, d_w = 1, name = 'g_h4_conv')
        h4 = tf.nn.relu(batch_norm(h4, 'g_bn4'))

        h5 = conv2d(h4, 2, d_h = 1, d_w = 1, name = 'g_h5_conv')
        out = tf.nn.tanh(h5)
        print 'generator out shape:', out.get_shape()

        return out

    def generator2(self, z, config=None):
        s = config.image_size
        s2, s4, s8, s16 = int(s / 2), int(s / 4), int(s / 8), int(s / 16)

        # project `z` and reshape
        zr, self.h0_w, self.h0_b = linear(z, config.gf_dim * 8 * s16 * s16, 'g_h0_lin', with_w=True)

        h0 = tf.reshape(self.zr, [-1, s16, s16, config.gf_dim * 8])
        h0 = tf.nn.relu(batch_norm(h0,'g_bn0'))

        h1, self.h1_w, self.h1_b = deconv2d(h0, [config.batch_size, s8, s8, config.gf_dim * 4], name='g_h1', with_w=True)
        h1 = tf.nn.relu(batch_norm(h1,'g_bn1'))

        h2, self.h2_w, self.h2_b = deconv2d(h1, [config.batch_size, s4, s4, config.gf_dim * 2], name='g_h2', with_w=True)
        h2 = tf.nn.relu(batch_norm(h2,'g_bn2'))

        h3, self.h3_w, self.h3_b = deconv2d(h2, [config.batch_size, s2, s2, config.gf_dim * 1], name='g_h3', with_w=True)
        h3 = tf.nn.relu(batch_norm(h3,'g_bn3'))

        h4, self.h4_w, self.h4_b = deconv2d(h3, [config.batch_size, s, s, config.c_dim], name='g_h4', with_w=True)

        return tf.nn.tanh(h4)

    def save(self, config=None, step=0):
        model_name = "DCGAN.model"
        model_dir = "%s_%s_%s" % (config.dataset, config.batch_size, config.image_size)
        checkpoint_dir = os.path.join(config.checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess, os.path.join(checkpoint_dir, model_name), global_step=step)

    def load(self, config=None):
        print(" [*] Reading checkpoints...")

        model_dir = "%s_%s_%s" % (config.dataset, config.batch_size, config.image_size)
        checkpoint_dir = os.path.join(config.checkpoint_dir, model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False
