from __future__ import division

import os
import time
import cPickle
import scipy.misc
from glob import glob
#import matplotlib.pyplot as plt

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
        elif config.dataset == 'lsun':
            with open('data/lsun_64/bedroom_train_valid.lst', 'r') as lstfile:
                self.data = ['data/lsun_64/bedroom_train/'+imgname for imgname in lstfile.read().split()]
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
            #batch_images = [get_image(batch_file, config.center_crop_size, is_crop=config.is_crop, resize_w=config.image_size, is_grayscale=config.is_grayscale) for batch_file in batch_files]
            batch_images = np.array([get_image_faster(batch_file) for batch_file in batch_files], dtype=np.float32)

        return batch_images  # , batch_labels

    def load_the_data(self, config, idx):
        batch_files = self.data[idx * config.batch_size:(idx + 1) * config.batch_size]
        return np.array([get_image_faster(batch_file) for batch_file in batch_files], dtype=np.float32)

    def load_one_data(self, config, idx):
        if idx<0:
            idx = np.random.randint(0, self.len)
        #image = get_image(self.data[idx], config.center_crop_size, is_crop=config.is_crop, resize_w=config.image_size, is_grayscale=config.is_grayscale)
        image = get_image_faster(self.data[idx])
        return image

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

        self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.logits_fake, tf.ones([config.batch_size,1])*config.smooth))
        #self.d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.logits_fake, tf.ones([config.batch_size,1])*(1.-config.smooth)))
        self.d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.logits_fake, tf.zeros([config.batch_size,1])))
        #self.d_loss_madefake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.logits_madefake, tf.zeros([config.batch_size, 1])))
        self.d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.logits_real, tf.ones([config.batch_size,1])*config.smooth))
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

        log_txt = open(config.result_dir+'log/'+config.dataset+'_'+config.dir_tag+'_log.txt', 'w')

        data = DataProvider(config)

        #sample_images = data.load_data(config, 0)
        sample_images = data.load_the_data(config, 0)
        sample_z1 = np.random.uniform(-1, 1, size=(config.batch_size, config.z_dim))
        sample_z2 = np.random.uniform(-1, 1, size=(config.batch_size, config.z_dim))
        sample_z3 = np.random.uniform(-1, 1, size=(config.batch_size, config.z_dim))

        save_size = int(math.sqrt(config.batch_size))
        save_images(sample_images[:save_size * save_size], [save_size, save_size], '{}/train_{:02d}_{:05d}.png'.format(config.sample_dir, 0, 0))

        if config.b_loadcheckpoint:
            if self.load(config):
                print(" [*] Load SUCCESS")
            else:
                print(" [!] Load failed...")
                return

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

                print("Epoch: [%2d] [%5d/%5d] time: %4.4f, loss: %.8f, prob_real: %.8f, prob_fake: %.8f" % (
                    epoch, idx, batch_idxs, time.time() - start_time, _loss, _prob_real, _prob_fake))
                log_txt.write("{:d} {:d} {:d} {:.8f} {:.8f} {:.8f}\n".format(epoch, idx, batch_idxs, _loss, _prob_real, _prob_fake))

                if np.mod(counter, 200) == 1:
                    save_size = int(math.sqrt(config.batch_size))
                    #z1
                    _generate_image, _loss, _prob_real, _prob_fake = self.sess.run([self.generate_image, self.total_loss, self.avg_prob_real, self.avg_prob_fake], feed_dict={self.z: sample_z1, self.images: sample_images})
                    save_images(_generate_image[:save_size * save_size], [save_size, save_size], '{}/train_{:02d}_{:05d}_z1.png'.format(config.sample_dir, epoch, idx))
                    print("[Sample] loss z1: %.8f, prob_real: %.8f, prob_fake: %.8f" % (_loss, _prob_real, _prob_fake))
                    log_txt.write("0 0 -1 {:.8f} {:.8f} {:.8f}\n".format(_loss, _prob_real, _prob_fake))
                    #z2
                    _generate_image, _loss, _prob_real, _prob_fake = self.sess.run([self.generate_image, self.total_loss, self.avg_prob_real, self.avg_prob_fake], feed_dict={self.z: sample_z2, self.images: sample_images})
                    save_images(_generate_image[:save_size * save_size], [save_size, save_size], '{}/train_{:02d}_{:05d}_z2.png'.format(config.sample_dir, epoch, idx))
                    print("[Sample] loss z2: %.8f, prob_real: %.8f, prob_fake: %.8f" % (_loss, _prob_real, _prob_fake))
                    log_txt.write("0 0 -2 {:.8f} {:.8f} {:.8f}\n".format(_loss, _prob_real, _prob_fake))
                    #z3
                    _generate_image, _loss, _prob_real, _prob_fake = self.sess.run([self.generate_image, self.total_loss, self.avg_prob_real, self.avg_prob_fake], feed_dict={self.z: sample_z3, self.images: sample_images})
                    save_images(_generate_image[:save_size * save_size], [save_size, save_size], '{}/train_{:02d}_{:05d}_z3.png'.format(config.sample_dir, epoch, idx))
                    print("[Sample] loss z3: %.8f, prob_real: %.8f, prob_fake: %.8f" % (_loss, _prob_real, _prob_fake))
                    log_txt.write("0 0 -3 {:.8f} {:.8f} {:.8f}\n".format(_loss, _prob_real, _prob_fake))
                    #save_images(_generate_image[:save_size * save_size], [save_size, save_size], '{}/train_{:02d}_{:04d}.png'.format(config.sample_dir, epoch, idx))
                    #print("[Sample] loss: %.8f, prob_real: %.8f, prob_fake: %.8f" % (_loss, _prob_real, _prob_fake))

                    raw_input("pause")

                if np.mod(counter, 1000) == 0:
                    self.save(config, counter)

                log_txt.flush()
                counter += 1

        log_txt.close()

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
        h0 = tf.nn.relu(batch_norm(h0, 'g_bn0'))
        
        # concat with Y
        h1 = tf.concat(3, [image_Y, h0])
        #print 'h0 shape after concat:', h0.get_shape()
        h1 = conv2d(h1, 128, k_h = 7, k_w = 7, d_h = 1, d_w = 1, name = 'g_h1_conv')
        h1 = tf.nn.relu(batch_norm(h1, 'g_bn1'))

        h2 = tf.concat(3, [image_Y, h1])
        h2 = conv2d(h2, 64, k_h = 5, k_w = 5, d_h = 1, d_w = 1, name = 'g_h2_conv')
        h2 = tf.nn.relu(batch_norm(h2, 'g_bn2'))
        
        h3 = tf.concat(3, [image_Y, h2])
        h3 = conv2d(h3, 64, k_h = 5, k_w = 5, d_h = 1, d_w = 1, name = 'g_h3_conv')
        h3 = tf.nn.relu(batch_norm(h3, 'g_bn3'))

        h4 = tf.concat(3, [image_Y, h3])
        h4 = conv2d(h4, 64, k_h = 5, k_w = 5,  d_h = 1, d_w = 1, name = 'g_h4_conv')
        h4 = tf.nn.relu(batch_norm(h4, 'g_bn4'))

        h5 = tf.concat(3, [image_Y, h4])
        h5 = conv2d(h5, 32, k_h = 5, k_w = 5,  d_h =1, d_w = 1, name = 'g_h5_conv')
        h5 = tf.nn.relu(batch_norm(h5, 'g_bn5'))
        
        h6 = tf.concat(3, [image_Y, h5])
        h6 = conv2d(h6, 2, k_h = 5, k_w = 5,  d_h = 1, d_w = 1, name = 'g_h6_conv')
        out = tf.nn.tanh(h6)

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

        this_checkpoint_dir = self.saver.save(self.sess, os.path.join(checkpoint_dir, model_name), global_step=step)
        print 'Saved checkpoint_dir:', this_checkpoint_dir

    def load(self, config=None):
        print(" [*] Reading checkpoints...")

        model_dir = "%s_%s_%s" % (config.dataset, config.batch_size, config.image_size)
        checkpoint_dir = os.path.join(config.checkpoint_dir, model_dir)
        print 'checkpoint_dir:', checkpoint_dir

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)  #get_checkpoint_state() returns CheckpointState Proto
        if ckpt and ckpt.model_checkpoint_path:
            latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
            print 'latest_checkpoint:', latest_checkpoint
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            #ckpt_name = 'DCGAN.model-35502'
            print 'ckpt_path:', os.path.join(checkpoint_dir, ckpt_name)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            
            return True
        else:
            return False

    def test_z(self, config=None):
        #plt.ion()
        data = DataProvider(config)
        save_size = int(math.sqrt(config.batch_size))

        '''#interactive version
        test_images = data.load_data(config, 0)
        show_size = int(math.sqrt(config.batch_size))
        show_images(test_images, [show_size, show_size], window_idx=0, close = False)
        select_idx = int(input('Choose an image(0-%d):'%config.batch_size))
        while(select_idx<0 or select_idx>=config.batch_size):
            select_idx = int(input('Wrong index!\nChoose an image(0-%d):'%config.batch_size))
        plt.close(0)
        test_image = test_images[select_idx]
        show_image(test_image, 'image_origin', window_idx=1, close = False)
        test_z_origin = np.random.uniform(-1, 1, size=(1, config.z_dim))
        '''

        test_image_idx = config.test_image_idx
        if test_image_idx<0:
            test_image_idx = np.random.randint(0, data.len)
        print 'Test image idx:', test_image_idx
        save_dir = '{}/{:06d}'.format(config.sample_dir, test_image_idx)
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        test_image = data.load_one_data(config, test_image_idx)
        save_image(test_image, '{}/test_{:06d}_origin.png'.format(save_dir, test_image_idx))
        test_image_batch = np.array([test_image for i in range(config.batch_size)])
        

        '''##test one z
        test_z_origin = np.array([-1.     , -1.     ,  1.     ,  1.     ,  1.     , -1.     ,
               -1.     ,  1.     , -1.     ,  1.     , -1.     ,  1.     ,
               -1.     ,  1.     , -1.     ,  1.     ,  1.     , -1.     ,
               -1.     ,  1.     ,  1.     ,  1.     ,  1.     , -0.53125,
               -1.     ,  1.     ,  0.90625,  1.     ,  1.     ,  0.6875 ,
                1.     ,  0.59375,  0.875  ,  1.     , -1.     ,  1.     ,
               -1.     , -0.90625, -1.     , -1.     , -1.     ,  1.     ,
                0.75   , -0.96875, -1.     , -1.     , -1.     , -1.     ,
               -0.6875 ,  1.     , -1.     ,  1.     ,  1.     ,  1.     ,
               -1.     , -1.     , -1.     ,  1.     , -1.     ,  1.     ,
                1.     ,  1.     ,  1.     ,  1.     , -0.375  , -1.     ,
                1.     ,  1.     ,  0.59375, -1.     , -1.     ,  1.     ,
                1.     , -1.     ,  1.     ,  1.     ,  0.375  , -1.     ,
               -0.75   , -1.     , -1.     , -1.     , -1.     , -1.     ,
               -0.5    ,  1.     , -1.     , -1.     ,  1.     , -1.     ,
              -1.     ,  1.     , -0.8125 ,  1.     ,  0.5    ,  0.5    ,
               1.     , -1.     , -1.     ,  1.     ], dtype=np.float32)
        test_z_batch = np.array([test_z_origin for i in range(config.batch_size)])
        generate_image, probs_real, probs_fake = self.sess.run([self.generate_image, self.probs_real, self.probs_fake], feed_dict={self.z: test_z_batch, self.images: test_image_batch})
        save_images(generate_image[:save_size * save_size], [save_size, save_size], '{}/test_{:06d}_fixZ.png'.format(config.sample_dir, test_image_idx))
        print 'Real:', probs_real
        print 'Fake:', probs_fake
            
        '''
        save_result_prob_real = []
        save_result_prob_fake = []

        test_z_origin = np.random.uniform(-1, 1, size=config.z_dim)
        for z_idx in range(config.z_dim):
            print 'Alter z[%d]\r'%z_idx,
            test_z_batch = np.array([test_z_origin for i in range(config.batch_size)])
            test_z_batch[:, z_idx] = np.arange(-1.,1.,2./config.batch_size, np.float32)

            generate_image, probs_real, probs_fake, avg_prob_real, avg_prob_fake = self.sess.run([self.generate_image, self.probs_real, self.probs_fake, self.avg_prob_real, self.avg_prob_fake], feed_dict={self.z: test_z_batch, self.images: test_image_batch})
            print "prob_real: %.8f, prob_fake: %.8f" % (avg_prob_real, avg_prob_fake)
            save_images(generate_image[:save_size * save_size], [save_size, save_size], '{}/test_{:06d}_{:04d}.png'.format(save_dir, test_image_idx, z_idx))

            #images_Y, images_U, images_V, generate_image, probs_real, probs_fake = self.sess.run([self.images_Y, self.images_U, self.images_V, self.generate_image, self.probs_real, self.probs_fake], feed_dict={self.z: test_z_batch, self.images: test_image_batch})
            #scipy.misc.imsave('{}/image_Y.png'.format(config.sample_dir), inverse_transform(image_Y[0].squeeze()))
            #scipy.misc.imsave('{}/image_U.png'.format(config.sample_dir), inverse_transform(image_U[0].squeeze()))
            #scipy.misc.imsave('{}/image_V.png'.format(config.sample_dir), inverse_transform(image_V[0].squeeze()))

            save_result_prob_real.append(probs_real)
            save_result_prob_fake.append(probs_fake)
        
        print 'Test done.'

        with open('{}/test_{:06d}.pkl'.format(save_dir, test_image_idx), 'w') as outfile:
            cPickle.dump((test_image, test_z_origin, save_result_prob_real, save_result_prob_fake), outfile)
        print 'Save done.'
        #'''

    def test_random_z(self, config=None):
        data = DataProvider(config)
        save_size = int(math.sqrt(config.batch_size))
        test_image_idxs = []
        test_images = []
        test_z_batch = np.random.uniform(-1, 1, size=(config.batch_size, config.z_dim))
        save_result_prob_real = []
        save_result_prob_fake = []

        print "Randomly choose %d images"%(config.number_of_test_images)
        test_idxs = range(data.len)
        np.random.shuffle(test_idxs)
        for test_idx, test_image_idx in enumerate(test_idxs[:config.number_of_test_images]):
            print 'Testing image {}, index {} ...'.format(test_idx, test_image_idx)
            test_image = data.load_one_data(config, test_image_idx)
            test_image_idxs.append(test_image_idx)
            test_images.append(test_image[0].squeeze())
            save_image(test_image, '{}/test_random_{:06d}_origin.png'.format(config.sample_dir, test_image_idx))
            test_image_batch = np.array([test_image for i in range(config.batch_size)])

            test_the_image_batch = data.load_the_data(config, 0)

            generate_image, probs_real, probs_fake, avg_prob_real, avg_prob_fake = self.sess.run([self.generate_image, self.probs_real, self.probs_fake, self.avg_prob_real, self.avg_prob_fake], feed_dict={self.z: test_z_batch, self.images: test_the_image_batch})
            print "prob_real: %.8f, prob_fake: %.8f" % (avg_prob_real, avg_prob_fake)
            save_images(generate_image[:save_size * save_size], [save_size, save_size], '{}/test_random_{:06d}.png'.format(config.sample_dir, test_image_idx))

            save_result_prob_real.append(probs_real)
            save_result_prob_fake.append(probs_fake)
        
        print 'Test done.'

        with open('{}/test_random.pkl'.format(config.sample_dir), 'w') as outfile:
            cPickle.dump((test_image_idxs, test_images, test_z_batch, save_result_prob_real, save_result_prob_fake), outfile)
        print 'Save done.'