import scipy.misc
import numpy as np
import time
import sys

resize_w = 64
read_dir = 'data/lsun/bedroom_train/'
save_dir = 'data/lsun_%d/bedroom_train/'%resize_w

with open('data/lsun/bedroom_train.lst', 'r') as lstfile:
    image_list = lstfile.read().split()
print 'data len:', len(image_list)

stime = time.time()
for count, image_name in enumerate(image_list):
    image_RGB = scipy.misc.imread(read_dir+image_name)
    h, w = image_RGB.shape[:2]
    crop_size = min(h, w)
    j = int(round((h - crop_size)/2.))
    i = int(round((w - crop_size)/2.))
    image_RGB = scipy.misc.imresize(x[j:j+crop_size, i:i+crop_size],
                               [resize_w, resize_w])
    scipy.misc.imsave(save_dir+image_name, image_RGB)
    if count%100 == 0: 
        print '{:06d}, {:.2f}s\r'.format(count, time.time()-stime), 
        sys.stdout.flush()
        raw_input('pause')
print 'Done in %.2f'%time.time()-stime