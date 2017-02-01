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

outfile = open('data/lsun_%d/bedroom_train_valid.lst'%resize_w,'w')

valid_count = 0
stime = time.time()
for count, image_name in enumerate(image_list):
    try:
        image_RGB = scipy.misc.imread(read_dir+image_name)
    except Exception, e:
        print Exception,":",e
        print 'BAD image:',image_name
        continue
    if len(image_RGB.shape)==3 and image_RGB.shape[3]==3:
        valid_count+=1
    else:
        print 'BAD shape:',image_name
        continue
    h, w = image_RGB.shape[:2]
    crop_size = min(h, w)
    j = int(round((h - crop_size)/2.))
    i = int(round((w - crop_size)/2.))
    image_RGB_resize = scipy.misc.imresize(image_RGB[j:j+crop_size, i:i+crop_size],
                               [resize_w, resize_w])
    scipy.misc.imsave(save_dir+image_name, image_RGB_resize)
    if count%100 == 0: 
        print '{:06d}, {:.2f}s\r'.format(count, time.time()-stime), 
        sys.stdout.flush()
        raw_input('pause')
print 'Done in %.2f'%time.time()-stime
print 'Valid %d/%d'%(valid_count, len(image_list))
outfile.close()