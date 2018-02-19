
from PIL import Image
import numpy as np
from scipy import misc
import random
import os
import shutil


#  im = Image.open('./datasets/images/0a0e.jpg')


#  im = np.fromfile('./datasets/images/0a0e.jpg')
#  print(im.shape)

im = misc.imread('./datasets/images/0a0e.jpg', flatten=False)
print(im.shape)


#  char_set = ['a','b','c','d','e']
#
#  label = 'adea'
#  ans = list(map(char_set.index, label))
#  print(ans)
#
#  ans = [char_set.index(e) for e in label]
#  print(ans)


image_files = os.listdir('./datasets/images')

train_ratio = 0.8
test_ratio = 0.2
num = len(image_files)

num_train = int(round(num*train_ratio))
num_test = int(round(num*test_ratio))
print(num_train)
print(num_test)


ind = random.sample(image_files, num_train)
src = [''.join(["./datasets/images/", e]) for e in ind]
dst = [''.join(['./datasets/images/train/', e]) for e in ind]
print(src[0])
print(dst[0])
#  shutil.move(src[0], dst[0])
os.makedirs('./datasets/images/train')
os.makedirs('./datasets/images/test')

list(map(shutil.move, src, dst))
print(image_files[0])

