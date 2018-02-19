#!/usr/bin/python
# -*- coding:utf8 -*-


from urllib.request import urlretrieve
import time
import random
import os
import argparse


class CAPTCHA(object):
    def __init__(self):
        self.url = 'http://cuijiahua.com/tutrial/discuz/index.php?label='
        self.num_set = ['0','1','2','3','4','5','6','7','8','9']
        self.alphabet_set = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
        self.char_set = self.num_set + self.alphabet_set


    def random_captcha(self, captcha_size=4):
        captcha_set = [random.choice(self.char_set) for i in range(captcha_size)]
        captcha = ''.join(captcha_set)
        return captcha


    def generate_list(self, img_dir, lst_name="captcha_train"):
        '''

        '''
        image_files = os.listdir(img_dir)
        nums = len(image_files)
        filename = "".join([lst_name,'_list.lst'])
        with open(filename, 'w') as f:
            for i in range(nums):
                iterm = image_files[i]
                label = iterm.split('.')[0]
                label_list = list(map(self.char_set.index, label))
                f.write("{:d}\t{:d}\t{:d}\t{:d}\t{:d}\t{}\n".format(i,label_list[0],label_list[1],label_list[2],label_list[3],iterm))


    def download_captcha(self, dir='./datasets', num=5000):
        if not os.path.exists(dir):
            os.makedirs(dir)

        for i in range(num):
            label = self.random_captcha()
            print('download captcha {}: {}'.format(i+1, label))
            url = ''.join([self.url+label])
            filename = ''.join([dir, '/', label, '.jpg'])
            urlretrieve(url=url, filename=filename)
            time.sleep(0.2)




if __name__ == '__main__':
    img_dir = './datasets/images'
    train_dir = ''.join([img_dir,'/train'])
    test_dir = ''.join([img_dir,'/test'])
    train_ratio = 0.8
    test_ratio = 0.2
    num = 20000

    num_train = int(round(num*train_ratio))
    num_test = int(round(num*test_ratio))

    cap = CAPTCHA()
    cap.download_captcha(dir=train_dir, num=num_train)
    cap.download_captcha(dir=test_dir, num=num_test)
    cap.generate_list(img_dir=train_dir,lst_name=img_dir+'/../train')
    cap.generate_list(img_dir=test_dir,lst_name=img_dir+'/../test')






