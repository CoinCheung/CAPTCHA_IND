#!/usr/bin/python
# -*- coding:utf8 -*-


from urllib.request import urlretrieve
import time
import random
import os


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


    def generate_list(self, dir, name="captcha_train"):
        image_files = os.listdir(dir)
        nums = len(image_files)
        filename = "".join([dir,'/',name,'_list.lst'])
        with open(filename, 'w') as f:
            for i in range(nums):
                iterm = image_files[i]
                label = iterm.split('.')[0]
                f.write("{:d}\t{}\t{}\n".format(i, label, iterm))


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
    img_dir = './datasets/images/'
    cap = CAPTCHA()
    cap.download_captcha(dir=img_dir, num=20000)






