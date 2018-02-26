#!/usr/bin/python
# -*- encoding: utf8 -*-


def vec2str(vector):
    '''
    Convert label vector to its associated captcha string
    params:
        vector: a list whose elements are integers in range [0, 35] standing for number 0-9 and alphabet a-z
    return:
        a string associated with the given integer vector
    '''
    num_set = ['0','1','2','3','4','5','6','7','8','9']
    alphabet_set = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
    char_set = num_set + alphabet_set

    return ''.join([char_set[i] for i in vector])


