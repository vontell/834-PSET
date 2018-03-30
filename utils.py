#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Various utiltiy and testing functions
"""
import numpy as np
import cv2
import os
from numpy_sift import SIFTDescriptor 
from nose.tools import assert_equal, ok_
from IPython.display import display, HTML, clear_output

def test_ok():
    """ If execution gets to this point, print out a happy message """
    try:
        from IPython.display import display_html
        display_html("""<div class="alert alert-success">
        <strong>Tests passed!!</strong>
        </div>""", raw=True)
    except:
        print("Tests passed!!")

def test_patch_entropy_solution(get_patch_entropy):
    test_probs = [-.1, -0, -.5, -.4, -.3, -1, -0, -.5, -.7]
    assert get_patch_entropy(test_probs) - 0.3888888888888889 < 0.0001, "Total patch entropy is incorrect"
    return True
        
def check_example(param):
    assert param, "This should be true!"
    return True

def kmeans_accuracy(preds, labels_list):
    if labels_list[0] == 'Auditorium':
        gt1 = np.array([1 for i in range(20)] + [0 for i in range(31)])
        gt2 = np.array([0 for i in range(20)] + [1 for i in range(31)])
    elif labels_list[0] == 'Bowling':
        gt1 = np.array([1 for i in range(31)] + [0 for i in range(20)])
        gt2 = np.array([0 for i in range(31)] + [1 for i in range(20)])
    else:
        return -1
    return max(np.sum(gt1 == preds), np.sum(gt2 == preds)) * 1.0 / gt1.size

def return_labels(directoryList, range_size, added_limit, count_limit, patch_size=32):
    print(added_limit)
    print("range: " , range_size)
    print("count: " , count_limit)
    SD = SIFTDescriptor(patchSize = patch_size)

    #creates initial vector of size 3968
    sift_pictures = np.asarray([[0 for _ in range(range_size)]])
    for directory in directoryList:
        #go through the directories with the relevant images
        for filename in os.listdir(directory):
            name = directory + '/' + filename
            if name[-3:] == 'png' or name[-3:] == 'jpg':
                image = cv2.imread(name)
                
                #transform it to black and white and get features
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                h, w = gray.shape
                corners = cv2.goodFeaturesToTrack(gray,count_limit,0.01,10)

                corners = np.int0(corners)
                image_sift_features = np.asarray([]) 
                count, actually_added = 0, 0
                
                #Here we want to get 31 images for our feature vector and then break out of the loop
                while actually_added < added_limit and count < count_limit:
                    corner = corners[count][0]
                    
                    if patch_size/2 <= corner[1] <= h-patch_size/2 and patch_size/2 <= corner[0] <= w-patch_size/2:
                        patch = gray[corner[1]-int(patch_size/2):corner[1]+int(patch_size/2), corner[0]-int(patch_size/2):corner[0]+int(patch_size/2)]
                        sift = SD.describe(patch)
                        image_sift_features = np.append(image_sift_features, sift)
                        actually_added += 1
                    count += 1

                sift_pictures = np.append(sift_pictures, [image_sift_features], axis=0)

    return sift_pictures[1:]

