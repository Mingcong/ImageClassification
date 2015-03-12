#!/usr/bin/env python
"""
classify.py is an out-of-the-box image classifer callable from the command line.

By default it configures and runs the Caffe reference ImageNet model.
"""
import numpy as np
import os
import sys
import argparse
import glob
import time

import caffe

import cv2
import string
import numpy as np

from pyspark import SparkContext,SparkConf


def main(argv):
    conf = SparkConf()
    sc = SparkContext(appName="PythonCaffe",pyFiles=['/usr/local/lib/python2.7/site-packages/cv2.so'],environment={'KMP_AFFINITY':'scatter','OMP_NUM_THREADS':'16'});
    #sc = SparkContext(appName="PythonCaffe",pyFiles=['/usr/local/lib/python2.7/site-packages/cv2.so','/home/ideal/caffe-memory/build/lib/libcaffe.so']);
    conf.setExecutorEnv("KMP_AFFINITY", "scatter")
    conf.setExecutorEnv("GOMP_CPU_AFFINITY", "0-15")
    conf.setExecutorEnv("OMP_NUM_THREADS", "16")

    pycaffe_dir = os.path.dirname("/home/ideal/caffe-memory/python/ImageClassification")

    parser = argparse.ArgumentParser()
    # Required arguments: input and output files.
    parser.add_argument(
        "input_file",
        help="Input image, directory, or npy."
    )
    parser.add_argument(
        "batch_size",
        help="Input image, directory, or npy."
    )
    # Optional arguments.
    parser.add_argument(
        "--model_def",
        default=os.path.join(pycaffe_dir,
                "../models/bvlc_reference_caffenet/deploy.prototxt"),
        help="Model definition file."
    )
    parser.add_argument(
        "--pretrained_model",
        default=os.path.join(pycaffe_dir,
                "../models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel"),
        help="Trained model weights file."
    )
    parser.add_argument(
        "--gpu",
        action='store_true',
        help="Switch for gpu computation."
    )
    parser.add_argument(
        "--center_only",
        action='store_true',
        help="Switch for prediction from center crop alone instead of " +
             "averaging predictions across crops (default)."
    )
    parser.add_argument(
        "--images_dim",
        default='256,256',
        help="Canonical 'height,width' dimensions of input images."
    )
    parser.add_argument(
        "--mean_file",
        default=os.path.join(pycaffe_dir,
                             'ImageClassification/caffe/imagenet/ilsvrc_2012_mean.npy'),
        help="Data set image mean of H x W x K dimensions (numpy array). " +
             "Set to '' for no mean subtraction."
    )
    parser.add_argument(
        "--input_scale",
        type=float,
        help="Multiply input features by this scale to finish preprocessing."
    )
    parser.add_argument(
        "--raw_scale",
        type=float,
        default=255.0,
        help="Multiply raw input by this scale before preprocessing."
    )
    parser.add_argument(
        "--channel_swap",
        default='2,1,0',
        help="Order to permute input channels. The default converts " +
             "RGB -> BGR since BGR is the Caffe default by way of OpenCV."
    )
    parser.add_argument(
        "--ext",
        default='JPEG',
        help="Image file extension to take as input when a directory " +
             "is given as the input file."
    )
    args = parser.parse_args()
    #sys.path.append('/usr/local/lib/python2.7/site-packages')

    image_dims = [int(s) for s in args.images_dim.split(',')]

    mean, channel_swap = None, None
    if args.mean_file:
        mean = np.load(args.mean_file)
    if args.channel_swap:
        channel_swap = [int(s) for s in args.channel_swap.split(',')]

    def myFunc(s):
    	# Make classifier.
    	classifier = caffe.Classifier(args.model_def, args.pretrained_model,
            image_dims=image_dims, gpu=args.gpu, mean=mean,
            input_scale=args.input_scale, raw_scale=args.raw_scale,
            channel_swap=channel_swap)
	return classifier.predict(s,False)	

    if args.gpu:
        print 'GPU mode'

    # Load numpy array (.npy), directory glob (*.jpg), or image file.
    args.input_file = os.path.expanduser(args.input_file)
    start = time.time()
    L = sc.binaryFiles(args.input_file + '/*.' + args.ext)
    #L = sc.binaryFiles('hdfs://dirt06:9000/100/ILSVRC2012_test_0000009*')
    #L = sc.binaryFiles('hdfs://localhost:9000/*.JPEG').take(batch_size)
    #L = sc.binaryFiles('/home/ideal/ILSVRC2012_test/100/ILSVRC2012_test_00000100.JPEG').take(batch_size)
    batch_size = int(args.batch_size)
    img = L.map(lambda s: (int(string.atoi(s[0][-13:-5])/batch_size),cv2.imdecode(np.asarray(bytearray(s[1]),dtype=np.uint8),1)/255.0))
    imgs = img.groupByKey()
    imgs = imgs.map(lambda s: (s[0],list(s[1]))) 
    # Classify.
    predicts = imgs.map(lambda s: (s[0],myFunc(s[1])))
    class_name = []
    file = open("synset_words.txt")
    for line in file:
	class_name.append(line)
    results = predicts.collect()
    for s in results:
    	for i in range(0,batch_size):
       		print s[0]*batch_size + i, ' predicted class:', s[1][i].argmax()
       		print s[0]*batch_size + i, ' predicted class:', class_name[s[1][i].argmax()]
		
    
    print "Done in %.2f s." % (time.time() - start)
    

    sc.stop()

    # Save
    #np.save(args.output_file, predictions)


if __name__ == '__main__':
    main(sys.argv)
