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


import cv2
import string
import numpy as np

from pyspark import SparkContext,SparkConf


def main(argv):
    conf = SparkConf()
    sc = SparkContext(appName="PythonCaffe",master="local")

    pycaffe_dir = os.path.dirname("/home/ideal/caffe-memory/python/ImageClassification")

    parser = argparse.ArgumentParser()
    # Required arguments: input and output files.
   # parser.add_argument(
    #    "input_file",
     #   help="Input image, directory, or npy."
   # )
    parser.add_argument(
        "batch_size",
        help="the size of Batch Processing."
    )
    #parser.add_argument(
     #   "outName",
    #    help="Output Name."
    #)
    parser.add_argument(
        "--ext",
        default='JPEG',
        help="Image file extension to take as input when a directory " +
             "is given as the input file."
    )
    args = parser.parse_args()
    #sys.path.append('/usr/local/lib/python2.7/site-packages'
    batch_size = int(args.batch_size)
 
    # Load numpy array (.npy), directory glob (*.jpg), or image file.
   # args.input_file = os.path.expanduser(args.input_file)
    start = time.time()
#change happens here
##################################################################
    def displayFoo(iterator):
	for s in iterator:
	    print  "the key is %s" %(s[0])#"the key is " % (int(string.atoi(s[0][-13:-5])/batch_size))
    
    def hash_domain(s):
	#print s[-11:-5]
	return (int(string.atoi(s[-13:-5])/batch_size))

    L = sc.binaryFiles("/home/ideal/hdd/0" + '/*.' + args.ext).partitionBy(2, hash_domain)
    img = L.map(lambda s: (int(string.atoi(s[0][-13:-5])),100,cv2.imdecode(np.asarray(bytearray(s[1]),dtype=np.uint8),1)/255.0))
    img.foreachPartition(displayFoo)
    #there should be a value map to change the size of image
##################################################################
    #print countNum
    
   # print "the numeber of partition is %.f . " % (partitionNum)
    #L = sc.binaryFiles('hdfs://dirt06:9000/100/ILSVRC2012_test_0000009*')
    #L = sc.binaryFiles('hdfs://localhost:9000/*.JPEG').take(batch_size)
    #partitionNum1 = L.getNumPartitions(); 
  
    #print "the numeber of partition is %.f . " % (partitionNum1)

   # imgs = img.groupByKey()
    #imgs = imgs.map(lambda s: (s[0],list(s[1]))) 
    
    print "Done in %.2f s." % (time.time() - start)
    
    sc.stop()

if __name__ == '__main__':
    main(sys.argv)

