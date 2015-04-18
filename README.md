# ImageClassification
a Spark program that calls caffe to do image classification

how to use it

~/spark-1.2.1-bin-hadoop2.4/bin/spark-submit --master local[1] spark_caffe.py ~/hdd/0 2 256 ~/hdd/out --gpu > out.txt 2>&1
