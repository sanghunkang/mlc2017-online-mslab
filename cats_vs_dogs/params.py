#!/usr/bin/python
# -*- coding: utf-8 -*-
#############################################################################
# Import 3rd party packages
import tensorflow as tf

params = {
	# len_ftmap_end = int(shape_ftmap_end[1]*shape_ftmap_end[2]*shape_ftmap_end[3])
	"conv11_W": tf.Variable(tf.random_normal([3, 3, 3, 32]), name="conv11_W"),
	"conv11_b": tf.Variable(tf.random_normal([32]), name="conv11_b"),

	"conv12_W": tf.Variable(tf.random_normal([3, 3, 32, 32]), name="conv12_W"),
	"conv12_b": tf.Variable(tf.random_normal([32]), name="conv12_b"),

	"conv21_W": tf.Variable(tf.random_normal([3, 3, 32, 64]), name="conv21_W"),
	"conv21_b": tf.Variable(tf.random_normal([64]), name="conv21_b"),

	"conv22_W": tf.Variable(tf.random_normal([3, 3, 64, 64]), name="conv22_W"),
	"conv22_b": tf.Variable(tf.random_normal([64]), name="conv22_b"),

	"fc1_W": tf.Variable(tf.random_normal([8*8*64, 1024]), name="fc1_W"),
	"fc1_b": tf.Variable(tf.random_normal([1024]), name="fc1_b"),

	"fc2_W": tf.Variable(tf.random_normal([1024, 2]), name="fc2_W"),
	"fc2_b": tf.Variable(tf.random_normal([2]), name="fc2_b"), # 2 outputs (class prediction)
}