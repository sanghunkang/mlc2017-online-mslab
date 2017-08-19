# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Contains model definitions."""
import math

import models
import tensorflow as tf
import utils

import tensorflow.contrib.slim as slim

# from nets import inception_utils

trunc_normal = lambda stddev: tf.truncated_normal_initializer(0.0, stddev)

def vgg16(inputs):
	with slim.arg_scope([slim.conv2d, slim.fully_connected],
											activation_fn=tf.nn.relu,
											weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
											weights_regularizer=slim.l2_regularizer(0.0005)):
		net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
		net = slim.max_pool2d(net, [2, 2], scope='pool1')
		net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
		net = slim.max_pool2d(net, [2, 2], scope='pool2')
		net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
		net = slim.max_pool2d(net, [2, 2], scope='pool3')
		net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
		net = slim.max_pool2d(net, [2, 2], scope='pool4')
		net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
		net = slim.max_pool2d(net, [2, 2], scope='pool5')
		net = slim.fully_connected(net, 4096, scope='fc6')
		net = slim.dropout(net, 0.5, scope='dropout6')
		net = slim.fully_connected(net, 4096, scope='fc7')
		net = slim.dropout(net, 0.5, scope='dropout7')
		net = slim.fully_connected(net, 1, activation_fn=None, scope='fc8')
	return net

def inception_v1_base(inputs,
											final_endpoint='Mixed_3c',
											scope='InceptionV1'):
	"""Defines the Inception V1 base architecture.
	This architecture is defined in:
		Going deeper with convolutions
		Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed,
		Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke, Andrew Rabinovich.
		http://arxiv.org/pdf/1409.4842v1.pdf.
	Args:
		inputs: a tensor of size [batch_size, height, width, channels].
		final_endpoint: specifies the endpoint to construct the network up to. It
			can be one of ['Conv2d_1a_7x7', 'MaxPool_2a_3x3', 'Conv2d_2b_1x1',
			'Conv2d_2c_3x3', 'MaxPool_3a_3x3', 'Mixed_3b', 'Mixed_3c',
			'MaxPool_4a_3x3', 'Mixed_4b', 'Mixed_4c', 'Mixed_4d', 'Mixed_4e',
			'Mixed_4f', 'MaxPool_5a_2x2', 'Mixed_5b', 'Mixed_5c']
		scope: Optional variable_scope.
	Returns:
		A dictionary from components of the network to the corresponding activation.
	Raises:
		ValueError: if final_endpoint is not set to one of the predefined values.
	"""
	end_points = {}
	with tf.variable_scope(scope, 'InceptionV1', [inputs]):
		with slim.arg_scope(
				[slim.conv2d, slim.fully_connected],
				weights_initializer=trunc_normal(0.01)):
			with slim.arg_scope([slim.conv2d, slim.max_pool2d],
													stride=1, padding='SAME'):
				end_point = 'Conv2d_1a_7x7'
				net = slim.conv2d(inputs, 64, [3, 3], stride=2, scope=end_point)
				end_points[end_point] = net
				if final_endpoint == end_point: return net, end_points
				end_point = 'MaxPool_2a_3x3'
				net = slim.max_pool2d(net, [3, 3], stride=2, scope=end_point)
				end_points[end_point] = net
				if final_endpoint == end_point: return net, end_points
				end_point = 'Conv2d_2b_1x1'
				net = slim.conv2d(net, 64, [1, 1], scope=end_point)
				end_points[end_point] = net
				if final_endpoint == end_point: return net, end_points
				end_point = 'Conv2d_2c_3x3'
				net = slim.conv2d(net, 192, [3, 3], scope=end_point)
				end_points[end_point] = net
				if final_endpoint == end_point: return net, end_points
				end_point = 'MaxPool_3a_3x3'
				net = slim.max_pool2d(net, [3, 3], stride=2, scope=end_point)
				end_points[end_point] = net
				if final_endpoint == end_point: return net, end_points

				end_point = 'Mixed_3b'
				with tf.variable_scope(end_point):
					with tf.variable_scope('Branch_0'):
						branch_0 = slim.conv2d(net, 64, [1, 1], scope='Conv2d_0a_1x1')
					with tf.variable_scope('Branch_1'):
						branch_1 = slim.conv2d(net, 96, [1, 1], scope='Conv2d_0a_1x1')
						branch_1 = slim.conv2d(branch_1, 128, [3, 3], scope='Conv2d_0b_3x3')
					with tf.variable_scope('Branch_2'):
						branch_2 = slim.conv2d(net, 16, [1, 1], scope='Conv2d_0a_1x1')
						branch_2 = slim.conv2d(branch_2, 32, [3, 3], scope='Conv2d_0b_3x3')
					with tf.variable_scope('Branch_3'):
						branch_3 = slim.max_pool2d(net, [3, 3], scope='MaxPool_0a_3x3')
						branch_3 = slim.conv2d(branch_3, 32, [1, 1], scope='Conv2d_0b_1x1')
					net = tf.concat(
							axis=3, values=[branch_0, branch_1, branch_2, branch_3])
				end_points[end_point] = net
				if final_endpoint == end_point: return net, end_points

				end_point = 'Mixed_3c'
				with tf.variable_scope(end_point):
					with tf.variable_scope('Branch_0'):
						branch_0 = slim.conv2d(net, 128, [1, 1], scope='Conv2d_0a_1x1')
					with tf.variable_scope('Branch_1'):
						branch_1 = slim.conv2d(net, 128, [1, 1], scope='Conv2d_0a_1x1')
						branch_1 = slim.conv2d(branch_1, 192, [3, 3], scope='Conv2d_0b_3x3')
					with tf.variable_scope('Branch_2'):
						branch_2 = slim.conv2d(net, 32, [1, 1], scope='Conv2d_0a_1x1')
						branch_2 = slim.conv2d(branch_2, 96, [3, 3], scope='Conv2d_0b_3x3')
					with tf.variable_scope('Branch_3'):
						branch_3 = slim.max_pool2d(net, [3, 3], scope='MaxPool_0a_3x3')
						branch_3 = slim.conv2d(branch_3, 64, [1, 1], scope='Conv2d_0b_1x1')
					net = tf.concat(
							axis=3, values=[branch_0, branch_1, branch_2, branch_3])
				end_points[end_point] = net
				if final_endpoint == end_point: return net, end_points

		# 		end_point = 'MaxPool_4a_3x3'
		# 		net = slim.max_pool2d(net, [3, 3], stride=2, scope=end_point)
		# 		end_points[end_point] = net
		# 		if final_endpoint == end_point: return net, end_points

		# 		end_point = 'Mixed_4b'
		# 		with tf.variable_scope(end_point):
		# 			with tf.variable_scope('Branch_0'):
		# 				branch_0 = slim.conv2d(net, 192, [1, 1], scope='Conv2d_0a_1x1')
		# 			with tf.variable_scope('Branch_1'):
		# 				branch_1 = slim.conv2d(net, 96, [1, 1], scope='Conv2d_0a_1x1')
		# 				branch_1 = slim.conv2d(branch_1, 208, [3, 3], scope='Conv2d_0b_3x3')
		# 			with tf.variable_scope('Branch_2'):
		# 				branch_2 = slim.conv2d(net, 16, [1, 1], scope='Conv2d_0a_1x1')
		# 				branch_2 = slim.conv2d(branch_2, 48, [3, 3], scope='Conv2d_0b_3x3')
		# 			with tf.variable_scope('Branch_3'):
		# 				branch_3 = slim.max_pool2d(net, [3, 3], scope='MaxPool_0a_3x3')
		# 				branch_3 = slim.conv2d(branch_3, 64, [1, 1], scope='Conv2d_0b_1x1')
		# 			net = tf.concat(
		# 					axis=3, values=[branch_0, branch_1, branch_2, branch_3])
		# 		end_points[end_point] = net
		# 		if final_endpoint == end_point: return net, end_points

		# 		end_point = 'Mixed_4c'
		# 		with tf.variable_scope(end_point):
		# 			with tf.variable_scope('Branch_0'):
		# 				branch_0 = slim.conv2d(net, 160, [1, 1], scope='Conv2d_0a_1x1')
		# 			with tf.variable_scope('Branch_1'):
		# 				branch_1 = slim.conv2d(net, 112, [1, 1], scope='Conv2d_0a_1x1')
		# 				branch_1 = slim.conv2d(branch_1, 224, [3, 3], scope='Conv2d_0b_3x3')
		# 			with tf.variable_scope('Branch_2'):
		# 				branch_2 = slim.conv2d(net, 24, [1, 1], scope='Conv2d_0a_1x1')
		# 				branch_2 = slim.conv2d(branch_2, 64, [3, 3], scope='Conv2d_0b_3x3')
		# 			with tf.variable_scope('Branch_3'):
		# 				branch_3 = slim.max_pool2d(net, [3, 3], scope='MaxPool_0a_3x3')
		# 				branch_3 = slim.conv2d(branch_3, 64, [1, 1], scope='Conv2d_0b_1x1')
		# 			net = tf.concat(
		# 					axis=3, values=[branch_0, branch_1, branch_2, branch_3])
		# 		end_points[end_point] = net
		# 		if final_endpoint == end_point: return net, end_points

		# 		end_point = 'Mixed_4d'
		# 		with tf.variable_scope(end_point):
		# 			with tf.variable_scope('Branch_0'):
		# 				branch_0 = slim.conv2d(net, 128, [1, 1], scope='Conv2d_0a_1x1')
		# 			with tf.variable_scope('Branch_1'):
		# 				branch_1 = slim.conv2d(net, 128, [1, 1], scope='Conv2d_0a_1x1')
		# 				branch_1 = slim.conv2d(branch_1, 256, [3, 3], scope='Conv2d_0b_3x3')
		# 			with tf.variable_scope('Branch_2'):
		# 				branch_2 = slim.conv2d(net, 24, [1, 1], scope='Conv2d_0a_1x1')
		# 				branch_2 = slim.conv2d(branch_2, 64, [3, 3], scope='Conv2d_0b_3x3')
		# 			with tf.variable_scope('Branch_3'):
		# 				branch_3 = slim.max_pool2d(net, [3, 3], scope='MaxPool_0a_3x3')
		# 				branch_3 = slim.conv2d(branch_3, 64, [1, 1], scope='Conv2d_0b_1x1')
		# 			net = tf.concat(
		# 					axis=3, values=[branch_0, branch_1, branch_2, branch_3])
		# 		end_points[end_point] = net
		# 		if final_endpoint == end_point: return net, end_points

		# 		end_point = 'Mixed_4e'
		# 		with tf.variable_scope(end_point):
		# 			with tf.variable_scope('Branch_0'):
		# 				branch_0 = slim.conv2d(net, 112, [1, 1], scope='Conv2d_0a_1x1')
		# 			with tf.variable_scope('Branch_1'):
		# 				branch_1 = slim.conv2d(net, 144, [1, 1], scope='Conv2d_0a_1x1')
		# 				branch_1 = slim.conv2d(branch_1, 288, [3, 3], scope='Conv2d_0b_3x3')
		# 			with tf.variable_scope('Branch_2'):
		# 				branch_2 = slim.conv2d(net, 32, [1, 1], scope='Conv2d_0a_1x1')
		# 				branch_2 = slim.conv2d(branch_2, 64, [3, 3], scope='Conv2d_0b_3x3')
		# 			with tf.variable_scope('Branch_3'):
		# 				branch_3 = slim.max_pool2d(net, [3, 3], scope='MaxPool_0a_3x3')
		# 				branch_3 = slim.conv2d(branch_3, 64, [1, 1], scope='Conv2d_0b_1x1')
		# 			net = tf.concat(
		# 					axis=3, values=[branch_0, branch_1, branch_2, branch_3])
		# 		end_points[end_point] = net
		# 		if final_endpoint == end_point: return net, end_points

		# 		end_point = 'Mixed_4f'
		# 		with tf.variable_scope(end_point):
		# 			with tf.variable_scope('Branch_0'):
		# 				branch_0 = slim.conv2d(net, 256, [1, 1], scope='Conv2d_0a_1x1')
		# 			with tf.variable_scope('Branch_1'):
		# 				branch_1 = slim.conv2d(net, 160, [1, 1], scope='Conv2d_0a_1x1')
		# 				branch_1 = slim.conv2d(branch_1, 320, [3, 3], scope='Conv2d_0b_3x3')
		# 			with tf.variable_scope('Branch_2'):
		# 				branch_2 = slim.conv2d(net, 32, [1, 1], scope='Conv2d_0a_1x1')
		# 				branch_2 = slim.conv2d(branch_2, 128, [3, 3], scope='Conv2d_0b_3x3')
		# 			with tf.variable_scope('Branch_3'):
		# 				branch_3 = slim.max_pool2d(net, [3, 3], scope='MaxPool_0a_3x3')
		# 				branch_3 = slim.conv2d(branch_3, 128, [1, 1], scope='Conv2d_0b_1x1')
		# 			net = tf.concat(
		# 					axis=3, values=[branch_0, branch_1, branch_2, branch_3])
		# 		end_points[end_point] = net
		# 		if final_endpoint == end_point: return net, end_points

		# 		end_point = 'MaxPool_5a_2x2'
		# 		net = slim.max_pool2d(net, [2, 2], stride=2, scope=end_point)
		# 		end_points[end_point] = net
		# 		if final_endpoint == end_point: return net, end_points

		# 		end_point = 'Mixed_5b'
		# 		with tf.variable_scope(end_point):
		# 			with tf.variable_scope('Branch_0'):
		# 				branch_0 = slim.conv2d(net, 256, [1, 1], scope='Conv2d_0a_1x1')
		# 			with tf.variable_scope('Branch_1'):
		# 				branch_1 = slim.conv2d(net, 160, [1, 1], scope='Conv2d_0a_1x1')
		# 				branch_1 = slim.conv2d(branch_1, 320, [3, 3], scope='Conv2d_0b_3x3')
		# 			with tf.variable_scope('Branch_2'):
		# 				branch_2 = slim.conv2d(net, 32, [1, 1], scope='Conv2d_0a_1x1')
		# 				branch_2 = slim.conv2d(branch_2, 128, [3, 3], scope='Conv2d_0a_3x3')
		# 			with tf.variable_scope('Branch_3'):
		# 				branch_3 = slim.max_pool2d(net, [3, 3], scope='MaxPool_0a_3x3')
		# 				branch_3 = slim.conv2d(branch_3, 128, [1, 1], scope='Conv2d_0b_1x1')
		# 			net = tf.concat(
		# 					axis=3, values=[branch_0, branch_1, branch_2, branch_3])
		# 		end_points[end_point] = net
		# 		if final_endpoint == end_point: return net, end_points

		# 		end_point = 'Mixed_5c'
		# 		with tf.variable_scope(end_point):
		# 			with tf.variable_scope('Branch_0'):
		# 				branch_0 = slim.conv2d(net, 384, [1, 1], scope='Conv2d_0a_1x1')
		# 			with tf.variable_scope('Branch_1'):
		# 				branch_1 = slim.conv2d(net, 192, [1, 1], scope='Conv2d_0a_1x1')
		# 				branch_1 = slim.conv2d(branch_1, 384, [3, 3], scope='Conv2d_0b_3x3')
		# 			with tf.variable_scope('Branch_2'):
		# 				branch_2 = slim.conv2d(net, 48, [1, 1], scope='Conv2d_0a_1x1')
		# 				branch_2 = slim.conv2d(branch_2, 128, [3, 3], scope='Conv2d_0b_3x3')
		# 			with tf.variable_scope('Branch_3'):
		# 				branch_3 = slim.max_pool2d(net, [3, 3], scope='MaxPool_0a_3x3')
		# 				branch_3 = slim.conv2d(branch_3, 128, [1, 1], scope='Conv2d_0b_1x1')
		# 			net = tf.concat(
		# 					axis=3, values=[branch_0, branch_1, branch_2, branch_3])
		# 		end_points[end_point] = net
		# 		if final_endpoint == end_point: return net, end_points
		# raise ValueError('Unknown final endpoint %s' % final_endpoint)

class LogisticModel(models.BaseModel):
	"""Logistic model with L2 regularization."""

	def create_model(self, model_input, num_classes=2, l2_penalty=1e-8, **unused_params):
		"""Creates a logistic model.

		Args:
			model_input: 'batch' x 'num_features' matrix of input features.
			vocab_size: The number of classes in the dataset.

		Returns:
			A dictionary with a tensor containing the probability predictions of the
			model in the 'predictions' key. The dimensions of the tensor are
			batch_size x num_classes."""
		net = slim.flatten(model_input)
		print(net)
		output = slim.fully_connected(
				net, num_classes - 1, activation_fn=tf.nn.sigmoid,
				weights_regularizer=slim.l2_regularizer(l2_penalty))
		return {"predictions": output}

class SubmissionModel(models.BaseModel):
	def create_model(self, model_input, num_classes=2, l2_penalty=0.001, **unused_params):
		print("___________________________________________")
		print(unused_params)
		is_training=unused_params["is_training"]
		# is_training=True
		print(is_training)

		net = model_input
		print(net.shape)

		net = slim.batch_norm(net, is_training=is_training, scope='bn_conv11')
		net = slim.conv2d(net, 32, [3, 3], scope='conv11', weights_regularizer=slim.l2_regularizer(l2_penalty))
		net = slim.dropout(net, 0.5, scope='dropout11')
		net = slim.batch_norm(net, is_training=is_training, scope='bn_conv11')
		net = slim.conv2d(net, 32, [3, 3], scope='conv12', weights_regularizer=slim.l2_regularizer(l2_penalty))
		net = slim.dropout(net, 0.5, scope='dropout11')
		net = slim.max_pool2d(net, [2, 2], scope='pool1')
		# net = slim.batch_norm(net, is_training=is_training, is_training=is_training, scope='bn_conv1')
		
		print(net.shape)

		net = slim.batch_norm(net, is_training=is_training, scope='bn_conv21')
		net = slim.conv2d(net, 64, [3, 3], scope='conv21', weights_regularizer=slim.l2_regularizer(l2_penalty))
		net = slim.dropout(net, 0.5, scope='dropout21')
		net = slim.batch_norm(net, is_training=is_training, scope='bn_conv22')
		net = slim.conv2d(net, 64, [3, 3], scope='conv22', weights_regularizer=slim.l2_regularizer(l2_penalty))
		net = slim.dropout(net, 0.5, scope='dropout22')
		net = slim.max_pool2d(net, [2, 2], scope='pool2')
		# net = slim.batch_norm(net, is_training=is_training, is_training=is_training, scope='bn_conv2')
		# net = slim.dropout(net, 0.5, scope='dropout2')
		print(net.shape)

		net = slim.batch_norm(net, is_training=is_training, scope='bn_conv31')
		net = slim.conv2d(net, 128, [3, 3], scope='conv31', weights_regularizer=slim.l2_regularizer(l2_penalty))
		net = slim.dropout(net, 0.5, scope='dropout31')
		net = slim.batch_norm(net, is_training=is_training, scope='bn_conv32')
		net = slim.conv2d(net, 128, [3, 3], scope='conv32', weights_regularizer=slim.l2_regularizer(l2_penalty))
		net = slim.dropout(net, 0.5, scope='dropout32')
		net = slim.max_pool2d(net, [2, 2], scope='pool3')
		# net = slim.batch_norm(net, is_training=is_training, is_training=is_training, scope='bn_conv3')
		# net = slim.dropout(net, 0.5, scope='dropout3')
		print(net.shape)

		net = slim.batch_norm(net, is_training=is_training, scope='bn_conv41')
		net = slim.conv2d(net, 256, [3, 3], scope='conv41', weights_regularizer=slim.l2_regularizer(l2_penalty))
		net = slim.dropout(net, 0.5, scope='dropout41')
		net = slim.batch_norm(net, is_training=is_training, scope='bn_conv42')
		net = slim.conv2d(net, 256, [3, 3], scope='conv42', weights_regularizer=slim.l2_regularizer(l2_penalty))
		net = slim.dropout(net, 0.5, scope='dropout42')
		net = slim.max_pool2d(net, [2, 2], scope='pool4')
		# net = slim.batch_norm(net, is_training=is_training, is_training=is_training, scope='bn_conv4')
		# net = slim.dropout(net, 0.5, scope='dropout3')
		print(net.shape)

		net = slim.batch_norm(net, is_training=is_training, scope='bn_conv51')
		net = slim.conv2d(net, 512, [3, 3], scope='conv51', weights_regularizer=slim.l2_regularizer(l2_penalty))
		net = slim.dropout(net, 0.5, scope='dropout51')
		net = slim.batch_norm(net, is_training=is_training, scope='bn_conv52')
		net = slim.conv2d(net, 512, [3, 3], scope='conv52', weights_regularizer=slim.l2_regularizer(l2_penalty))
		net = slim.dropout(net, 0.5, scope='dropout52')
		net = slim.max_pool2d(net, [2, 2], scope='pool5')
		# # net = slim.dropout(net, 0.5, scope='dropout3')
		print(net.shape)

		net = slim.flatten(net)
		net = slim.batch_norm(net, is_training=is_training, scope='bn1')
		net = slim.fully_connected(net, int(net.shape[-1]), activation_fn=tf.nn.relu, weights_regularizer=slim.l2_regularizer(l2_penalty), scope='fc1')
		net = slim.dropout(net, 0.5, is_training=is_training, scope='dropoutfc1')
		net = slim.batch_norm(net, is_training=is_training, scope='bn2')
		net = slim.fully_connected(net, num_classes - 1, activation_fn=tf.nn.sigmoid, weights_regularizer=slim.l2_regularizer(l2_penalty), scope='fc2')
		print("___________________________________________")
		return {"predictions": net}

class MoeModel(models.BaseModel):
	"""A softmax over a mixture of logistic models (with L2 regularization)."""

	def create_model(self,
									 model_input,
									 vocab_size=2,
									 num_mixtures=None,
									 l2_penalty=1e-8,
									 **unused_params):
		"""Creates a Mixture of (Logistic) Experts model.

		 The model consists of a per-class softmax distribution over a
		 configurable number of logistic classifiers. One of the classifiers in the
		 mixture is not trained, and always predicts 0.

		Args:
			model_input: 'batch_size' x 'num_features' matrix of input features.
			vocab_size: The number of classes in the dataset.
			num_mixtures: The number of mixtures (excluding a dummy 'expert' that
				always predicts the non-existence of an entity).
			l2_penalty: How much to penalize the squared magnitudes of parameter
				values.
		Returns:
			A dictionary with a tensor containing the probability predictions of the
			model in the 'predictions' key. The dimensions of the tensor are
			batch_size x num_classes.
		"""
		num_mixtures = num_mixtures or FLAGS.moe_num_mixtures

		gate_activations = slim.fully_connected(
				model_input,
				vocab_size * (num_mixtures + 1),
				activation_fn=None,
				biases_initializer=None,
				weights_regularizer=slim.l2_regularizer(l2_penalty),
				scope="gates")
		expert_activations = slim.fully_connected(
				model_input,
				vocab_size * num_mixtures,
				activation_fn=None,
				weights_regularizer=slim.l2_regularizer(l2_penalty),
				scope="experts")

		gating_distribution = tf.nn.softmax(tf.reshape(
				gate_activations,
				[-1, num_mixtures + 1]))  # (Batch * #Labels) x (num_mixtures + 1)
		expert_distribution = tf.nn.sigmoid(tf.reshape(
				expert_activations,
				[-1, num_mixtures]))  # (Batch * #Labels) x num_mixtures

		final_probabilities_by_class_and_batch = tf.reduce_sum(
				gating_distribution[:, :num_mixtures] * expert_distribution, 1)
		final_probabilities = tf.reshape(final_probabilities_by_class_and_batch,
																		 [-1, vocab_size])
		return {"predictions": final_probabilities}
