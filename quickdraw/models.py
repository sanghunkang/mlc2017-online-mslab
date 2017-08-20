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

import math

import tensorflow as tf
import utils

import tensorflow.contrib.slim as slim

"""Contains the base class for models."""
class BaseModel(object):
	"""Inherit from this class when implementing new models."""

	def create_model(self, unused_model_input, **unused_params):
		raise NotImplementedError()

class SubmissionModel(BaseModel):
	def create_model(self, model_input, num_classes=10, l2_penalty=1e-8, **unused_params):
		print("___________________________________________")
		print(unused_params)
		is_training=unused_params["is_training"]
		# is_training=True
		print(is_training)
		# net = slim.batch_norm(net, scope='bn_conv2')
		# net = slim.dropout(net, 0.1, scope='dropout2')

		net = model_input
		print(net.shape)

		net = slim.conv2d(net, 32, [8, 8], scope='conv11')#, weights_regularizer=slim.l2_regularizer(l2_penalty))
		net = slim.max_pool2d(net, [2, 2], scope='pool11')
		print(net.shape)

		net = slim.conv2d(net, 64, [3, 3], scope='conv12')#, weights_regularizer=slim.l2_regularizer(l2_penalty))
		net = slim.max_pool2d(net, [2, 2], scope='pool12')
		print(net.shape)

		net = slim.conv2d(net, 128, [3, 3], scope='conv21')#, weights_regularizer=slim.l2_regularizer(l2_penalty))
		net = slim.max_pool2d(net, [2, 2], scope='pool21')
		print(net.shape)

		net = slim.conv2d(net, 256, [3, 3], scope='conv22')#, weights_regularizer=slim.l2_regularizer(l2_penalty))
		net = slim.max_pool2d(net, [2, 2], scope='pool22')
		print(net.shape)

		net = slim.conv2d(net, 512, [3, 3], scope='conv31')#, weights_regularizer=slim.l2_regularizer(l2_penalty))
		net = slim.max_pool2d(net, [3, 3], scope='pool31')
		print(net.shape)

		# net = slim.conv2d(net, 128, [3, 3], scope='conv32')#, weights_regularizer=slim.l2_regularizer(l2_penalty))
		# net = slim.max_pool2d(net, [2, 2], scope='pool32')
		# print(net.shape)

		# net = slim.conv2d(net, 256, [3, 3], scope='conv41', weights_regularizer=slim.l2_regularizer(l2_penalty))
		# net = slim.conv2d(net, 256, [3, 3], scope='conv42', weights_regularizer=slim.l2_regularizer(l2_penalty))
		# net = slim.max_pool2d(net, [2, 2], scope='pool4')
		# # net = slim.batch_norm(net, scope='bn_conv4')
		# # net = slim.dropout(net, 0.1, scope='dropout3')
		# print(net.shape)

		# net = slim.conv2d(net, 512, [3, 3], scope='conv51', weights_regularizer=slim.l2_regularizer(l2_penalty))
		# net = slim.conv2d(net, 512, [3, 3], scope='conv52', weights_regularizer=slim.l2_regularizer(l2_penalty))
		# net = slim.max_pool2d(net, [2, 2], scope='pool5')
		# # net = slim.batch_norm(net, scope='bn_conv5')
		# # net = slim.dropout(net, 0.1, scope='dropout3')
		# print(net.shape)

		net = slim.flatten(net)
		# net = slim.batch_norm(net, scope='bn1')
		net = slim.dropout(net, 0.9, is_training=is_training, scope='dropout1')
		net = slim.fully_connected(net, int(net.shape[-1]), activation_fn=tf.nn.relu, weights_regularizer=slim.l2_regularizer(l2_penalty), scope='fc1')
		# net = slim.batch_norm(net, scope='bn2')
		net = slim.dropout(net, 0.9, is_training=is_training, scope='dropout2')
		net = slim.fully_connected(net, num_classes, activation_fn=tf.nn.sigmoid, weights_regularizer=slim.l2_regularizer(l2_penalty), scope='fc2')
		print("___________________________________________")
		return {"predictions": net}

class LogisticModel(BaseModel):
	"""Logistic model with L2 regularization."""

	def create_model(self, model_input, num_classes=10, l2_penalty=1e-8, **unused_params):
		"""Creates a logistic model.

		Args:
			model_input: 'batch' x 'num_features' matrix of input features.
			num_classes: The number of classes in the dataset.

		Returns:
			A dictionary with a tensor containing the probability predictions of the
			model in the 'predictions' key. The dimensions of the tensor are
			batch_size x num_classes."""
		net = slim.flatten(model_input)
		output = slim.fully_connected(
				net, num_classes, activation_fn=None,
				weights_regularizer=slim.l2_regularizer(l2_penalty))
		return {"predictions": output}
