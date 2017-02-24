"""Module for constructing RNN Cells with multiplicative_integration"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import math, numpy as np
from six.moves import xrange
import tensorflow as tf
# from multiplicative_integration_modern import multiplicative_integration
from tensorflow.python.ops.nn import rnn_cell
# import highway_network_modern
from linear_modern import linear
import normalization_ops_modern as nom
from normalization_ops_modern import layer_norm
import pdb
RNNCell = rnn_cell.RNNCell

class HighwayRNNCell_LayerNorm(RNNCell):
  """Highway RNN Network with multiplicative_integration"""

  def __init__(self, num_units, num_highway_layers = 3, use_inputs_on_each_layer = False):
    self._num_units = num_units
    self.num_highway_layers = num_highway_layers
    self.use_inputs_on_each_layer = use_inputs_on_each_layer


  @property
  def input_size(self):
    return self._num_units

  @property
  def output_size(self):
    return self._num_units

  @property
  def state_size(self):
    return self._num_units

  def __call__(self, inputs, state, timestep = 0, scope=None):
    current_state = state
    for highway_layer in xrange(self.num_highway_layers):
      with tf.variable_scope('highway_factor_'+str(highway_layer)):
        if self.use_inputs_on_each_layer or highway_layer == 0:
          
          highway_factor = tf.tanh(layer_norm(linear([inputs, current_state], self._num_units, False)))
        else:
          
          highway_factor = tf.tanh(layer_norm(linear([current_state], self._num_units, False)))
      with tf.variable_scope('gate_for_highway_factor_'+str(highway_layer)):
        if self.use_inputs_on_each_layer or highway_layer == 0:
          gate_for_highway_factor = tf.sigmoid(linear([inputs, current_state], self._num_units, True, -3.0))
        else:
          gate_for_highway_factor = tf.sigmoid(linear([current_state], self._num_units, True, -3.0))

        # relax C = 1 - T constrain
        # gate_for_hidden_factor = 1.0 - gate_for_highway_factor
      with tf.variable_scope('gate_for_hidden_factor_'+str(highway_layer)):
        if self.use_inputs_on_each_layer or highway_layer == 0:
          gate_for_hidden_factor = tf.sigmoid(linear([inputs, current_state], self._num_units, True, -3.0))
        else:
          gate_for_hidden_factor = tf.sigmoid(linear([current_state], self._num_units, True, -3.0))

      current_state = highway_factor * gate_for_highway_factor + current_state * gate_for_hidden_factor
    # pdb.set_trace()
    return current_state, current_state
	

