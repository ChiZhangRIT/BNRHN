from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import time
import pdb
from tensorflow.contrib.framework.python.ops import variables
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.client import timeline
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.lib.io import file_io
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables as tf_variables
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.summary import summary
from tensorflow.python.training import optimizer as tf_optimizer
from tensorflow.python.training import saver as tf_saver
from tensorflow.python.training import supervisor
from tensorflow.python.training import sync_replicas_optimizer
from tensorflow.python.training import training_util

__all__ = [
    'add_gradients_summaries', 'clip_gradient_norms', 'multiply_gradients',
    'create_train_op', 'train_step', 'train'
]


def clip_gradient_norms(gradients_to_variables, max_norm):
  """Clips the gradients by the given value.
  Args:
    gradients_to_variables: A list of gradient to variable pairs (tuples).
    max_norm: the maximum norm value.
  Returns:
    A list of clipped gradient to variable pairs.
  """
  clipped_grads_and_vars = []
  for grad, var in gradients_to_variables:
    if grad is not None:
      if isinstance(grad, ops.IndexedSlices):
        tmp = clip_ops.clip_by_norm(grad.values, max_norm)
        grad = ops.IndexedSlices(tmp, grad.indices, grad.dense_shape)
      else:
        grad = clip_ops.clip_by_norm(grad, max_norm)
    clipped_grads_and_vars.append((grad, var))
  return clipped_grads_and_vars


def multiply_gradients(grads_and_vars, gradient_multipliers):
  """Multiply specified gradients.
  Args:
    grads_and_vars: A list of gradient to variable pairs (tuples).
    gradient_multipliers: A map from either `Variables` or `Variable` op names
      to the coefficient by which the associated gradient should be scaled.
  Returns:
    The updated list of gradient to variable pairs.
  Raises:
    ValueError: If `grads_and_vars` is not a list or if `gradient_multipliers`
    is empty or None or if `gradient_multipliers` is not a dictionary.
  """
  if not isinstance(grads_and_vars, list):
    raise ValueError('`grads_and_vars` must be a list.')
  if not gradient_multipliers:
    raise ValueError('`gradient_multipliers` is empty.')
  if not isinstance(gradient_multipliers, dict):
    raise ValueError('`gradient_multipliers` must be a dict.')

  multiplied_grads_and_vars = []
  for grad, var in grads_and_vars:
    if var in gradient_multipliers or var.op.name in gradient_multipliers:
      key = var if var in gradient_multipliers else var.op.name
      if grad is None:
        raise ValueError('Requested multiple of `None` gradient.')

      if isinstance(grad, ops.IndexedSlices):
        tmp = grad.values * constant_op.constant(
            gradient_multipliers[key], dtype=grad.dtype)
        grad = ops.IndexedSlices(tmp, grad.indices, grad.dense_shape)
      else:
        grad *= constant_op.constant(
            gradient_multipliers[key], dtype=grad.dtype)
    multiplied_grads_and_vars.append((grad, var))
  return multiplied_grads_and_vars


def add_gradients_summaries(grads_and_vars):
  """Add summaries to gradients.
  Args:
    grads_and_vars: A list of gradient to variable pairs (tuples).
  Returns:
    The list of created summaries.
  """
  summaries = []
  for grad, var in grads_and_vars:
    if grad is not None:
      if isinstance(grad, ops.IndexedSlices):
        grad_values = grad.values
      else:
        grad_values = grad
      summaries.append(
          summary.histogram(var.op.name + '/gradient', grad_values))
      summaries.append(
          summary.scalar(var.op.name + '/gradient_norm',
                         clip_ops.global_norm([grad_values])))
    else:
      logging.info('Var %s has no gradient', var.op.name)

  return summaries


_USE_GLOBAL_STEP = 0


def create_train_op(total_loss,
                    optimizer,
                    global_step=_USE_GLOBAL_STEP,
                    update_ops=None,
                    variables_to_train=None,
                    clip_gradient_norm=0,
                    summarize_gradients=False,
                    gate_gradients=tf_optimizer.Optimizer.GATE_OP,
                    aggregation_method=None,
                    colocate_gradients_with_ops=False,
                    gradient_multipliers=None):

  if global_step is _USE_GLOBAL_STEP:
    global_step = variables.get_or_create_global_step()

  # Update ops use GraphKeys.UPDATE_OPS collection if update_ops is None.
  global_update_ops = set(ops.get_collection(ops.GraphKeys.UPDATE_OPS))
  if update_ops is None:
    update_ops = global_update_ops
  else:
    update_ops = set(update_ops)
  if not global_update_ops.issubset(update_ops):
    logging.warning('update_ops in create_train_op does not contain all the '
                    ' update_ops in GraphKeys.UPDATE_OPS')

  # Make sure update_ops are computed before total_loss.
  if update_ops:
    with ops.control_dependencies(update_ops):
      barrier = control_flow_ops.no_op(name='update_barrier')
    total_loss = control_flow_ops.with_dependencies([barrier], total_loss)

  if variables_to_train is None:
    # Default to tf.trainable_variables()
    variables_to_train = tf_variables.trainable_variables()
  else:
    # Make sure that variables_to_train are in tf.trainable_variables()
    for v in variables_to_train:
      assert v in tf_variables.trainable_variables()

  assert variables_to_train

  # Create the gradients. Note that apply_gradients adds the gradient
  # computation to the current graph.
  grads = optimizer.compute_gradients(
      total_loss,
      variables_to_train,
      gate_gradients=gate_gradients,
      aggregation_method=aggregation_method,
      colocate_gradients_with_ops=colocate_gradients_with_ops)

  # Scale gradients.
  if gradient_multipliers:
    with ops.name_scope('multiply_grads'):
      grads = multiply_gradients(grads, gradient_multipliers)

  # Clip gradients.
  if clip_gradient_norm > 0:
    with ops.name_scope('clip_grads'):
      grads = clip_gradient_norms(grads, clip_gradient_norm)

  # Summarize gradients.
  if summarize_gradients:
    with ops.name_scope('summarize_grads'):
      add_gradients_summaries(grads)

  # Create gradient updates.
  grad_updates = optimizer.apply_gradients(grads, global_step=global_step)

  with ops.name_scope('train_op'):
    # Make sure total_loss is valid.
    total_loss = array_ops.check_numerics(total_loss,
                                          'LossTensor is inf or nan')

    # Ensure the train_tensor computes grad_updates.
    train_op = control_flow_ops.with_dependencies([grad_updates], total_loss)

  # Add the operation used for training to the 'train_op' collection
  train_ops = ops.get_collection_ref(ops.GraphKeys.TRAIN_OP)
  if train_op not in train_ops:
    train_ops.append(train_op)

  return train_op


def _wait_for_step(sess, global_step, step):
  """Wait till the global step has reached at least 'step'.
  Args:
    sess: A session.
    global_step: A Tensor.
    step: Int.  The global step to reach.
  """
  while True:
    if training_util.global_step(sess, global_step) >= step:
      break
    time.sleep(1.0)


def train_step(sess, train_op, global_step, train_step_kwargs):
  """Function that takes a gradient step and specifies whether to stop.
  Args:
    sess: The current session.
    train_op: An `Operation` that evaluates the gradients and returns the
      total loss.
    global_step: A `Tensor` representing the global training step.
    train_step_kwargs: A dictionary of keyword arguments.
  Returns:
    The total loss and a boolean indicating whether or not to stop training.
  Raises:
    ValueError: if 'should_trace' is in `train_step_kwargs` but `logdir` is not.
  """
  start_time = time.time()

  trace_run_options = None
  run_metadata = None
  if 'should_trace' in train_step_kwargs:
    if 'logdir' not in train_step_kwargs:
      raise ValueError('logdir must be present in train_step_kwargs when '
                       'should_trace is present')
    if sess.run(train_step_kwargs['should_trace']):
      trace_run_options = config_pb2.RunOptions(
          trace_level=config_pb2.RunOptions.FULL_TRACE)
      run_metadata = config_pb2.RunMetadata()

  total_loss, np_global_step = sess.run([train_op, global_step],
                                        options=trace_run_options,
                                        run_metadata=run_metadata)
  time_elapsed = time.time() - start_time

  if run_metadata is not None:
    tl = timeline.Timeline(run_metadata.step_stats)
    trace = tl.generate_chrome_trace_format()
    trace_filename = os.path.join(train_step_kwargs['logdir'],
                                  'tf_trace-%d.json' % np_global_step)
    logging.info('Writing trace to %s', trace_filename)
    file_io.write_string_to_file(trace_filename, trace)
    if 'summary_writer' in train_step_kwargs:
      train_step_kwargs['summary_writer'].add_run_metadata(run_metadata,
                                                           'run_metadata-%d' %
                                                           np_global_step)

  if 'should_log' in train_step_kwargs:
    if sess.run(train_step_kwargs['should_log']):
      logging.info('global step %d: loss = %.4f (%.2f sec/step)',
                   np_global_step, total_loss, time_elapsed)

  # TODO(nsilberman): figure out why we can't put this into sess.run. The
  # issue right now is that the stop check depends on the global step. The
  # increment of global step often happens via the train op, which used
  # created using optimizer.apply_gradients.
  #
  # Since running `train_op` causes the global step to be incremented, one
  # would expected that using a control dependency would allow the
  # should_stop check to be run in the same session.run call:
  #
  #   with ops.control_dependencies([train_op]):
  #     should_stop_op = ...
  #
  # However, this actually seems not to work on certain platforms.
  if 'should_stop' in train_step_kwargs:
    should_stop = sess.run(train_step_kwargs['should_stop'])
  else:
    should_stop = False

  return total_loss, should_stop


_USE_DEFAULT = 0

def mytrain(train_op,
          logdir,
          train_step_fn=train_step,
          train_step_kwargs=_USE_DEFAULT,
          log_every_n_steps=1,
          graph=None,
          master='',
          is_chief=True,
          global_step=None,
          number_of_steps=None,
          init_op=_USE_DEFAULT,
          init_feed_dict=None,
          local_init_op=_USE_DEFAULT,
          init_fn=None,
          ready_op=_USE_DEFAULT,
          summary_op=_USE_DEFAULT,
          save_summaries_secs=600,
          summary_writer=_USE_DEFAULT,
          startup_delay_steps=0,
          saver=None,
          save_interval_secs=600,
          sync_optimizer=None,
          session_config=None,
          trace_every_n_steps=None):
  if train_op is None:
    raise ValueError('train_op cannot be None.')

  if logdir is None:
    if summary_op != _USE_DEFAULT:
      raise ValueError('Cannot provide summary_op because logdir=None')
    if saver is not None:
      raise ValueError('Cannot provide saver because logdir=None')
    if trace_every_n_steps is not None:
      raise ValueError('Cannot provide trace_every_n_steps because '
                       'logdir=None')

  if sync_optimizer is not None and startup_delay_steps > 0:
    raise ValueError(
        'startup_delay_steps must be zero when sync_optimizer is supplied.')

  if number_of_steps is not None and number_of_steps <= 0:
    raise ValueError(
        '`number_of_steps` must be either None or a positive number.')

  graph = graph or ops.get_default_graph()
  with graph.as_default():
    if global_step is None:
      global_step = variables.get_or_create_global_step()
    saver = saver or tf_saver.Saver()

    with ops.name_scope('init_ops'):
      if init_op == _USE_DEFAULT:
        init_op = tf_variables.global_variables_initializer()

      if ready_op == _USE_DEFAULT:
        ready_op = tf_variables.report_uninitialized_variables()

      if local_init_op == _USE_DEFAULT:
        local_init_op = control_flow_ops.group(
            tf_variables.local_variables_initializer(),
            data_flow_ops.initialize_all_tables())

      if sync_optimizer is not None and isinstance(
          sync_optimizer, sync_replicas_optimizer.SyncReplicasOptimizer):
        with ops.control_dependencies([local_init_op] if local_init_op is
                                      not None else []):
          if is_chief:
            local_init_op = sync_optimizer.chief_init_op
          else:
            local_init_op = sync_optimizer.local_step_init_op
        ready_for_local_init_op = sync_optimizer.ready_for_local_init_op
      else:
        ready_for_local_init_op = None

    if summary_op == _USE_DEFAULT:
      summary_op = summary.merge_all()

    if summary_writer == _USE_DEFAULT:
      summary_writer = supervisor.Supervisor.USE_DEFAULT

    cleanup_op = None

    if is_chief and sync_optimizer is not None:
      if not isinstance(sync_optimizer,
                        (sync_replicas_optimizer.SyncReplicasOptimizer)):
        raise ValueError(
            '`sync_optimizer` must be a tf.train.SyncReplicasOptimizer.')

      # Need to create these BEFORE the supervisor finalizes the graph:
      init_tokens_op = sync_optimizer.get_init_tokens_op()
      chief_queue_runner = sync_optimizer.get_chief_queue_runner()
      if isinstance(sync_optimizer,
                    sync_replicas_optimizer.SyncReplicasOptimizer):
        cleanup_op = sync_optimizer.get_clean_up_op()

    if train_step_kwargs == _USE_DEFAULT:
      with ops.name_scope('train_step'):
        train_step_kwargs = {}

        if number_of_steps:
          should_stop_op = math_ops.greater_equal(global_step, number_of_steps)
        else:
          should_stop_op = constant_op.constant(False)
        train_step_kwargs['should_stop'] = should_stop_op
        train_step_kwargs['should_log'] = math_ops.equal(
            math_ops.mod(global_step, log_every_n_steps), 0)
        if is_chief and trace_every_n_steps is not None:
          train_step_kwargs['should_trace'] = math_ops.equal(
              math_ops.mod(global_step, trace_every_n_steps), 0)
          train_step_kwargs['logdir'] = logdir

  sv = supervisor.Supervisor(
      graph=graph,
      is_chief=is_chief,
      logdir=logdir,
      init_op=init_op,
      init_feed_dict=init_feed_dict,
      local_init_op=local_init_op,
      ready_for_local_init_op=ready_for_local_init_op,
      ready_op=ready_op,
      summary_op=summary_op,
      summary_writer=summary_writer,
      global_step=global_step,
      saver=saver,
      save_summaries_secs=save_summaries_secs,
      save_model_secs=save_interval_secs,
      init_fn=init_fn)

  if summary_writer is not None:
    train_step_kwargs['summary_writer'] = sv.summary_writer

  should_retry = True
  while should_retry:
    try:
      should_retry = False
      with sv.managed_session(
          master, start_standard_services=False, config=session_config) as sess:
        logging.info('Starting Session.')
        if is_chief:
          if logdir:
            sv.start_standard_services(sess)
        elif startup_delay_steps > 0:
          _wait_for_step(sess, global_step,
                         min(startup_delay_steps, number_of_steps or
                             sys.maxint))
        sv.start_queue_runners(sess)
        logging.info('Starting Queues.')
        if is_chief and sync_optimizer is not None:
          sv.start_queue_runners(sess, [chief_queue_runner])
          sess.run(init_tokens_op)
        try:
          try:
            while not sv.should_stop():
              
              total_loss, should_stop = train_step_fn(sess, train_op, global_step,
                                                      train_step_kwargs)
              if should_stop:
                logging.info('Stopping Training.')
                break
          except errors.OutOfRangeError:
            # OutOfRangeError is thrown when epoch limit per
            # tf.train.limit_epochs is reached.
            logging.info('Caught OutOfRangeError. Stopping Training.')
          if logdir and sv.is_chief:
            logging.info('Finished training! Saving model to disk.')
            sv.saver.save(sess, sv.save_path, global_step=sv.global_step)
        except:
          if sv.is_chief and cleanup_op is not None:
            logging.info('About to execute sync_clean_up_op!')
            sess.run(cleanup_op)
          raise

    except errors.AbortedError:
      # Always re-run on AbortedError as it indicates a restart of one of the
      # distributed tensorflow servers.
      logging.info('Retrying training!')
      should_retry = True

  return total_loss
