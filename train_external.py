import functools
import os
import time
from absl import app
from absl import flags
from absl import logging
import json

from flax import jax_utils
from flax import nn
from flax import optim
from flax import serialization

from flax.metrics import tensorboard
from flax.training import checkpoints
from flax.training import common_utils
import jax
from jax import lax
from jax import random
import jax.nn
import jax.numpy as jnp
#from flax_nlp.nn import recurrent
import numpy as onp


import numpy as np

import tensorflow.compat.v2 as tf

import input_pipeline

FLAGS = flags.FLAGS

flags.DEFINE_string(
    'model_dir',
    default='models/ttl=30d/eeg/',
    help=('Directory for model data'))
flags.DEFINE_integer(
    'batch_size', default=32, help=('Batch size for training.'))
flags.DEFINE_integer(
    'eval_frequency',
    default=125,
    help=('Frequency of eval during training, e.g. every 1000 steps.'))
flags.DEFINE_integer(
    'num_train_steps', default=200000, help=('Number of train steps.'))

flags.DEFINE_float('learning_rate', default=0.01, help=('Learning rate.'))
flags.DEFINE_float(
    'weight_decay',
    default=0.1,
    help=('decay factor for AdamW style weight decay.'))
flags.DEFINE_integer(
    'offset', default=0, help=('decay factor for AdamW style weight decay.'))

flags.DEFINE_integer(
    'max_target_length',
    default=176,
    help=('maximum length of training examples.'))

flags.DEFINE_enum('classifier_type', 'transformer', ['transformer', 'lstm'],
                  'The classifier used.')


flags.DEFINE_string('n_avg_train', default='10_', help='avg train')
flags.DEFINE_string('n_avg_test', default='10_', help='avg test')
flags.DEFINE_string('what', default='alex', help='experiments')
flags.DEFINE_string(
    'data_path',
    'content/drive/My Drive/Bernd_EEG/data',
    help='data path')
flags.DEFINE_integer('step_size', default=2, help=('step size for the window.'))

flags.DEFINE_integer('random_seed', default=0, help=('random seed.'))

flags.DEFINE_integer(
    'window_size', default=30, help=('window size for the window.'))


SWITCH_ACC = False


@functools.partial(jax.jit, static_argnums=(1, 2, 3))
def create_model(key, input_shape, model_kwargs, cl_type):
  """Crate model."""
  print('creating model', cl_type)
  if cl_type == 'transformer':
    model_def = Transformer.partial(train=False, **model_kwargs)
  elif cl_type == 'lstm':
    print('creating lstm model')
    model_def = LSTM.partial(train=False, **model_kwargs)
  else:
    raise f'Not defined classifier type {cl_type}'

  _, params = model_def.init_by_shape(key, [input_shape])
  model = nn.Model(model_def, params)
  return model


class LSTM(nn.Module):
  """LSTM Model for eeg data."""

  def apply(self,
            inputs,
            output_vocab_size,
            num_layers=2,
            max_len=2048,
            train=True,
            dropout_rate=0.3):
    """Applies LSTM model on the inputs.

    Args:
      inputs: input data
      output_vocab_size: size of the output classes
      num_layers: number of layers
      max_len: maximum length.
      train: if it is training,
      dropout_rate: dropout rate

    Returns:
      the logits of the clasification per eeg time step.

    """
    assert inputs.ndim == 3

    x = inputs.astype('float32')
    x = nn.dropout(x, rate=dropout_rate, deterministic=not train)

    lens = jnp.full((x.shape[0]), max_len)
    x, _ = recurrent.LSTM(
        x,
        lens,
        hidden_size=300,
        num_layers=num_layers,
        bidirectional=True,
        dropout_rate=0,
        recurrent_dropout_rate=dropout_rate,
        train=train)
    x = nn.dropout(x, rate=dropout_rate, deterministic=not train)

    x = nn.LayerNorm(x)

    logits = nn.Dense(
        x,
        output_vocab_size,
        kernel_init=nn.initializers.xavier_uniform(),
        bias_init=nn.initializers.normal(stddev=1e-6))
    return logits





def shard(xs):
  local_device_count = jax.local_device_count()
  return jax.tree_map(
      lambda x: x.reshape((local_device_count, -1) + x.shape[1:]), xs)


def shard_prng_key(prng_key):
  # PRNG keys can used at train time to drive stochastic modules
  # e.g. DropOut. We would like a different PRNG key for each local
  # device so that we end up with different random numbers on each one,
  # hence we split our PRNG key and put the resulting keys into the batch
  return jax.random.split(prng_key, num=jax.local_device_count())


def onehot(labels, num_classes):
  x = (labels[..., None] == jnp.arange(num_classes)[None])
  return x.astype(jnp.float32)


def pmean(tree, axis_name='batch'):
  num_devices = lax.psum(1., axis_name)
  return jax.tree_map(lambda x: lax.psum(x, axis_name) / num_devices, tree)


def psum(tree, axis_name='batch'):
  return jax.tree_map(lambda x: lax.psum(x, axis_name), tree)


def stack_forest(forest):
  stack_args = lambda *args: onp.stack(args)
  return jax.tree_multimap(stack_args, *forest)


def get_metrics(device_metrics):
  device_metrics = jax.tree_map(lambda x: x[0], device_metrics)
  metrics_np = jax.device_get(device_metrics)
  return stack_forest(metrics_np)


"""Transformer-based langauge models."""

#from flax import nn
#import jax.numpy as jnp
#import numpy as np


class Embed(nn.Module):
  """Embedding Module.

  A parameterized function from integers [0, n) to d-dimensional vectors.
  """

  def apply(self,
            inputs,
            num_embeddings,
            features,
            mode='input',
            emb_init=nn.initializers.normal(stddev=1.0)):
    """Applies Embed module.

    Args:
      inputs: input data
      num_embeddings: number of embedding
      features: size of the embedding dimension
      mode: either 'input' or 'output' -> to share input/output embedding
      emb_init: embedding initializer

    Returns:
      output which is embedded input data
    """
    embedding = self.param('embedding', (num_embeddings, features), emb_init)
    if mode == 'input':
      if inputs.dtype not in [jnp.int32, jnp.int64, jnp.uint32, jnp.uint64]:
        raise ValueError('Input type must be an integer or unsigned integer.')
      return jnp.take(embedding, inputs, axis=0)
    if mode == 'output':
      return jnp.einsum('bld,vd->blv', inputs, embedding)


def sinusoidal_init(max_len=2048):
  """1D Sinusoidal Position Embedding Initializer.

  Args:
      max_len: maximum possible length for the input

  Returns:
      output: init function returning `(1, max_len, d_feature)`
  """

  def init(key, shape, dtype=np.float32):
    """Sinusoidal init."""
    del key, dtype
    d_feature = shape[-1]
    print(f'd_feature / shape[-1] is {d_feature}')

    pe = np.zeros((max_len, d_feature), dtype=np.float32)

    position = np.arange(0, max_len)[:, np.newaxis]
    div_term = np.exp(
        np.arange(0, d_feature, 2) * -(np.log(10000.0) / d_feature))
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)
    pe = pe[np.newaxis, :, :]  # [1, max_len, d_feature]
    return jnp.array(pe)

  return init


class AddPositionEmbs(nn.Module):
  """Adds learned positional embeddings to the inputs."""

  def apply(self,
            inputs,
            max_len=2048,
            posemb_init=nn.initializers.normal(stddev=1.0)):
    """Applies AddPositionEmbs module.

    Args:
      inputs: input data
      max_len: maximum possible length for the input
      posemb_init: positional embedding initializer

    Returns:
      output: `(bs, timesteps, in_dim)`
    """
    assert inputs.ndim == 3, ('Number of dimention should be 3, but it is: %d' %
                              inputs.ndim)
    length = inputs.shape[1]
    pos_emb_shape = (1, max_len, inputs.shape[-1])
    pos_embedding = self.param('pos_embedding', pos_emb_shape, posemb_init)
    return inputs + pos_embedding[:, :length, :]


class MlpBlock(nn.Module):
  """Transformer MLP block."""

  def apply(self,
            inputs,
            mlp_dim,
            out_dim=None,
            dropout_rate=0.3,
            deterministic=False,
            kernel_init=nn.initializers.xavier_uniform(),
            bias_init=nn.initializers.normal(stddev=1e-6)):
    """Applies Transformer MlpBlock module."""
    actual_out_dim = inputs.shape[-1] if out_dim is None else out_dim
    x = nn.Dense(inputs, mlp_dim, kernel_init=kernel_init, bias_init=bias_init)
    x = nn.gelu(x)
    x = nn.dropout(x, rate=dropout_rate, deterministic=deterministic)
    output = nn.Dense(
        x, actual_out_dim, kernel_init=kernel_init, bias_init=bias_init)
    output = nn.dropout(output, rate=dropout_rate, deterministic=deterministic)
    return output


class Transformer1DBlock(nn.Module):
  """Transformer layer (https://openreview.net/forum?id=H1e5GJBtDr)."""

  def apply(self,
            inputs,
            qkv_dim,
            mlp_dim,
            num_heads,
            causal_mask=False,
            padding_mask=None,
            dropout_rate=0.3,
            attention_dropout_rate=0.3,
            deterministic=False,
            attention=True):
    """Applies Transformer1DBlock module.

    Args:
      inputs: input data
      qkv_dim: dimension of the query/key/value
      mlp_dim: dimension of the mlp on top of attention block
      num_heads: number of heads
      causal_mask: bool, mask future or not
      padding_mask: bool, mask padding tokens
      dropout_rate: dropout rate
      attention_dropout_rate: dropout rate for attention weights
      deterministic: bool, deterministic or not (to apply dropout)

    Returns:
      output after transformer block.

    """

    # Attention block.
    assert inputs.ndim == 3
    x = inputs
    if attention:
      x = nn.LayerNorm(x)
      x = nn.SelfAttention(
          x,
          num_heads=num_heads,
          qkv_features=qkv_dim,
          attention_axis=(1,),
          causal_mask=causal_mask,
          padding_mask=padding_mask,
          kernel_init=nn.initializers.xavier_uniform(),
          bias_init=nn.initializers.normal(stddev=1e-6),
          bias=False,
          broadcast_dropout=False,
          dropout_rate=attention_dropout_rate,
          deterministic=deterministic)
      x = nn.dropout(x, rate=dropout_rate, deterministic=deterministic)
      x = x + inputs

    # MLP block.
    y = nn.LayerNorm(x)
    y = MlpBlock(
        y,
        mlp_dim=mlp_dim,
        dropout_rate=dropout_rate,
        deterministic=deterministic)

    return x + y


class Transformer(nn.Module):
  """Transformer Model for sequence tagging."""

  def apply(self,
            inputs,
            vocab_size,
            output_vocab_size,
            emb_dim=512,
            num_heads=8,
            num_layers=6,
            qkv_dim=512,
            mlp_dim=2048,
            max_len=2048,
            train=True,
            dropout_rate=0.1,
            attention_dropout_rate=0.1,
            ablation=''):
    """Applies Transformer model on the inputs.

    Args:
      inputs: input data
      vocab_size: size of the input vocabulary
      output_vocab_size: size of the output classes
      emb_dim: dimension of embedding
      num_heads: number of heads
      num_layers: number of layers
      qkv_dim: dimension of the query/key/value
      mlp_dim: dimension of the mlp on top of attention block
      max_len: maximum length.
      train: if it is training,
      dropout_rate: dropout rate
      attention_dropout_rate: dropout rate for attention weights

    Returns:
      output of a transformer decoder.

    """
    print('inputs.shape', inputs.shape)

    x = inputs
    x = x.astype('float32')
    x = nn.dropout(x, rate=dropout_rate, deterministic=not train)
    x = AddPositionEmbs(
        x, max_len=max_len, posemb_init=sinusoidal_init(max_len=max_len))
    if 'dense' in ablation:

      x = nn.Dense(x, mlp_dim, kernel_init=nn.initializers.xavier_uniform(),
                   bias_init=nn.initializers.normal(stddev=1e-6))
      x = nn.dropout(x, rate=dropout_rate, deterministic=not train)
    else:
      attention = True
      if 'noatt' in ablation:
        attention = False

      for _ in range(num_layers):
        x = Transformer1DBlock(
            x,
            qkv_dim=qkv_dim,
            mlp_dim=mlp_dim,
            num_heads=num_heads,
            causal_mask=False,
            padding_mask=None,
            dropout_rate=dropout_rate,
            attention_dropout_rate=attention_dropout_rate,
            deterministic=not train,
            attention=attention)
    x = nn.LayerNorm(x)
    logits = nn.Dense(
        x,
        output_vocab_size,
        kernel_init=nn.initializers.xavier_uniform(),
        bias_init=nn.initializers.normal(stddev=1e-6))
    time_dim = 1
    num_time_steps = logits.shape[time_dim]
    logits = jnp.sum(logits, axis=time_dim)
    logits = logits / num_time_steps
    return logits


def pad_examples(x, desired_batch_size):
  """Expand batch to desired size by repeating last slice."""
  print('pad_examples shape', x.shape)
  batch_pad = desired_batch_size - x.shape[0]
  return np.concatenate([x, np.tile(x[-1], (batch_pad, 1, 1))], axis=0)


def pad_target(x, desired_batch_size):
  """Expand batch to desired size by repeating last slice."""
  print('target shape', x.shape)
  batch_pad = desired_batch_size - x.shape[0]
  return np.concatenate([x, np.tile(x[-1], (batch_pad))], axis=0)


"""Sequence Tagging example.

This script trains a Transformer on the Universal dependency dataset.
"""

# check if this is still used.
N_WINDOW = 176  #  201 # 151 # 40
N_EMBEDDING = 64
N_AVG_TRAIN = '3_'
N_AVG_TEST = '3_'
N_CLASS = 6

SWITCH_ACC = False
best_acc = 0.0

FLAGS = flags.FLAGS


def create_model(key, input_shape, model_kwargs):
  module = Transformer.partial(train=False, **model_kwargs)

  @jax.jit
  def init(key):
    _, initial_params = module.init_by_shape(key, [(input_shape, jnp.float32)])
    model = nn.Model(module, initial_params)
    return model

  return init(key)


def create_optimizer(model, learning_rate):
  optimizer_def = optim.Adam(
      learning_rate, beta1=0.9, beta2=0.98, eps=1e-9, weight_decay=0)
  optimizer = optimizer_def.create(model)
  optimizer = optimizer.replicate()
  return optimizer


def create_learning_rate_scheduler(
    factors='constant * linear_warmup * rsqrt_decay',
    base_learning_rate=0.5,
    warmup_steps=50000,  # 25000
    decay_factor=0.5,
    steps_per_decay=25000,
    steps_per_cycle=100000):
  """creates learning rate schedule.

  Interprets factors in the factors string which can consist of:
  * constant: interpreted as the constant value,
  * linear_warmup: interpreted as linear warmup until warmup_steps,
  * rsqrt_decay: divide by square root of max(step, warmup_steps)
  * decay_every: Every k steps decay the learning rate by decay_factor.
  * cosine_decay: Cyclic cosine decay, uses steps_per_cycle parameter.

  Args:
    factors: a string with factors separated by '*' that defines the schedule.
    base_learning_rate: float, the starting constant for the lr schedule.
    warmup_steps: how many steps to warm up for in the warmup schedule.
    decay_factor: The amount to decay the learning rate by.
    steps_per_decay: How often to decay the learning rate.
    steps_per_cycle: Steps per cycle when using cosine decay.

  Returns:
    a function learning_rate(step): float -> {'learning_rate': float}, the
    step-dependent lr.
  """
  factors = [n.strip() for n in factors.split('*')]

  def step_fn(step):
    """Step to learning rate function."""
    ret = 1.0
    for name in factors:
      if name == 'constant':
        ret *= base_learning_rate
      elif name == 'linear_warmup':
        ret *= jnp.minimum(1.0, step / warmup_steps)
      elif name == 'rsqrt_decay':
        ret /= jnp.sqrt(jnp.maximum(step, warmup_steps))
      elif name == 'rsqrt_normalized_decay':
        ret *= jnp.sqrt(warmup_steps)
        ret /= jnp.sqrt(jnp.maximum(step, warmup_steps))
      elif name == 'decay_every':
        ret *= (decay_factor**(step // steps_per_decay))
      elif name == 'cosine_decay':
        progress = jnp.maximum(0.0,
                               (step - warmup_steps) / float(steps_per_cycle))
        ret *= jnp.maximum(0.0,
                           0.5 * (1.0 + jnp.cos(jnp.pi * (progress % 1.0))))
      else:
        raise ValueError('Unknown factor %s.' % name)
    return jnp.asarray(ret, dtype=jnp.float32)

  return step_fn


def compute_weighted_cross_entropy(logits, targets, weights=None):
  """Compute weighted cross entropy and entropy for log probs and targets.

  Args:
   logits: [batch, length, num_classes] float array.
   targets: categorical targets [batch, length] int array.
   weights: None or array of shape [batch x length]

  Returns:
    Tuple of scalar loss and batch normalizing factor.
  """


  if logits.ndim == 3:
    num_classes = logits.shape[-1]
    onehot_targets = onehot(targets, num_classes)

    # Now, onehot_targets and logits should be completely matching now in shape
    # Both should be 3-dimensional of the form [batch, length, num_classes]

    log_softmax_targets = nn.log_softmax(logits)
    onehot_softmax = onehot_targets * log_softmax_targets

    loss = -jnp.sum(onehot_softmax, axis=-1)
    normalizing_factor = onehot_targets.sum()

  if logits.ndim == 2:
    num_classes = logits.shape[-1]
    #targets = targets[:,0:1]
    onehot_targets = onehot(targets, num_classes)

    log_softmax_targets = nn.log_softmax(logits)
    onehot_softmax = onehot_targets * log_softmax_targets
    loss = -jnp.sum(onehot_softmax, axis=-1)
    normalizing_factor = onehot_targets.sum()
  if weights is not None:
    loss = loss * weights
    normalizing_factor = weights.sum()
  return loss.sum(), normalizing_factor


def compute_weighted_accuracy(logits, targets, weights=None):
  """Compute weighted accuracy for log probs and targets.

  Args:
   logits: [batch, length, num_classes] float array.
   targets: categorical targets [batch, length] int array.
   weights: None or array of shape [batch x length]

  Returns:
    Tuple of scalar accuracy and batch normalizing factor.
  """

  total = jnp.argmax(logits, axis=-1)
  loss = jnp.equal(total, targets)


  normalizing_factor = np.prod(logits.shape[:-1])

  return loss.sum(), normalizing_factor


def compute_metrics(logits, labels, weights):
  """Compute summary metrics."""
  loss, weight_sum = compute_weighted_cross_entropy(logits, labels, weights)
  acc, normalizing_factor = compute_weighted_accuracy(logits, labels, weights)
  metrics = {
      'loss': loss,
      'accuracy': acc,
      'denominator': weight_sum,
      'normalizing_factor': normalizing_factor,
  }
  metrics = psum(metrics)
  return metrics


def train_step(optimizer, batch, learning_rate_fn, dropout_rng=None):
  """Perform a single training step."""

  inputs, targets = batch
  print(f'inputs shape is {inputs.shape}')
  n_batch, n_time, n_embed = inputs.shape


  print(f'targets (2) shape is {targets.shape}')

  # warning set all to 1 for eeg data.
  weights = jnp.ones_like(targets).astype(jnp.float32)

  dropout_rng, new_dropout_rng = random.split(dropout_rng)

  def loss_fn(model):
    """Loss function used for training."""
    with nn.stochastic(dropout_rng):
      logits = model(inputs, train=True)
    loss, weight_sum = compute_weighted_cross_entropy(logits, targets, weights)
    mean_loss = loss / weight_sum
    return mean_loss, logits

  step = optimizer.state.step
  lr = learning_rate_fn(step)
  new_optimizer, _, logits = optimizer.optimize(loss_fn, learning_rate=lr)
  metrics = compute_metrics(logits, targets, weights)
  metrics['learning_rate'] = lr

  return new_optimizer, metrics, new_dropout_rng


def eval_step(model, batch):
  """Calculate evaluation metrics on a batch."""

  inputs, targets = batch
  n_batch, n_time, n_embed = inputs.shape

  weights = jnp.ones_like(targets).astype(jnp.float32)
  logits = model(inputs, train=False)
  argmax = jnp.argmax(logits, axis=-1)
  return compute_metrics(logits, targets, weights), argmax


# evaluate
def evaluate(eval_ds,
             num_eval_steps,
             best_acc,
             batch_size,
             p_eval_step,
             optimizer,
             step,
             train_accuracy,  # remove never used !
             summary_writer,
             eval_type='eval',
             eval_output=None,
             eval_output_force=False):

  eval_metrics = []
  eval_iter = iter(eval_ds)

  acc = 0
  total, xt = 0, 0
  argmax_batch = []
  targets_batch = []
  correct_wrong = []
  for eval_batch in eval_iter:

    cur_pred_batch_size = eval_batch[0].shape[0]

    if cur_pred_batch_size != batch_size and jax.device_count() > 1:

      desired_batch_size = jax.device_count()

      source = eval_batch[0].numpy()
      target = eval_batch[1].numpy()
      source_pad = pad_examples(source, jax.device_count())
      target_pad = pad_target(target, jax.device_count())
      padded = (source_pad, target_pad)

      eval_batch = padded

      eval_batch = shard(eval_batch)

      metrics, argmax = p_eval_step(optimizer.target, eval_batch)

      eval_metrics.append(metrics)
    else:
      eval_batch = shard(jax.tree_map(lambda x: x._numpy(), eval_batch))
      metrics, argmax = p_eval_step(optimizer.target, eval_batch)
      eval_metrics.append(metrics)

    argmax_batch.append(argmax)
    targets_batch.append(eval_batch[1])

  cnt = 0
  correct = 0
  class_dist = {}
  pred_dist = {}
  for batch, trg in zip(argmax_batch, targets_batch):
    for a, t in zip(batch, trg):
      for ax, tx in zip(a, t):
        single_pred = {}
        if isinstance(ax, np.int32):  # using a sum
          single_pred[ax] = single_pred.get(ax, 0) + 1
        else: # using voting
          for x in ax:
            single_pred[x] = single_pred.get(x, 0) + 1


        best = max(single_pred, key=single_pred.get)
        class_dist[tx] = class_dist.get(tx, 0) + 1
        pred_dist[best] = pred_dist.get(best, 0) + 1
        cnt += 1
        if best == tx:
          correct_wrong.append((1, best, tx))
          correct += 1
        else:
          correct_wrong.append((0, best, tx))

  if cnt != 0:
    accuracy = (correct / cnt)
  else:
    accuracy = 0

  print('cnt, correct', cnt, correct, accuracy)
  print('class_dist', class_dist)
  print('pred_dist', pred_dist)

  eval_metrics = get_metrics(eval_metrics)
  eval_metrics_sums = jax.tree_map(jnp.sum, eval_metrics)
  eval_denominator = eval_metrics_sums.pop('denominator')
  normalizing_factor = eval_metrics_sums.pop('normalizing_factor')

  logging.info('normalizing_factor %d, eval_denominator %d',
               normalizing_factor, eval_denominator)



  eval_summary = jax.tree_map(lambda x: x / eval_denominator, eval_metrics_sums)
  print('eval_summary', eval_summary)
  del eval_summary['loss']
  eval_summary['accuracy'] = accuracy

  print(f'step:{step}, {eval_type} accuracy: {accuracy:.2f}')

  # best best_acc is past
  if accuracy > best_acc:

    best_acc = accuracy
    print('wirte evals to ', eval_output)
    with tf.io.gfile.GFile(eval_output, mode='w') as f:
      for (c, b, t) in correct_wrong:
        out_str = str(c) + '\t' + str(b) + '\t' + str(t) + '\n'
        f.write(out_str)

  if eval_output_force:
    print('write evals to ', eval_output)
    with tf.io.gfile.GFile(eval_output, mode='w') as f:
      for (c, b, t) in correct_wrong:
        out_str = str(c) + '\t' + str(b) + '\t' + str(t) + '\n'
        f.write(out_str)

  logging.info('eval in step: %d accuracy: %.4f', step,
               eval_summary['accuracy'])

  for key, val in eval_summary.items():
    summary_writer.scalar(eval_type + '/' + key, val, step)
    summary_writer.flush()
  return best_acc


def train(offset=0,
          window_size=1,
          name='v1',
          what=None,
          random_seed=0,
          num_train_steps=400000,
          ablation=''):

  tf.enable_v2_behavior()

  print('random seed', random_seed)

  batch_size = 8  # not used anymore? see eval_batch_size !!
  learning_rate = 0.04
  num_eval_steps = -1
  eval_freq = 2000
  max_target_length = 176  #226 # 200 #176 #176

  eval_batch_size = 8  # multiplier of jax.device_count()

  N_EMBEDDING = 64
  N_CLASS = 6
  logging.info('eval_batch_size', eval_batch_size)
  logging.info('start offset', offset)
  logging.info('window_size', window_size)

  name += '_seed'+str(random_seed)

  logging.info('name', name)
  logging.info('random_seed', random_seed)

  logging.info('data what', what)


  #DATA_PATH  = 'data/eeg/bigram/10class/avg3/'
  #model_dir = 'models/ttl=720d/eeg_ablation/' + name + '/'
  if len(ablation) > 1:
    model_dir = 'models/ttl=720d/eeg_ablation/' + name + '/'
  else:
    model_dir = 'models/ttl=720d/eeg_server_windows_v2/' + name + '/'
  logging.info('model_dir', model_dir)
  tf.io.gfile.makedirs(model_dir)

  # test sample without bootstrapping.
  filename_pattern_test_wr = None
  if what == 'c10_a10_rev_v2':
    max_target_length = 226
    N_CLASS = 10
    DATA_PATH  = 'data/eeg/bigram/10class_revision_v2/avg10/'
    filename_pattern_test = f'bigram_10class_avg10_test_file*.tfrecords'
    filename_pattern_test_wr = f'bigram_10class_avg10_test_wr_file*.tfrecords'
    filename_pattern_train = f'bigram_10class_avg10_train_file*.tfrecords'
    filename_pattern_dev = f'bigram_10class_avg10_dev_file*.tfrecords'

  # For Alex to use
  elif what == "alex":
    max_target_length = 176
    N_CLASS = 6
    DATA_PATH  = '/content/drive/My Drive/Bernd_EEG/data/unigram_6class/verify_data/'
    filename_pattern_test = f'unigram_6class_avg10_test_small_file*.tfrecords'
    filename_pattern_test_wr = f'unigram_6class_avg10_test_wr_small_file*.tfrecords'
    filename_pattern_train = f'unigram_6class_avg10_train_small_file*.tfrecords'
    filename_pattern_dev = f'unigram_6class_avg10_dev_small_file*.tfrecords'

  elif what == 'c10_a3_rev_v2':
    max_target_length = 226
    N_CLASS = 10
    DATA_PATH  = 'data/eeg/bigram/10class_revision_v2/avg3/'
    filename_pattern_test = f'bigram_10class_avg3_test_file*.tfrecords'
    filename_pattern_test_wr = f'bigram_10class_avg3_test_wr_file*.tfrecords'
    filename_pattern_train = f'bigram_10class_avg3_train_file*.tfrecords'
    filename_pattern_dev = f'bigram_10class_avg3_dev_file*.tfrecords'
  elif what == 'c10_a1_rev_v2':
    max_target_length = 226
    N_CLASS = 10
    DATA_PATH  = 'data/eeg/bigram/10class_revision_v2/avg1/'
    filename_pattern_test = f'bigram_10class_avg1_test_file*.tfrecords'
    filename_pattern_test_wr = f'bigram_10class_avg1_test_file*.tfrecords'  # Same as test files as no bootstrapping was used.
    filename_pattern_train = f'bigram_10class_avg1_train_file*.tfrecords'
    filename_pattern_dev = f'bigram_10class_avg1_dev_file*.tfrecords'
  elif what == 'matched_6class_avg10':
    max_target_length = 176
    N_CLASS = 6
    DATA_PATH  = 'data/eeg/matched_6class_avg10/'
    filename_pattern_test = f'lexgram_prev_matched_6class_avg10_test_file*.tfrecords'
    filename_pattern_test_wr = f'lexgram_prev_matched_6class_avg10_test_wr_file*.tfrecords'
    filename_pattern_train = f'lexgram_prev_matched_6class_avg10_train_file*.tfrecords'
    filename_pattern_dev = f'lexgram_prev_matched_6class_avg10_dev_file*.tfrecords'
  elif what == 'matched_6class_avg3':
    max_target_length = 176
    N_CLASS = 6
    DATA_PATH  = 'data/eeg/matched_6class_avg3/'
    filename_pattern_test = f'lexgram_prev_matched_6class_avg3_test_file*.tfrecords'
    filename_pattern_test_wr = f'lexgram_prev_matched_6class_avg3_test_wr_file*.tfrecords'
    filename_pattern_train = f'lexgram_prev_matched_6class_avg3_train_file*.tfrecords'
    filename_pattern_dev = f'lexgram_prev_matched_6class_avg3_dev_file*.tfrecords'
  elif what == 'single_trial':
    max_target_length = 226
    DATA_PATH  = 'data/eeg/single_trial/'
    filename_pattern_test = f'bigram_10class_avg1_test_file*.tfrecords'
    filename_pattern_train = f'bigram_10class_avg1_train_file*.tfrecords'
    filename_pattern_dev = f'bigram_10class_avg1_dev_file*.tfrecords'
  elif what == 'matched_6class_avg1':
    max_target_length = 176
    N_CLASS = 6
    DATA_PATH = 'data/eeg/matched_6class_avg1/'
    filename_pattern_test = f'lexgram_prev_matched_6class_avg1_test_file*.tfrecords'
    filename_pattern_test_wr = f'lexgram_prev_matched_6class_avg1_test_file*.tfrecords'
    filename_pattern_train = f'lexgram_prev_matched_6class_avg1_train_file*.tfrecords'
    filename_pattern_dev = f'lexgram_prev_matched_6class_avg1_dev_file*.tfrecords'

  elif what == 'avg10_c10_revision':
    max_target_length = 226
    DATA_PATH= 'data/eeg/bigram/10class_revision/avg10/'
    filename_pattern_test = f'bigram_10class_avg10_test_file*.tfrecords'
    filename_pattern_train = f'bigram_10class_avg10_train_file*.tfrecords'
    filename_pattern_dev = f'bigram_10class_avg10_dev_file*.tfrecords'
  elif what == 'unmatched_6class_avg1':
    DATA_PATH= 'data/eeg/lexgram_unmatched_6class_avg1/'
    max_target_length = 176
    N_CLASS = 6
    filename_pattern_test =    'lexgram_unmatched_6class_single_trials_test_file*.tfrecords'
    filename_pattern_test_wr = 'lexgram_unmatched_6class_single_trials_test_file*.tfrecords'
    filename_pattern_train =   'lexgram_unmatched_6class_single_trials_train_file*.tfrecords'
    filename_pattern_dev =     'lexgram_unmatched_6class_single_trials_dev_file*.tfrecords'
  elif what == 'unmatched_6class_avg3':
    DATA_PATH= 'data/eeg/lexgram_unmatched_6class_avg3/'
    max_target_length = 176
    N_CLASS = 6
    filename_pattern_test = f'lexgram_unmatched_6class_avg3_test_file*.tfrecords'
    filename_pattern_test_wr = f'lexgram_unmatched_6class_avg3_test_wr_file*.tfrecords'
    filename_pattern_train = f'lexgram_unmatched_6class_avg3_train_file*.tfrecords'
    filename_pattern_dev = f'lexgram_unmatched_6class_avg3_dev_file*.tfrecords'
  elif what == 'unmatched_6class_avg10':
    DATA_PATH= 'data/eeg/lexgram_unmatched_6class_avg10/'
    max_target_length = 176
    N_CLASS = 6
    filename_pattern_test = f'lexgram_unmatched_6class_avg10_test_file*.tfrecords'
    filename_pattern_test_wr = f'lexgram_unmatched_6class_avg10_test_wr_file*.tfrecords'
    filename_pattern_train = f'lexgram_unmatched_6class_avg10_train_file*.tfrecords'
    filename_pattern_dev = f'lexgram_unmatched_6class_avg10_dev_file*.tfrecords'
  elif what == 'bigger_unmatched_6class_avg1':
    DATA_PATH= 'data/eeg/bigger_lexgram_unmatched_6class_avg1/'
    max_target_length = 176
    N_CLASS = 6  # 6class_single_trials_train_bigger
    filename_pattern_test = f'lexgram_unmatched_6class_single_trials_test_bigger_file*.tfrecords'
    filename_pattern_test_wr = f'lexgram_unmatched_6class_single_trials_test_bigger_file*.tfrecords'
    filename_pattern_train = f'lexgram_unmatched_6class_single_trials_train_bigger_file*.tfrecords'
    filename_pattern_dev = f'lexgram_unmatched_6class_single_trials_dev_bigger_file*.tfrecords'
  elif what == 'bigger_unmatched_6class_avg3':
    DATA_PATH= 'data/eeg/bigger_lexgram_unmatched_6class_avg3/'
    max_target_length = 176
    N_CLASS = 6
    filename_pattern_test = f'lexgram_unmatched_6class_avg3_test_bigger_file*.tfrecords'
    filename_pattern_test_wr = f'lexgram_unmatched_6class_avg3_test_wr_bigger_file*.tfrecords'
    filename_pattern_train = f'lexgram_unmatched_6class_avg3_train_bigger_file*.tfrecords'
    filename_pattern_dev = f'lexgram_unmatched_6class_avg3_dev_bigger_file*.tfrecords'
  elif what == 'bigger_unmatched_6class_avg10':
    DATA_PATH= 'data/eeg/bigger_lexgram_unmatched_6class_avg10/'
    max_target_length = 176
    N_CLASS = 6
    filename_pattern_test = f'lexgram_unmatched_6class_avg10_test_bigger_file*.tfrecords'
    filename_pattern_test_wr = f'lexgram_unmatched_6class_avg10_test_wr_bigger_file*.tfrecords'
    filename_pattern_train = f'lexgram_unmatched_6class_avg10_train_bigger_file*.tfrecords'
    filename_pattern_dev = f'lexgram_unmatched_6class_avg10_dev_bigger_file*.tfrecords'

  # wr unmatched
  elif what == 'wr_unmatched_6class_avg1':
    DATA_PATH= 'data/eeg/bigger_lexgram_unmatched_6class_avg1/'
    max_target_length = 176
    N_CLASS = 6  # 6class_single_trials_train_bigger
    filename_pattern_test = f'lexgram_unmatched_6class_single_trials_test_bigger_file*.tfrecords'
    filename_pattern_test_wr = f'lexgram_unmatched_6class_single_trials_test_bigger_file*.tfrecords'
    filename_pattern_train = f'lexgram_unmatched_6class_single_trials_train_bigger_file*.tfrecords'
    filename_pattern_dev = f'lexgram_unmatched_6class_single_trials_dev_bigger_file*.tfrecords'
  elif what == 'wr_unmatched_6class_avg3':
    DATA_PATH= 'data/eeg/bigger_lexgram_unmatched_6class_avg3/'
    max_target_length = 176
    N_CLASS = 6
    filename_pattern_test = f'lexgram_unmatched_6class_avg3_test_wr_bigger_file*.tfrecords'
    filename_pattern_test_wr = f'lexgram_unmatched_6class_avg3_test_wr_bigger_file*.tfrecords'
    #                          lexgram_unmatched_6class_avg3_train_wr_bigger_file0.tfrecords
    filename_pattern_train = f'lexgram_unmatched_6class_avg3_train_wr_bigger_file*.tfrecords'

    filename_pattern_dev = f'lexgram_unmatched_6class_avg3_dev_wr_bigger_file*.tfrecords'
  elif what == 'wr_unmatched_6class_avg10':
    DATA_PATH= 'data/eeg/bigger_lexgram_unmatched_6class_avg10/'
    max_target_length = 176
    N_CLASS = 6
    filename_pattern_test = f'lexgram_unmatched_6class_avg10_test_wr_bigger_file*.tfrecords'
    filename_pattern_test_wr = f'lexgram_unmatched_6class_avg10_test_wr_bigger_file*.tfrecords'
    filename_pattern_train = f'lexgram_unmatched_6class_avg10_train_wr_bigger_file*.tfrecords'
    filename_pattern_dev = f'lexgram_unmatched_6class_avg10_dev_wr_bigger_file*.tfrecords'

  elif what == 'from_bigram_6class_avg1':
    DATA_PATH= 'data/eeg/from_bigram_6class_avg1/'
    max_target_length = 226
    N_CLASS = 6  # 6class_single_trials_train_bigger
    pattern = 'from_bigram_6class_6class_avg1'
    filename_pattern_test = f'{pattern}_test_file*.tfrecords'
    filename_pattern_test_wr = f'{pattern}_test_file*.tfrecords'
    filename_pattern_train = f'{pattern}_train_file*.tfrecords'
    filename_pattern_dev = f'{pattern}_dev_file*.tfrecords'
  elif what == 'from_bigram_6class_avg3':
    DATA_PATH= 'data/eeg/from_bigram_6class_avg3/'
    max_target_length = 226
    N_CLASS = 6
    pattern = 'from_bigram_6class_6class_avg3'
    filename_pattern_test = f'{pattern}_test_file*.tfrecords'
    filename_pattern_test_wr = f'{pattern}_test_wr_file*.tfrecords'
    filename_pattern_train = f'{pattern}_train_file*.tfrecords'
    filename_pattern_dev = f'{pattern}_dev_file*.tfrecords'
  elif what == 'from_bigram_6class_avg10':
    DATA_PATH= 'data/eeg/from_bigram_6class_avg10/'
    max_target_length = 226
    N_CLASS = 6
    pattern = 'from_bigram_6class_6class_avg10'
    filename_pattern_test = f'{pattern}_test_file*.tfrecords'
    filename_pattern_test_wr = f'{pattern}_test_wr_file*.tfrecords'
    filename_pattern_train = f'{pattern}_train_file*.tfrecords'
    filename_pattern_dev = f'{pattern}_dev_file*.tfrecords'
  # new bigram set: with 6 classes
  elif what == 'bigram_6class_avg1_v2':
    DATA_PATH= 'data/eeg/bigram_6class_avg1_v2/'
    max_target_length = 226
    N_CLASS = 6  # 6class_single_trials_train_bigger
    pattern = 'bigram_6class_avg1'
    filename_pattern_test = f'{pattern}_test_file*.tfrecords'
    filename_pattern_test_wr = f'{pattern}_test_file*.tfrecords'
    filename_pattern_train = f'{pattern}_train_file*.tfrecords'
    filename_pattern_dev = f'{pattern}_dev_file*.tfrecords'
  elif what == 'bigram_6class_avg3_v2':
    DATA_PATH= 'data/eeg/bigram_6class_avg3_v2/'
    max_target_length = 226
    N_CLASS = 6
    pattern = 'bigram_6class_avg3'
    filename_pattern_test = f'{pattern}_test_file*.tfrecords'
    filename_pattern_test_wr = f'{pattern}_test_wr_file*.tfrecords'
    filename_pattern_train = f'{pattern}_train_file*.tfrecords'
    filename_pattern_dev = f'{pattern}_dev_file*.tfrecords'
  elif what == 'bigram_6class_avg10_v2':
    DATA_PATH= 'data/eeg/bigram_6class_avg10_v2/'
    max_target_length = 226
    N_CLASS = 6
    pattern = 'bigram_6class_avg10'
    filename_pattern_test = f'{pattern}_test_file*.tfrecords'
    filename_pattern_test_wr = f'{pattern}_test_wr_file*.tfrecords'
    filename_pattern_train = f'{pattern}_train_file*.tfrecords'
    filename_pattern_dev = f'{pattern}_dev_file*.tfrecords'

  elif what == 'bigram_6class_avg10_v3':
    DATA_PATH= 'data/eeg/bigram_6class_avg10_v3/'
    max_target_length = 326
    N_CLASS = 6
    pattern = 'bigram_6class_avg10'
    filename_pattern_test = f'{pattern}_test_file*.tfrecords'
    filename_pattern_test_wr = f'{pattern}_test_wr_file*.tfrecords'
    filename_pattern_train = f'{pattern}_train_file*.tfrecords'
    filename_pattern_dev = f'{pattern}_dev_file*.tfrecords'
  elif what == 'bigram_6class_avg10_rw_v4':
    DATA_PATH= 'data/eeg/bigram_6class_avg_v4/'
    max_target_length = 326
    N_CLASS = 6
    pattern = 'bigram_6class_avg10'
    filename_pattern_test = f'{pattern}_test_file*.tfrecords'
    filename_pattern_test_wr = f'{pattern}_test_wr_file*.tfrecords'
    filename_pattern_train = f'{pattern}_train_wr_file*.tfrecords'
    filename_pattern_dev = f'{pattern}_dev_file*.tfrecords'
  elif what == 'bigram_6class_train_a10_test_a1_v3':
    DATA_PATH= 'data/eeg/bigram_6class_avg_mix/'
    max_target_length = 226
    N_CLASS = 6
    pattern_train = 'bigram_6class_avg10'
    pattern_test = 'bigram_6class_avg1'
    filename_pattern_test = f'{pattern_test}_test_file*.tfrecords'
    filename_pattern_test_wr = f'{pattern_train}_test_file*.tfrecords'
    filename_pattern_train = f'{pattern_train}_train_file*.tfrecords'
    filename_pattern_dev = f'{pattern_test}_dev_file*.tfrecords'

  elif what == 'bigram_6class_train_a1_test_a1_v3':
    DATA_PATH= 'data/eeg/bigram_6class_avg_mix/'
    max_target_length = 226
    N_CLASS = 6
    pattern_train = 'bigram_6class_avg1'
    pattern_test = 'bigram_6class_avg1'
    pattern_test_10 = 'bigram_6class_avg10'
    filename_pattern_test = f'{pattern_test}_test_file*.tfrecords'
    filename_pattern_test_wr = f'{pattern_test_10}_test_file*.tfrecords'
    filename_pattern_train = f'{pattern_train}_train_file*.tfrecords'
    filename_pattern_dev = f'{pattern_test}_dev_file*.tfrecords'
  elif what == 'bigram_6class_train_a1_v0204':
    DATA_PATH= 'data/eeg/bigram_6class_avg1_v0204/'
    max_target_length = 226
    N_CLASS = 6
    pattern = 'bigram_6class_avg1'
    filename_pattern_test = f'{pattern}_test_noica_file*.tfrecords'
    filename_pattern_test_wr = f'{pattern}_test_noica_file*.tfrecords'
    filename_pattern_train = f'{pattern}_train_noica_file*.tfrecords'
    filename_pattern_dev = f'{pattern}_dev_noica_file*.tfrecords'
  elif what == 'bigram_6class_train_a10_v0204':
    DATA_PATH= 'data/eeg/bigram_6class_avg10_v0204/'
    max_target_length = 226
    N_CLASS = 6
    pattern = 'bigram_6class_avg10'
    filename_pattern_test = f'{pattern}_test_noica_file*.tfrecords'
    filename_pattern_test_wr = f'{pattern}_test_wr_noica_file*.tfrecords'
    filename_pattern_train = f'{pattern}_train_noica_file*.tfrecords'
    filename_pattern_dev = f'{pattern}_dev_noica_file*.tfrecords'
  elif what == 'bigram_6class_train_a3_v0204':
    DATA_PATH= 'data/eeg/bigram_6class_avg3_v0204/'
    max_target_length = 226
    N_CLASS = 6
    pattern = 'bigram_6class_avg3'
    filename_pattern_test = f'{pattern}_test_noica_file*.tfrecords'
    filename_pattern_test_wr = f'{pattern}_test_wr_noica_file*.tfrecords'
    filename_pattern_train = f'{pattern}_train_noica_file*.tfrecords'
    filename_pattern_dev = f'{pattern}_dev_noica_file*.tfrecords'
  elif what == 'bigram_6class_avg10_105k_v0208':
    DATA_PATH= f'data/eeg/{what}/'
    max_target_length = 226
    N_CLASS = 6
    pattern = 'bigram_6class_avg10'
    filename_pattern_test = f'{pattern}_test_noica_*.tfrecords'
    filename_pattern_test_wr = f'{pattern}_test_wr_noica*.tfrecords'
    filename_pattern_train = f'{pattern}_train_*.tfrecords'
    filename_pattern_dev = f'{pattern}_dev_*.tfrecords'
  elif what == 'bigram_6class_avg10_504k_v0208':
    DATA_PATH= f'data/eeg/{what}/'
    max_target_length = 226
    N_CLASS = 6
    pattern = 'bigram_6class_avg10'
    filename_pattern_test = f'{pattern}_test_noica_*.tfrecords'
    filename_pattern_test_wr = f'{pattern}_test_wr_noica*.tfrecords'
    filename_pattern_train = f'{pattern}_train_*.tfrecords'
    filename_pattern_dev = f'{pattern}_dev_*.tfrecords'
  elif what == 'bigram_6class_avg10_250k_v0208':
    DATA_PATH= f'data/eeg/{what}/'
    max_target_length = 226
    N_CLASS = 6
    pattern = 'bigram_6class_avg10'
    filename_pattern_test = f'{pattern}_test_noica_*.tfrecords'
    filename_pattern_test_wr = f'{pattern}_test_wr_noica*.tfrecords'
    filename_pattern_train = f'{pattern}_train_*.tfrecords'
    filename_pattern_dev = f'{pattern}_dev_*.tfrecords'
  elif what == 'bigram_6class_avg10_250k_v0401':
    DATA_PATH= f'data/eeg/{what}/'
    max_target_length = 176
    N_CLASS = 6
    pattern = 'unigram_6class_avg10'  #bigram_6class_avg10_aug_train_minica_
    filename_pattern_test = f'{pattern}_test_f*.tfrecords'
    filename_pattern_test_wr = f'{pattern}_test_wr_*.tfrecords'
    filename_pattern_train = f'{pattern}_train*.tfrecords'
    filename_pattern_dev = f'{pattern}_dev_*.tfrecords'
  elif what == 'bigram_6class_avg10_500k_v0401':
    DATA_PATH= f'data/eeg/{what}/'
    max_target_length = 176
    N_CLASS = 6
    pattern = 'unigram_6class_avg10'  #bigram_6class_avg10_aug_train_minica_
    filename_pattern_test = f'{pattern}_test_f*.tfrecords'
    filename_pattern_test_wr = f'{pattern}_test_wr_*.tfrecords'
    filename_pattern_train = f'{pattern}_train*.tfrecords'
    filename_pattern_dev = f'{pattern}_dev_*.tfrecords'
  elif what == 'bigram_6class_avg10_100k_v0401':
    DATA_PATH= f'data/eeg/{what}/'
    max_target_length = 176
    N_CLASS = 6
    pattern = 'unigram_6class_avg10'  #bigram_6class_avg10_aug_train_minica_
    filename_pattern_test = f'{pattern}_test_f*.tfrecords'
    filename_pattern_test_wr = f'{pattern}_test_wr_*.tfrecords'
    filename_pattern_train = f'{pattern}_train*.tfrecords'
    filename_pattern_dev = f'{pattern}_dev_*.tfrecords'
  elif what == 'pretrain_10_3_1':
    model_dir = 'models/ttl=720d/pretrain_10_3_1/' + name + '/'

    path_avg1 = 'data/eeg/pretrain/20210415/avg1/'
    path_avg10 = 'data/eeg/bigram_6class_avg10_250k_v0401/'   # it is unigram data despite the path
    path_avg3 = 'data/eeg/unigram_6class_avg3_250k_v0416/'
    # DATA_PATH= f'data/eeg/{what}/'
    max_target_length = 176
    N_CLASS = 6
    pattern = 'unigram_6class_avg10'  #bigram_6class_avg10_aug_train_minica_
    filename_pattern_test = f'{pattern}_test_f*.tfrecords'
    filename_pattern_test_wr = f'{pattern}_test_wr_*.tfrecords'
    filename_pattern_train = f'{pattern}_train*.tfrecords'
    filename_pattern_dev = f'{pattern}_dev_*.tfrecords'
  else:
    raise 'dataset not found for key ' + what


  summary_path = os.path.join(model_dir,
                              'eeg_' + str(offset) + '_' + str(window_size))
  print('summary_path', summary_path)
  summary_writer = tensorboard.SummaryWriter(summary_path)

  if batch_size % jax.device_count() > 0:
    raise ValueError('Batch size must be divisible by the number of devices')
  device_batch_size = batch_size // jax.device_count()

  print('start preparing training data')
  logging.info('data path: %s', DATA_PATH)
  logging.info('training path: %s', filename_pattern_train)
  logging.info('dev path: %s', filename_pattern_dev)
  logging.info('test path: %s', filename_pattern_test)
  logging.info('test_wr path: %s', filename_pattern_test_wr)

  train_ds = input_pipeline.get_dataset(
      filename_pattern=filename_pattern_train,
      data_path=DATA_PATH,
      batch_size=eval_batch_size,
      bucket_size=max_target_length,
      offset=offset,
      window_size=window_size)
  print('end preparing training data')
  print('start preparing dev data')
  train_ds_iter = iter(train_ds)

  eval_ds = input_pipeline.get_dataset(
      #filename_pattern=f'test_avg{N_AVG_TEST}*.tfrecords',
      #filename_pattern=f'class*test*avg{N_AVG_TEST}*.tfrecords',
      filename_pattern=filename_pattern_dev,  #'bigram_4class_avg3_dev_file*.tfrecords',  #f'frequency_avg{N_AVG_TRAIN}dev_file*_1D.tfrecords',
      data_path=DATA_PATH,
      batch_size=eval_batch_size,
      bucket_size=max_target_length,
      offset=offset,
      window_size=window_size,
      repeat=1)
  # read ones the entire data set in
  eval_iter = iter(eval_ds)
  cnt = 0
  eval_shapes = []
  labs = {}
  train_iter = iter(train_ds)


  print('compute eval buckets')
  test_ds = input_pipeline.get_dataset(
      filename_pattern=filename_pattern_test,
      data_path=DATA_PATH,
      batch_size=eval_batch_size,
      bucket_size=max_target_length,
      offset=offset,
      window_size=window_size,
      repeat=1)
  test_iter = iter(test_ds)

  if filename_pattern_test_wr:
    test_wr_ds = input_pipeline.get_dataset(
      filename_pattern=filename_pattern_test_wr,
      data_path=DATA_PATH,
      batch_size=eval_batch_size,
      bucket_size=max_target_length,
      offset=offset,
      window_size=window_size,
      repeat=1)
    test_wr_iter = iter(test_wr_ds)


  print('create model')
  bs = device_batch_size * jax.device_count()

  rng = random.PRNGKey(random_seed)
  rng, init_rng = random.split(rng)


  real_len = window_size - offset

  input_shape = (bs, real_len, 64)
  print(f'Input shape is: {input_shape}')


  transformer_kwargs = {
      'vocab_size': N_CLASS,
      'output_vocab_size': N_CLASS,
      'emb_dim': N_EMBEDDING,  # 512
      'num_heads': 8,  # 16
      'num_layers': 4,  # 6
      'qkv_dim': 512,  # 1024
      'mlp_dim': 1024,  # 2024
      'max_len': max_target_length,
      'ablation': ablation
  }
  if 'l1' in ablation:
    transformer_kwargs['num_layers'] = 1

  print('transformer_kwargs', transformer_kwargs)
  model = create_model(init_rng, tuple(input_shape), transformer_kwargs)
  #model = create_model(init_rng, tuple(input_shape))

  print('end create model')
  optimizer = create_optimizer(model, learning_rate)
  del model  # don't keep a copy of the initial model
  learning_rate_fn = create_learning_rate_scheduler(
      base_learning_rate=learning_rate)

  p_train_step = jax.pmap(
      functools.partial(train_step, learning_rate_fn=learning_rate_fn),
      axis_name='batch')
  p_eval_step = jax.pmap(eval_step, axis_name='batch')

  # We init the first set of dropout PRNG keys, but update it afterwards inside
  # the main pmap'd training update for performance.
  dropout_rngs = random.split(rng, jax.local_device_count())

  metrics_all = []
  tick = time.time()
  best_acc = 0.0
  new_best_test_wr_acc = 0.0
  print('start training')
  for step, batch in zip(range(num_train_steps), train_iter):

    cur_pred_batch_size = batch[0].shape[0]

    if cur_pred_batch_size != batch_size:
      logging.info('Uneven train batch size %d.', cur_pred_batch_size)

    batch = shard(jax.tree_map(lambda x: x._numpy(), batch))  # pylint: disable=protected-access

    optimizer, metrics, dropout_rngs = p_train_step(
        optimizer, batch, dropout_rng=dropout_rngs)
    metrics_all.append(metrics)

    if (step + 1) % eval_freq == 0:

      metrics_all = get_metrics(metrics_all)
      lr = metrics_all.pop('learning_rate').mean()
      metrics_sums = jax.tree_map(jnp.sum, metrics_all)
      denominator = metrics_sums.pop('denominator')


      normalizing_factor = metrics_sums.pop('normalizing_factor')
      accuracy = metrics_sums['accuracy'] / normalizing_factor
      train_accuracy = accuracy
      print(f'step:{step} train acc: {train_accuracy:.2f}')


      summary = jax.tree_map(lambda x: x / denominator, metrics_sums)  # pylint: disable=cell-var-from-loop
      summary['learning_rate'] = lr
      # Calculate (clipped) perplexity after averaging log-perplexities:
      summary['perplexity'] = jnp.clip(jnp.exp(summary['loss']), a_max=1.0e4)
      summary['accuracy'] = accuracy
      logging.info('train in step: %d, loss: %.4f', step, summary['loss'])
      if jax.host_id() == 0:
        tock = time.time()
        steps_per_sec = eval_freq / (tock - tick)
        tick = tock
        summary_writer.scalar('steps per second', steps_per_sec, step)
        for key, val in summary.items():
          summary_writer.scalar('train/' + key, val, step)
        #train_summary_writer.flush()
      # reset metric accumulation for next evaluation cycle.
      metrics_all = []
      eval_output = model_dir + 'dev_classification.txt'
      new_best_acc = evaluate(
          eval_ds,
          num_eval_steps,
          best_acc,
          eval_batch_size,
          p_eval_step,
          optimizer,
          step,
          train_accuracy,
          summary_writer,
          'eval',
          eval_output=eval_output)
      if best_acc < new_best_acc:
        best_acc = new_best_acc
        eval_output = model_dir + 'test_classification.txt'
        new_best_test_acc = evaluate(
            test_ds,
            num_eval_steps,
            0,
            eval_batch_size,
            p_eval_step,
            optimizer,
            step,
            train_accuracy,
            summary_writer,
            'test',
            eval_output=eval_output,
            eval_output_force=True)
        if filename_pattern_test_wr:
          eval_output = model_dir + 'test_wr.txt'
          new_best_test_wr_acc = evaluate(
              test_wr_ds,
              num_eval_steps,
              0,
              eval_batch_size,
              p_eval_step,
              optimizer,
              step,
              train_accuracy,
              summary_writer,
              'test',
              eval_output=eval_output,
              eval_output_force=True)
        summary_writer.scalar('eval/best_eval', best_acc, step)
        summary_writer.scalar('test/best_eval', new_best_test_acc, step)

  # write early stopping results:
  final_accuracy_path = model_dir + 'accuracy.txt'
  with tf.io.gfile.GFile(final_accuracy_path, mode='w') as f:
    f.write('best_dev:')
    f.write(str(best_acc))
    f.write('\n')
    f.write('best_test_acc:')
    f.write(str(new_best_test_acc))
    f.write('\n')
    f.write('best_test_wr_acc:')
    f.write(str(new_best_test_wr_acc))
    f.write('\n')

  return (best_acc, new_best_test_acc, new_best_test_wr_acc)

################################################################################
# This is for pretrain scenarios
#
#

def pretrain(offset=0,
          window_size=1,
          name='v1',
          what=None,
          random_seed=0,
          num_train_steps=400000,
          ablation=''):

  tf.enable_v2_behavior()

  print('random seed', random_seed)
  #logging.info()

  batch_size = 8  # 512 # not used but. eval_batch_size !!
  learning_rate = 0.04
  num_eval_steps = -1
  eval_freq = 2000
  max_target_length = 176  #226 # 200 #176 #176

  eval_batch_size = 8  # jax.device_count()

  N_EMBEDDING = 64
  N_CLASS = 6
  logging.info('eval_batch_size', eval_batch_size)
  logging.info('start offset', offset)
  logging.info('window_size', window_size)

  name += '_seed'+str(random_seed)

  logging.info('name', name)
  logging.info('random_seed', random_seed)

  logging.info('data what', what)

  path_prefix = ''


  if len(ablation) > 1:
    model_dir = path_prefix + 'models/ttl=720d/eeg_ablation/' + name + '/'
  else:
    model_dir = path_prefix + 'models/ttl=720d/eeg_server_windows_v2/' + name + '/'
  logging.info('model_dir', model_dir)


  # test sample without bootstrapping.
  filename_pattern_test_wr = None
  if what == 'bigram_6class_avg10_250k_v0401':
    DATA_PATH= f'data/eeg/{what}/'
    max_target_length = 176
    N_CLASS = 6
    pattern = 'unigram_6class_avg10'  #bigram_6class_avg10_aug_train_minica_
    filename_pattern_test = f'{pattern}_test_f*.tfrecords'
    filename_pattern_test_wr = f'{pattern}_test_wr_*.tfrecords'
    filename_pattern_train = f'{pattern}_train*.tfrecords'
    filename_pattern_dev = f'{pattern}_dev_*.tfrecords'
  elif what == 'bigram_6class_avg10_500k_v0401':
    DATA_PATH= f'data/eeg/{what}/'
    max_target_length = 176
    N_CLASS = 6
    pattern = 'unigram_6class_avg10'  #bigram_6class_avg10_aug_train_minica_
    filename_pattern_test = f'{pattern}_test_f*.tfrecords'
    filename_pattern_test_wr = f'{pattern}_test_wr_*.tfrecords'
    filename_pattern_train = f'{pattern}_train*.tfrecords'
    filename_pattern_dev = f'{pattern}_dev_*.tfrecords'
  elif what == 'bigram_6class_avg10_100k_v0401':
    DATA_PATH= f'data/eeg/{what}/'
    max_target_length = 176
    N_CLASS = 6
    pattern = 'unigram_6class_avg10'  #bigram_6class_avg10_aug_train_minica_
    filename_pattern_test = f'{pattern}_test_f*.tfrecords'
    filename_pattern_test_wr = f'{pattern}_test_wr_*.tfrecords'
    filename_pattern_train = f'{pattern}_train*.tfrecords'
    filename_pattern_dev = f'{pattern}_dev_*.tfrecords'
  elif what == 'pretrain_10_3_1':
    model_dir = 'models/ttl=720d/pretrain/' + name + '/'

    path_avg1 = 'data/eeg/pretrain/20210415/avg1/'
    path_avg10 = 'data/eeg/bigram_6class_avg10_250k_v0401/'   # it is unigram data despite the path
    path_avg3 = 'data/eeg/unigram_6class_avg3_250k_v0416/'

    paths = [path_avg10, path_avg3, path_avg1]
    patterns = ['unigram_6class_avg10', 'unigram_6class_avg3', 'unigram_6class_avg1']
    patterns_test = ['unigram_6class_avg1', 'unigram_6class_avg1', 'unigram_6class_avg1']
    avgs = ['10', '3', '1']

    max_target_length = 176
    N_CLASS = 6
    pattern = 'unigram_6class_avg10'
    filename_pattern_test = f'{pattern}_test_f*.tfrecords'
    # filename_pattern_test_wr = f'{pattern}_test_wr_*.tfrecords'
    filename_pattern_train = f'{pattern}_train*.tfrecords'
    filename_pattern_dev = f'{pattern}_dev_*.tfrecords'
  elif what == 'pretrain_10_3_1_b':
    model_dir = 'models/ttl=720d/pretrain/' + name + '/'

    path_avg1 = 'data/eeg/pretrain/20210415/avg1/'
    path_avg10 = 'data/eeg/bigram_6class_avg10_250k_v0401/'   # it is unigram data despite the path
    path_avg3 = 'data/eeg/unigram_6class_avg3_250k_v0416/'

    paths = [path_avg10, path_avg3, path_avg1]
    patterns = ['unigram_6class_avg10', 'unigram_6class_avg3', 'unigram_6class_avg1']
    patterns_test = ['unigram_6class_avg10', 'unigram_6class_avg3', 'unigram_6class_avg1']
    avgs = ['10', '3', '1']

    max_target_length = 176
    N_CLASS = 6
    pattern = 'unigram_6class_avg10'
    filename_pattern_test = f'{pattern}_test_f*.tfrecords'
    # filename_pattern_test_wr = f'{pattern}_test_wr_*.tfrecords'
    filename_pattern_train = f'{pattern}_train*.tfrecords'
    filename_pattern_dev = f'{pattern}_dev_*.tfrecords'
  elif what == 'pretrain_1':
    model_dir = 'models/ttl=720d/pretrain/' + name + '/'

    path_avg1 = 'data/eeg/pretrain/20210415/avg1/'
    path_avg10 = 'data/eeg/bigram_6class_avg10_250k_v0401/'   # it is unigram data despite the path
    path_avg3 = 'data/eeg/unigram_6class_avg3_250k_v0416/'

    paths = [path_avg1]
    patterns = ['unigram_6class_avg1']
    patterns_test = ['unigram_6class_avg1']
    avgs = ['1']

    max_target_length = 176
    N_CLASS = 6
    # pattern = 'unigram_6class_avg10'
    # filename_pattern_test = f'{pattern}_test_f*.tfrecords'
    # # filename_pattern_test_wr = f'{pattern}_test_wr_*.tfrecords'
    # filename_pattern_train = f'{pattern}_train*.tfrecords'
    # filename_pattern_dev = f'{pattern}_dev_*.tfrecords'
  elif what == 'pretrain_3_1':
    model_dir = 'models/ttl=720d/pretrain/' + name + '/'

    path_avg1 = 'data/eeg/pretrain/20210415/avg1/'
    path_avg10 = 'data/eeg/bigram_6class_avg10_250k_v0401/'   # it is unigram data despite the path
    path_avg3 = 'data/eeg/unigram_6class_avg3_250k_v0416/'

    paths = [path_avg3, path_avg1]
    patterns = ['unigram_6class_avg3', 'unigram_6class_avg1']
    patterns_test = ['unigram_6class_avg1', 'unigram_6class_avg1']
    avgs = ['3', '1']

    max_target_length = 176
    N_CLASS = 6
  elif what == 'pretrain_3_1_b':
    model_dir = 'models/ttl=720d/pretrain/' + name + '/'

    path_avg1 = 'data/eeg/pretrain/20210415/avg1/'
    path_avg10 = 'data/eeg/bigram_6class_avg10_250k_v0401/'   # it is unigram data despite the path
    path_avg3 = 'data/eeg/unigram_6class_avg3_250k_v0416/'

    paths = [path_avg3, path_avg1]
    patterns = ['unigram_6class_avg3', 'unigram_6class_avg1']
    patterns_test = ['unigram_6class_avg3', 'unigram_6class_avg1']
    avgs = ['3', '1']

    max_target_length = 176
    N_CLASS = 6

  elif what == 'pretrain_3':
    model_dir = 'models/ttl=720d/pretrain/' + name + '/'

    path_avg1 = 'data/eeg/pretrain/20210415/avg1/'
    path_avg10 = 'data/eeg/bigram_6class_avg10_250k_v0401/'   # it is unigram data despite the path
    path_avg3 = 'data/eeg/unigram_6class_avg3_250k_v0416/'

    paths = [path_avg3]
    patterns = ['unigram_6class_avg3']
    patterns_test = ['unigram_6class_avg1']
    avgs = ['3']

    max_target_length = 176
    N_CLASS = 6

  elif what == 'pretrain_10':
    model_dir = 'models/ttl=720d/pretrain/' + name + '/'

    path_avg1 = 'data/eeg/pretrain/20210415/avg1/'
    path_avg10 = 'data/eeg/bigram_6class_avg10_250k_v0401/'   # it is unigram data despite the path
    path_avg3 = 'data/eeg/unigram_6class_avg3_250k_v0416/'

    paths = [path_avg10]
    patterns = ['unigram_6class_avg10']
    patterns_test = ['unigram_6class_avg1']
    avgs = ['10']

    max_target_length = 176
    N_CLASS = 6
  elif what == 'pretrain_1_baseline':
    model_dir = 'models/ttl=720d/pretrain/' + name + '/'

    path_avg1_bl = 'data/eeg/pretrain/20210415/avg1/'
    path_avg10 = 'data/eeg/bigram_6class_avg10_250k_v0401/'   # it is unigram data despite the path
    path_avg3 = 'data/eeg/unigram_6class_avg3_250k_v0416/'

    paths = [path_avg1_bl]
    patterns = ['unigram_6class_avg1']
    patterns_test = ['unigram_6class_avg1']
    avgs = ['1']

    max_target_length = 176
    N_CLASS = 6
  elif what == 'pretrain_3_baseline':
    model_dir = 'models/ttl=720d/pretrain/' + name + '/'

    path_avg1_bl = 'data/eeg/pretrain/20210415/avg1/'
    path_avg10 = 'data/eeg/bigram_6class_avg10_250k_v0401/'   # it is unigram data despite the path
    path_avg3_bl = 'data/eeg/pretrain/20210415/avg3/'

    paths = [path_avg3_bl]
    patterns = ['unigram_6class_avg3']
    patterns_test = ['unigram_6class_avg1']
    avgs = ['3']

    max_target_length = 176
    N_CLASS = 6
  elif what == 'pretrain_3_1_b_baseline':
    model_dir = 'models/ttl=720d/pretrain/' + name + '/'

    path_avg1_bl = 'data/eeg/pretrain/20210415/avg1/'
    path_avg10 = 'data/eeg/bigram_6class_avg10_250k_v0401/'   # it is unigram data despite the path
    path_avg3_bl = 'data/eeg/pretrain/20210415/avg3/'

    paths = [path_avg3_bl, path_avg1_bl]
    patterns = ['unigram_6class_avg3']
    patterns_test = ['unigram_6class_avg3', 'unigram_6class_avg1']
    avgs = ['3', '1']

    max_target_length = 176
    N_CLASS = 6
  elif what == 'pretrain_3_1_baseline':
    model_dir = 'models/ttl=720d/pretrain/' + name + '/'

    path_avg1_bl = 'data/eeg/pretrain/20210415/avg1/'
    path_avg10 = 'data/eeg/bigram_6class_avg10_250k_v0401/'   # it is unigram data despite the path
    path_avg3_bl = 'data/eeg/pretrain/20210415/avg3/'

    paths = [path_avg3_bl, path_avg1_bl]
    patterns = ['unigram_6class_avg3']
    patterns_test = ['unigram_6class_avg', 'unigram_6class_avg1']
    avgs = ['3', '1']

    max_target_length = 176
    N_CLASS = 6
  elif what == 'pretrain_10_baseline':
    model_dir = 'models/ttl=720d/pretrain/' + name + '/'

    path_avg1_bl = 'data/eeg/pretrain/20210415/avg1/'  # it is unigram data despite the path
    path_avg10_bl = 'data/eeg/bigram_6class_avg10_v0401/'   # it is unigram data despite the path
    path_avg3_bl = 'data/eeg/pretrain/20210415/avg3/'  # it is unigram data despite the path

    paths = [path_avg10_bl]
    patterns = ['unigram_6class_avg10']
    patterns_test = ['unigram_6class_avg1']
    avgs = ['10']

    max_target_length = 176
    N_CLASS = 6
  elif what == 'pretrain_10_3_1_baseline':
    model_dir = 'models/ttl=720d/pretrain/' + name + '/'

    path_avg1_bl = 'data/eeg/pretrain/20210415/avg1/'  # it is unigram data despite the path
    path_avg10_bl = 'data/eeg/bigram_6class_avg10_v0401/'   # it is unigram data despite the path
    path_avg3_bl = 'data/eeg/pretrain/20210415/avg3/'  # it is unigram data despite the path

    paths = [path_avg10_bl, path_avg3_bl, path_avg1_bl]
    patterns = ['unigram_6class_avg10', 'unigram_6class_avg3', 'unigram_6class_avg1']
    patterns_test = ['unigram_6class_avg1', 'unigram_6class_avg1', 'unigram_6class_avg1']
    avgs = ['10','3', '1']

    max_target_length = 176
    N_CLASS = 6
  elif what == 'pretrain_10_3_1_b_baseline':
    model_dir = 'models/ttl=720d/pretrain/' + name + '/'

    path_avg1_bl = 'data/eeg/pretrain/20210415/avg1/'  # it is unigram data despite the path
    path_avg10_bl = 'data/eeg/bigram_6class_avg10_v0401/'   # it is unigram data despite the path
    path_avg3_bl = 'data/eeg/pretrain/20210415/avg3/'  # it is unigram data despite the path

    paths = [path_avg10_bl, path_avg3_bl, path_avg1_bl]
    patterns = ['unigram_6class_avg10', 'unigram_6class_avg3', 'unigram_6class_avg1']
    patterns_test = ['unigram_6class_avg10', 'unigram_6class_avg3', 'unigram_6class_avg1']
    avgs = ['10','3', '1']

    max_target_length = 176
    N_CLASS = 6
  else:
    raise 'dataset not found for key ' + what

  logging.info('model_dir', model_dir)

  tf.io.gfile.makedirs(model_dir)
  summary_path = os.path.join(model_dir,
                              'eeg_' + str(offset) + '_' + str(window_size))
  print('summary_path', summary_path)
  summary_writer = tensorboard.SummaryWriter(summary_path)

  if batch_size % jax.device_count() > 0:
    raise ValueError('Batch size must be divisible by the number of devices')
  device_batch_size = batch_size // jax.device_count()




  print('create model')
  bs = device_batch_size * jax.device_count()

  rng = random.PRNGKey(random_seed)
  rng, init_rng = random.split(rng)


  real_len = window_size - offset

  input_shape = (bs, real_len, 64)
  print(f'Input shape is: {input_shape}')


  transformer_kwargs = {
      'vocab_size': N_CLASS,
      'output_vocab_size': N_CLASS,
      'emb_dim': N_EMBEDDING,  # 512
      'num_heads': 8,  # 16
      'num_layers': 4,  # 6
      'qkv_dim': 512,  # 1024
      'mlp_dim': 1024,  # 2024
      'max_len': max_target_length,
      'ablation': ablation
  }
  if 'l1' in ablation:
    transformer_kwargs['num_layers'] = 1

  print('transformer_kwargs', transformer_kwargs)
  model = create_model(init_rng, tuple(input_shape), transformer_kwargs)

  print('end create model')
  optimizer = create_optimizer(model, learning_rate)
  del model  # don't keep a copy of the initial model

  learning_rate_fn = create_learning_rate_scheduler(
      base_learning_rate=learning_rate)

  p_train_step = jax.pmap(
      functools.partial(train_step, learning_rate_fn=learning_rate_fn),
      axis_name='batch')
  p_eval_step = jax.pmap(eval_step, axis_name='batch')

  # We init the first set of dropout PRNG keys, but update it afterwards inside
  # the main pmap'd training update for performance.
  dropout_rngs = random.split(rng, jax.local_device_count())


  best_optimizer = None

  num_pretrain = 0
  new_best_test_acc = 0
  for data_path, pattern, pattern_test, avg in zip(paths, patterns, patterns_test, avgs):
    DATA_PATH = data_path

    step_offset = num_pretrain * num_train_steps
    num_pretrain += 1

    #    max_target_length = 176
    #    N_CLASS = 6

    filename_pattern_test = f'{pattern_test}_test_f*.tfrecords'
    filename_pattern_test_wr = f'{pattern_test}_test_f*.tfrecords'
    filename_pattern_train = f'{pattern}_train*.tfrecords'
    filename_pattern_dev = f'{pattern_test}_dev_*.tfrecords'

    print('start preparing training data')
    logging.info('data path: %s', DATA_PATH)
    logging.info('training path: %s', filename_pattern_train)
    logging.info('dev path: %s', filename_pattern_dev)
    logging.info('test path: %s', filename_pattern_test)
    logging.info('test_wr path: %s', filename_pattern_test_wr)

    train_ds = input_pipeline.get_dataset(
        filename_pattern=filename_pattern_train,
        data_path=data_path,
        batch_size=eval_batch_size,
        bucket_size=max_target_length,
        offset=offset,
        window_size=window_size)
    print('end preparing training data')
    print('start preparing dev data')
    train_ds_iter = iter(train_ds)

    eval_ds = input_pipeline.get_dataset(
        filename_pattern=filename_pattern_dev,
        data_path=data_path,
        batch_size=eval_batch_size,
        bucket_size=max_target_length,
        offset=offset,
        window_size=window_size,
        repeat=1)
    # read ones the entire data set in
    eval_iter = iter(eval_ds)
    cnt = 0
    eval_shapes = []
    labs = {}
    train_iter = iter(train_ds)


    print('compute eval buckets')
    test_ds = input_pipeline.get_dataset(
        filename_pattern=filename_pattern_test,
        data_path=data_path,
        batch_size=eval_batch_size,
        bucket_size=max_target_length,
        offset=offset,
        window_size=window_size,
        repeat=1)
    test_iter = iter(test_ds)

    if filename_pattern_test_wr:
      test_wr_ds = input_pipeline.get_dataset(
          filename_pattern=filename_pattern_test_wr,
          data_path=DATA_PATH,
          batch_size=eval_batch_size,
          bucket_size=max_target_length,
          offset=offset,
          window_size=window_size,
          repeat=1)
      test_wr_iter = iter(test_wr_ds)


    if best_optimizer is not None:
      optimizer = best_optimizer


    metrics_all = []
    tick = time.time()
    best_acc = 0.0
    new_best_test_wr_acc = 0.0
    print('start training')
    for step, batch in zip(range(num_train_steps), train_iter):

      cur_pred_batch_size = batch[0].shape[0]

      if cur_pred_batch_size != batch_size:
        logging.info('Uneven train batch size %d.', cur_pred_batch_size)

      batch = shard(jax.tree_map(lambda x: x._numpy(), batch))  # pylint: disable=protected-access

      optimizer, metrics, dropout_rngs = p_train_step(
          optimizer, batch, dropout_rng=dropout_rngs)
      metrics_all.append(metrics)

      if (step + 1) % eval_freq == 0:

        metrics_all = get_metrics(metrics_all)
        lr = metrics_all.pop('learning_rate').mean()
        metrics_sums = jax.tree_map(jnp.sum, metrics_all)
        denominator = metrics_sums.pop('denominator')

        normalizing_factor = metrics_sums.pop('normalizing_factor')
        accuracy = metrics_sums['accuracy'] / normalizing_factor
        train_accuracy = accuracy
        print(f'step:{step_offset+step} train acc: {train_accuracy:.2f}')


        summary = jax.tree_map(lambda x: x / denominator, metrics_sums)  # pylint: disable=cell-var-from-loop
        summary['learning_rate'] = lr
        # Calculate (clipped) perplexity after averaging log-perplexities:
        summary['perplexity'] = jnp.clip(jnp.exp(summary['loss']), a_max=1.0e4)
        summary['accuracy'] = accuracy
        summary['avgs'] = avg

        logging.info('train in step: %d, loss: %.4f', step, summary['loss'])

        tock = time.time()
        steps_per_sec = eval_freq / (tock - tick)
        tick = tock
        summary_writer.scalar('steps per second', steps_per_sec, step_offset+step)
        for key, val in summary.items():
          summary_writer.scalar('train/' + key, val, step_offset+step)


        # reset metric accumulation for next evaluation cycle.
        metrics_all = []
        eval_output = model_dir + 'dev_classification.txt'
        new_best_acc = evaluate(
            eval_ds,
            num_eval_steps,
            best_acc,
            eval_batch_size,
            p_eval_step,
            optimizer,
            step_offset+step,
            train_accuracy,
            summary_writer,
            'eval',
            eval_output=eval_output)
        if best_acc < new_best_acc:

          best_optimizer = optimizer

          best_acc = new_best_acc
          eval_output = model_dir + 'test_classification.txt'
          new_best_test_acc = evaluate(
              test_ds,
              num_eval_steps,
              0,
              eval_batch_size,
              p_eval_step,
              optimizer,
              step_offset+step,
              train_accuracy,
              summary_writer,
              'test',
              eval_output=eval_output,
              eval_output_force=True)
          if filename_pattern_test_wr:
            eval_output = model_dir + 'test_wr.txt'
            new_best_test_wr_acc = evaluate(
                test_wr_ds,
                num_eval_steps,
                0,
                eval_batch_size,
                p_eval_step,
                optimizer,
                step_offset+step,
                train_accuracy,
                summary_writer,
                'test',
                eval_output=eval_output,
                eval_output_force=True)
          summary_writer.scalar('eval/best_eval', best_acc, step_offset+step)
          summary_writer.scalar('test/best_eval', new_best_test_acc, step_offset+step)


    # write early stopping results:
    final_accuracy_path = model_dir + 'accuracy.txt'
    with tf.io.gfile.GFile(final_accuracy_path, mode='w') as f:
      f.write('best_dev:')
      f.write(str(best_acc))
      f.write('\n')
      f.write('best_test_acc:')
      f.write(str(new_best_test_acc))
      f.write('\n')
      f.write('best_test_wr_acc:')
      f.write(str(new_best_test_wr_acc))
      f.write('\n')

  return (best_acc, new_best_test_acc, new_best_test_wr_acc)


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')


  cl_type = FLAGS.classifier_type
  window_end = FLAGS.window_size

  # what='matched_6class_avg10'
  what = FLAGS.what
  step_size = FLAGS.step_size
  offset = FLAGS.offset
  random_seed = FLAGS.random_seed
  name = 'AlexTest'
  ablation = ''


  # xmanager might provide hyperparameters as 'xm_parameters'
  #d = json.loads(FLAGS.xm_parameters)

  #offset = int(d['offset'])
  #window_end = int(d ['window_size'])

  #logging.info(f'offset: %d window_size: %d', offset, window_end)

  #random_seed = d['seed']
  #what = d['what']

  #ablation = ''
  #if 'ablation' in d:
   # ablation = d['ablation']

  #i = random_seed
  #name='v21bpad_'+what+'_'+ablation+'_server_try2_step_o'+str(offset)+'_e'+str(window_end)+'_i'+str(i)
  #print(f'offset:{offset}, window_size:{window_end}')

  #random_seed = random_seed * 7
  if what.startswith('pretrain'):
    pretrain(
        offset=offset,
        window_size=window_end,
        name=name,
        what=what,
        random_seed=random_seed,
        num_train_steps=400000,  #   400000 for testing set to 10k
        ablation=ablation)  # change this back 400000
  else:
    train(
        offset=offset,
        window_size=window_end,
        name=name,
        what=what,
        random_seed=random_seed,
        num_train_steps=400000,
        ablation=ablation)  # change this back 400000





if __name__ == '__main__':
  app.run(main)
