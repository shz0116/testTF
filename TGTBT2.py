# TENSORFLOW VERSION: '2.3.0-dev20200620'
# Test command:
#   export TPU_IP=
#   python3  g2.py --features=1000000 --nnz=30 --em=128 --steps=4 --warmups=1 --batch=65536
# Results:
#   Test batch =  65536  nnz =  30 , em =  128
#   Lookup Shape:  (1966080, 128)  RES shape:  (65536, 128)
#   TPU: total test time: 0.002392 0.004470 seconds, for      4 steps 
#   TPU: total bytes 1006632960, mem bw 1683.631 GB/s
#   TPU: total bytes 1006632960, mem bw 900.720 GB/s

import time
import tensorflow as tf
import itertools
import numpy as np
import os
import sys

from tensorflow.python.ops import init_ops_v2
from tensorflow.python.tpu import tpu_embedding_v2
from tensorflow.python.tpu import tpu_embedding_v2_utils
from tensorflow.python.distribute import tpu_strategy
from tensorflow.python.tpu import tpu_strategy_util
from tensorflow.python.distribute.cluster_resolver import tpu_cluster_resolver
from tensorflow.python.eager import def_function
from tensorflow.python.eager import remote
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.util import nest

import argparse

parser = argparse.ArgumentParser(
    description="Measure the performance of pytorch embeddingbag" )
parser.add_argument("--features", type=int, default=20)
parser.add_argument("--em", type=int, default=4)
parser.add_argument("--nnz", type=int, default=2)
parser.add_argument("--batch", type=int, default=4)
parser.add_argument("--steps", type=int, default=1)
parser.add_argument("--warmups", type=int, default=0)
parser.add_argument("--randomseed", type=int, default=0)
parser.add_argument("--testtpu", type=int, default=0)
parser.add_argument("--verify", type=int, default=0)
parser.add_argument("--times", type=int, default=1)

args = parser.parse_args()

# embedding_values = np.array(list(range(32)), dtype=np.float64)
# initializer = init_ops_v2.Constant(embedding_values)

table_test = tpu_embedding_v2_utils.TableConfig(
        vocabulary_size=args.features,
        dim=args.em,
        initializer=None,
        combiner='sum',
        name='test')
feature_config = (
        tpu_embedding_v2_utils.FeatureConfig(
            table=table_test, name='watched'))

batch = args.batch
nnz = args.nnz
features = args.features

feature_watched_values = np.random.randint(0, features, (batch * nnz * 4, ))
# print("Feature: ", feature_watched_values)
batch_size = batch * nnz 

# feature_watched_values = [0, 0, 1, 0, 1, 1]
# feature_watched_row_lengths = [1, 2, 2, 1]

resolver = None

def get_strategy():
   resolver = tpu_cluster_resolver.TPUClusterResolver(tpu="grpc://"+os.environ["TPU_IP"])
   remote.connect_to_cluster(resolver)
   topology = tpu_strategy_util.initialize_tpu_system(resolver)
   print("Device coordinates: ", topology.device_coordinates)
   device_assignment = tf.python.tpu.device_assignment.DeviceAssignment.build(topology,computation_shape=[1, 1, 1, 1],num_replicas=1)
   # device_assignment = tf.python.tpu.device_assignment.DeviceAssignment.build(topology)

   return tpu_strategy.TPUStrategy(resolver, device_assignment=device_assignment)

def create_mid_level(optimizer=None):
    # Create `TPUEmbedding` object.
    if optimizer is None:
      optimizer = tpu_embedding_v2_utils.SGD(learning_rate=0.1)
    return tpu_embedding_v2.TPUEmbedding(
        feature_config=feature_config,
        batch_size=batch_size,
        optimizer=optimizer)

def create_strategy_and_mid_level(optimizer_name):
   strategy = get_strategy()
   with strategy.scope():
       if optimizer_name == 'sgd':
           optimizer = tpu_embedding_v2_utils.SGD(learning_rate=0.1)
       elif optimizer_name == 'adagrad':
           optimizer = tpu_embedding_v2_utils.Adagrad(learning_rate=0.1)
       elif optimizer_name == 'adam':
           optimizer = tpu_embedding_v2_utils.Adam(learning_rate=0.1)
       else:
           raise ValueError('optimizer is not recognized: ', optimizer_name)
       embedding = create_mid_level(optimizer=optimizer)
   return strategy, embedding, optimizer

def create_dense_input_fn(strategy, include_weights=False, weight=0.5):
    def input_fn(ctx):
      del ctx
#      features = (
#          constant_op.constant(feature_watched_values,
#                               dtype=dtypes.int32))
      features = (feature_watched_values)
      if include_weights:
        weights = [array_ops.ones_like(t, dtype=dtypes.float32) * weight
                   for t in features]
        features = (features, tuple(weights))
      return dataset_ops.DatasetV2.from_tensor_slices(features).repeat().batch(batch_size)
    return input_fn

def get_replica_numpy(structured, strategy, replica_id):

    def select_replica(x):
      x = strategy.experimental_local_results(x)
      if len(x) == 1:
        return x # x.numpy()
 
      return x[replica_id] # x[replica_id].numpy()

    return nest.map_structure(select_replica, structured)

@tf.function
def test_fn(TIMES):
    def step():
        print("In STEPs")
        activation = embedding.dequeue()
        # print_op = tf.print(activation, output_stream=sys.stderr)
        # with tf.control_dependencies([print_op]):
        # tensor = tf.range(10)
        # tf.print(tensor, output_stream=sys.stderr)
        return activation

    embedding.enqueue(next(dist_iter), training=False)
    activation = strategy.run(step)
    for j in tf.range(TIMES-1):
        embedding.enqueue(next(dist_iter), training=False)
        activation = strategy.run(step)
    return  activation

if __name__ == "__main__":

    strategy, embedding, optimizer = create_strategy_and_mid_level('sgd')
    training = False

    input_fn = create_dense_input_fn(strategy)
    dist = strategy.experimental_distribute_datasets_from_function(
        input_fn,
        options=distribute_lib.InputOptions(
            experimental_prefetch_to_device=False))
    dist_iter = iter(dist)

    t1 = 0.0
    t2 = 0.0
    t3 = 0.0
    steps = args.steps
    warmups = args.warmups
 
    # warmup 
    start = time.time()
    res = get_replica_numpy(test_fn(args.times), strategy, 0)
    res[0].numpy()
    end = time.time()
    t1 = end - start
    print("TIME warmups {0:.6f} : ".format(t1))

    # measure
    start = time.time()
    with tf.experimental.async_scope():
        for _ in tf.range(steps):
            res = test_fn(args.times)
        res0 = get_replica_numpy(res, strategy, 0)

    end = time.time()
    t2 = end - start
    res0[0].numpy()
    end3 = time.time()
    t3 = end3 - end
    print("TIME measure : ", t2, t3)

    total_bytes = args.batch * args.nnz * args.em * tf.float32.size
    print("Test batch = ", args.batch, " nnz = ", args.nnz, ", em = ", args.em, " times= ", args.times)
    print("Lookup Shape: ", res0[0].shape, " RES shape: ", res0[0].shape)
    print("TPU: total test time: {0:.6f} {1:.6f} {2:.6f} seconds, for {3:6d} steps ".format(t1, t2, t3, steps))
    print("TPU: total bytes {0}, mem bw {1:.3f} GB/s".format(total_bytes, total_bytes*1.0*steps*args.times/(t2+t3)/1.0e9))
    
    print("done")

