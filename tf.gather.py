
# command:
# python3 -m pdb embtest.py --features=1000 --nnz=30 --batch=128
#
# error:
# *** tensorflow.python.framework.errors_impl.ResourceExhaustedError: 
#     Ran out of memory in memory space vmem. It should not be possible to run out of vmem - please file a bug against XLA.
#
import tensorflow as tf
import numpy as np
import sys
import os
import time

def measure(params, sp_ids, steps, thr):
  res = tf.gather(params, sp_ids, axis=0, batch_dims=0, name="TEST1")
  return res

if __name__ == "__main__":

  import sys 
  import argparse

  parser = argparse.ArgumentParser(
     description="Measure the performance of tensorflow embeddingbag using tf.nn.embedding" )
  parser.add_argument("--features", type=int, default=10)
  parser.add_argument("--em", type=int, default=2)
  parser.add_argument("--nnz", type=int, default=2)
  parser.add_argument("--batch", type=int, default=4)
  parser.add_argument("--steps", type=int, default=1)
  parser.add_argument("--warmups", type=int, default=0)

  args, unknown = parser.parse_known_args()

  features     = args.features
  em           = args.em
  nnz          = args.nnz
  batch        = args.batch
  steps        = args.steps
  warmups      = args.warmups

  sp_ids = np.random.randint(0, features, (batch * nnz,))
  res = tf.zeros([batch, em])

  resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu="grpc://"+os.environ["TPU_IP"])
  tf.config.experimental_connect_to_cluster(resolver)
  tf.tpu.experimental.initialize_tpu_system(resolver)
  print("   ")
  tpus = tf.config.list_logical_devices('TPU')
  print("There are {} tpu logical devices".format(len(tpus)))
  print(tpus[0])

  with tf.device('TPU:0'):
    params = tf.random.uniform([features, em])
    for i in range(steps):
        t1 = time.time()
        res = measure(params, sp_ids, tf.constant(steps), features//2)
        res.numpy()
        t2 = time.time()
        print("Time is : ", t2 - t1)

  steps=1
  # print(params)
  # print(sp_ids)
  print(res)
  total_bytes = args.batch * args.nnz * args.em * tf.float32.size
  print("TPU: total test time: {0:.6f} seconds, for {1:6d} steps ".format(t2 - t1, steps))
  print("TPU: total bytes {0}, mem bw {1:.3f} GB/s".format(total_bytes, total_bytes*1.0*steps/(t2 - t1)/1.0e9))
  


