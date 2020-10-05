

import time
import tensorflow as tf
import os
import sys
import numpy as np

@tf.function
def measure_single(x, y):

  print("Using Single Step")
  z = tf.matmul(x, y)
  return z

## @tf.function
def measure_loop(x, y, steps):

  print("Using loop steps = ", steps)
  z = tf.matmul(x, y)
  for i in range(steps-1):
    z = tf.matmul(x, y)
  return z

@tf.function
def measure_square(x, steps):

  print("Using square steps: = ", steps)
  for i in range(steps):
    x = tf.matmul(x, x)
  return x

@tf.function
def measure_double(x, y, steps):

  xT =  tf.transpose(x)
  z = tf.matmul(x,y)
  for i in range(steps):
    y = tf.matmul(xT, z)
    z = tf.matmul(x, y)
  return z


def measure_update(x, y, steps, h):

  z = tf.matmul(x,y)
  indices = tf.expand_dims(tf.range(h), axis=1)
  for i in range(steps):
    z += tf.matmul(x, y)
    y = tf.tensor_scatter_nd_update(y,indices,z)
  return z


def measure_square_eager(x, steps):
  for i in range(steps):
    x = tf.matmul(x, x)
  return x

if __name__ == "__main__":
  import sys
  import argparse

  parser = argparse.ArgumentParser(
     description="Measure the performance of GEMM using matmul for tensorflow"
  )
  # model related parameters
  parser.add_argument("-m", "--msize", type=int, default=4)
  parser.add_argument("-n", "--nsize", type=int, default=4)
  parser.add_argument("-k", "--ksize", type=int, default=4)
  parser.add_argument("--dtype", type=str, default="float32")
  parser.add_argument("--steps", type=int, default=10)
  parser.add_argument("--warmups", type=int, default=10)
  args = parser.parse_args()

  m = args.msize
  n = args.nsize
  k = args.ksize
  steps = args.steps
  dt = tf.float32
  if (args.dtype == "float16" or args.dtype == "half"):
    dt = tf.float16
  elif (args.dtype == "bfloat16"):
    dt = tf.bfloat16

  print("Test problem size for m, n, k are : ", m, n, k)
  print("Test problem data type : ", dt)

  # tf.config.run_functions_eagerly(True)
  tf.random.set_seed(123)

  resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu="grpc://"+os.environ["TPU_IP"])
  tf.config.experimental_connect_to_cluster(resolver)
  tf.tpu.experimental.initialize_tpu_system(resolver)
  print("   ")
  tpus = tf.config.list_logical_devices('TPU')
  print("There are {} tpu logical devices".format(len(tpus)))
  print(tpus[0])

  with tf.device("/job:worker/device:TPU:0"):

    x = tf.random.uniform([m,k], dtype=dt)
    y = tf.random.uniform([k,n], dtype=dt)
    z = tf.zeros([m,n], dtype=dt)
    # x = tf.eye(m,k)
    # y = tf.eye(k,n)

    if False and m == k and m == n:
        for i in range(2):
            stime = time.perf_counter()
            z = measure_square(x, tf.constant(steps))
            print(z)
            etime = time.perf_counter()
            t1 = etime - stime
            print("Measure_squre time: ", t1)
            print("sum of x is : ", tf.reduce_sum(tf.linalg.diag_part(x)).numpy(), " should be : ", m)
            print("TPU1: {0:.6f} secs for {1} steps with rate {2} gflops".format(t1, steps, m*n*k*1.0*2/1024/1024/1024/(t1/steps)))

#    print(" ")
#    for i in range(2):
#        stime = time.perf_counter()
#        for j in range(steps):
#            z = measure_single(x, y)
#        print(z)
#        etime = time.perf_counter()
#        t2 = etime - stime
#        print("Measure_SINGLE time: ", t2)
#        print("TPU2: {0:.6f} secs for {1} steps with rate {2} gflops".format(t2, steps, m*n*k*1.0*2/1024/1024/1024/(t2/steps)))
#
#    print(" ")
#    for i in range(2):
#        stime = time.perf_counter()
#        z = measure_double(x, y, tf.constant(steps))
#        print(z)
#        etime = time.perf_counter()
#        t2 = etime - stime
#        t2 /= 2
#        print("Measure_DOUBLE time: ", t2)
#        print("TPU2: {0:.6f} secs for {1} steps with rate {2} gflops".format(t2, steps, m*n*k*1.0*2/1024/1024/1024/(t2/steps)))
#
#    print(" ")
#    print("Signamture: ", measure.pretty_printed_concrete_signatures())
#    print(" ")
#
    cf = measure_single.get_concrete_function(x, y)
    print(" ")
    for i in range(2):
        stime = time.perf_counter()
        for j in range(steps):
            z = cf(x, y)
        print(z)
        etime = time.perf_counter()
        t2 = etime - stime
        print("Measure_CF time: ", t2) 
        print("TPU2: {0:.6f} secs for {1} steps with rate {2} gflops for {3}, {4}, {5}".format(t2, 
               steps, m*n*k*1.0*2/1024/1024/1024/(t2/steps), m, n, k))

    cf = measure_double.get_concrete_function(x, y, steps)
    print(" ")
    for i in range(2):
        stime = time.perf_counter()
        z = cf(x, y, steps)
        print(z)
        etime = time.perf_counter()
        t2 = etime - stime
        t2 /= 2
        print("Measure_CF time: ", t2) 
        print("TPU2: {0:.6f} secs for {1} steps with rate {2} gflops for {3}, {4}, {5} ".format(t2, 
               steps, m*n*k*1.0*2/1024/1024/1024/(t2/steps), m, n, k))

