

import time
import tensorflow as tf
import os
import sys
import numpy as np

# @tf.function
def measure0(z, x, y, steps):

  xT =  tf.transpose(x)
  z = tf.matmul(x,y)
  mystart = time.perf_counter()
  for i in range(steps):
    y = tf.matmul(xT, z)
    z = tf.matmul(x, y)
  e = time.perf_counter()
  z.numpy()
  # print(z)
  myend = time.perf_counter()
  print("TIME: ", e - mystart, myend - mystart)
  return myend - mystart


def measure(z, x, y, steps, h):

  z = tf.matmul(x,y)
  indices = tf.expand_dims(tf.range(h), axis=1)
  mystart = time.perf_counter()
  for i in range(steps):
    z += tf.matmul(x, y)
    # x[:, 0:h] = tf.math.minimum(x[:, 0:h], z[:, 0:h])
    ## slowdown half
    ## x = tf.Variable(x)
    ## x[:, 0:h].assign(z[:, 0:h])
    ## x = tf.convert_to_tensor(x)
    tf.tensor_scatter_nd_update(y,indices,z)
  e = time.perf_counter()
  z.numpy()
  myend = time.perf_counter()
  print("TIME: ", e - mystart, myend - mystart)
  return myend - mystart


def measure_square_eager(x, steps):
  x = tf.matmul(x, x)
  mystart = time.perf_counter()
  for i in range(steps):
    x = tf.matmul(x, x)
  e = time.perf_counter()
  x.numpy()
  myend = time.perf_counter()
  return myend - mystart

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

  # save here`
  # tf.config.run_functions_eagerly(True)

  tf.random.set_seed(123)
  # x = tf.random.uniform([m,k], dtype=dt)
  # y = tf.random.uniform([k,n], dtype=dt)
  z = tf.zeros([m,n], dtype=dt)
  x = tf.eye(m,k)
  y = tf.eye(k,n)


  resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu="grpc://"+os.environ["TPU_IP"])
  tf.config.experimental_connect_to_cluster(resolver)
  tf.tpu.experimental.initialize_tpu_system(resolver)
  print("   ")
  tpus = tf.config.list_logical_devices('TPU')
  print("There are {} tpu logical devices".format(len(tpus)))
  print(tpus[0])

  with tf.device("/job:worker/device:TPU:0"):
  # with tf.device(tpus[0]):
  # with tf.device('TPU:0'):

    if m == k and m == n:
        t1 = measure_square_eager(x, tf.constant(steps))
        print("t1 is ", t1)
        print("sum of x is : ", tf.reduce_sum(tf.linalg.diag_part(x)).numpy(), " should be : ", min(m,k))
        print("TPU1: {0:.6f} secs for {1} steps with rate {2} gflops".format(t1, steps, m*n*k*1.0*2/1024/1024/1024/(t1/steps)))

        # print(" ")
        # t1 = measure_square_eager(x, tf.constant(steps))
        # print("t1 is ", t1)
        # print("TPU1: {0:.6f} secs for {1} steps with rate {2} gflops".format(t1, steps, m*n*k*1.0*2/1024/1024/1024/(t1/steps)))

    print(" ")
    t2 = measure(z, x, y, tf.constant(steps), min(k,m))
    print("Time2 ", t2)
    print("TPU2: {0:.6f} secs for {1} steps with rate {2} gflops".format(t2, steps, m*n*k*1.0*4/1024/1024/1024/(t2/steps)))

    print(" ")
    t2 = measure(z, x, y, tf.constant(steps), min(k,m))
    print("Time2 ", t2)
    print("TPU2: {0:.6f} secs for {1} steps with rate {2} gflops".format(t2, steps, m*n*k*1.0*4/1024/1024/1024/(t2/steps)))

    print(" ")
    print("Signamture: ", measure.pretty_printed_concrete_signatures())
    print(" ")

    cf = measure.get_concrete_function(z, x, y, tf.constant(steps), min(k,m))
    t2 = cf(z, x, y, tf.constant(steps))
    print("Time2 ", t2)
    print("TPU2: {0:.6f} secs for {1} steps with rate {2} gflops".format(t2, steps, m*n*k*1.0*2/1024/1024/1024/(t2/steps)))

