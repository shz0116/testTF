import time
import tensorflow as tf
import itertools
import numpy as np
import os

def RandomDataset(args):

    features = {}
    
    # PROBLEM: what's the correct dataset format ?
 
    features["feature_one"] = tf.random.uniform(shape=(args.batch,args.em),maxval=args.features, dtype=tf.dtypes.int32)
    features["feature_two"] = tf.random.uniform(shape=(args.batch,args.em),maxval=args.features, dtype=tf.dtypes.int32)
    # features["feature_one"] = tf.random.uniform(shape=(args.batch,),maxval=args.features, dtype=tf.dtypes.int32)
    # features["feature_two"] = tf.random.uniform(shape=(args.batch,),maxval=args.features, dtype=tf.dtypes.int32)
    ds = tf.data.Dataset.from_tensor_slices(features)
    ds = ds.batch(args.batch, drop_remainder=True)
    ds = ds.take(1).cache().repeat(4)
    return ds

if __name__ == "__main__":

  import sys
  import argparse

  parser = argparse.ArgumentParser(
     description="Measure the performance of pytorch embeddingbag" )
  parser.add_argument("--features", type=int, default=20)
  parser.add_argument("--em", type=int, default=2)
  parser.add_argument("--nnz", type=int, default=10)
  parser.add_argument("--batch", type=int, default=4)
  parser.add_argument("--steps", type=int, default=1)
  parser.add_argument("--warmups", type=int, default=0)
  parser.add_argument("--randomseed", type=int, default=0)
  parser.add_argument("--testtpu", type=int, default=0)
  parser.add_argument("--verify", type=int, default=0)

  args = parser.parse_args()

  num_features = args.features
  embed_dim    = args.em
  nnz          = args.nnz
  batch_size   = args.batch
  steps        = args.steps
  warmups      = args.warmups

  random_seed  = args.randomseed

  print("Test problem size:")
  print("Number of features : ", num_features)
  print("Embedding size     : ", embed_dim)
  print("Nnz_per_input      : ", nnz)
  print("Number of batches  : ", batch_size)
  print("Eager execution : ", tf.executing_eagerly())


  tf.random.set_seed(0)
  # indices = np.random.randint(0, num_features, (batch_size, nnz))
  # print(indices)

  #def data_fn(indices):
  #  dataset = tf.data.Dataset.from_tensor_slices(indices)
  #  dataset = dataset.repeat().cache().batch(1) 
  #  print(list(dataset.as_numpy_iterator()))
  #  return dataset
  # dataset = tf.data.Dataset.from_tensor_slices(indices)
  # dataset = dataset.cache().repeat(2).batch(1) 
  # input_fn = GetRandomDataset(False, args)
  # dataset = input_fn(args)
  dataset = RandomDataset(args)
  print(list(dataset.as_numpy_iterator())) 

  # tpu = "grpc://10.45.145.114"
  resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu="grpc://"+os.environ["TPU_IP"])
  tf.config.experimental_connect_to_cluster(resolver)
  topology = tf.tpu.experimental.initialize_tpu_system(resolver)
  print("   ")
  tpus = tf.config.list_logical_devices('TPU')
  print("There are {} tpu logical devices".format(len(tpus)))
  print(tpus)
  print("Mesh Shape: ", topology.mesh_shape)
  print("Mesh rank : ", topology.mesh_rank)
  print("Device coordinates: ", topology.device_coordinates)

  # if tf.config.experimental.list_physical_devices("TPU"):
  # with tf.device("/job:worker/device:TPU:0"):
  #  x = tf.random.normal(shape)

  device_assignment = tf.tpu.experimental.DeviceAssignment.build(topology,computation_shape=[1, 1, 1, 1],num_replicas=1)
  strategy = tf.distribute.TPUStrategy(resolver, experimental_device_assignment=device_assignment)
     
  table_config_one = tf.tpu.experimental.embedding.TableConfig(args.features, args.em, None, None, 'mean')
  print(table_config_one.dim)
  feature_config = {'feature_one': tf.tpu.experimental.embedding.FeatureConfig(table=table_config_one),
                    'feature_two': tf.tpu.experimental.embedding.FeatureConfig(table=table_config_one)}
  print(feature_config.keys())
  print(feature_config.values())

  with strategy.scope():
    embedding = tf.tpu.experimental.embedding.TPUEmbedding(
            feature_config, batch_size, tf.tpu.experimental.embedding.SGD(0.1), False, True)
    # print(embedding.embedding_tables)
  
  # change to strategy.experimental_distribute_datasets_from_function later
  distributed_dataset = (strategy.experimental_distribute_dataset(dataset,options=tf.distribute.InputOptions(experimental_prefetch_to_device=False)))
  # distributed_dataset = (strategy.experimental_distribute_datasets_from_function(input_fn,options=tf.distribute.InputOptions(experimental_prefetch_to_device=False)))
  # dataset_iterator = iter(distributed_dataset)
  print(distributed_dataset.element_spec)

  # local_result = strategy.experimental_local_results(distributed_values)
  # print(local_results)

  @tf.function
  def evalution_step(distributed_dataset, batch_size):
    def tpu_step(tpu_features):
      activations = embedding.dequeue()

      # PROBLEM: how to see the values for the activations ?
      # print only see something like: Tensor("strided_slice:0", shape=(4, 2), dtype=float32)
      # no numpy() can be used

      print("AAA activation: ", activations)
      print(activations["feature_one"])
      # tf.print(activations["feature_one"])
      print(activations["feature_one"].__dict__)
      print(activations["feature_one"])
      # model_output = model(activations)
      # Insert your evaluation code here.
      return activations

    for i in distributed_dataset:
      # embedding_features = {'feature_one' :  dataset_iterator.get_next()}
      # embedding_features = dataset_iterator.get_next()
      embedding_features = i
      embedding.enqueue(embedding_features, training=False)
      print("BBB embedding_features:: ", embedding_features)
    tpu_features = "GOOOD"
    strategy.run(tpu_step, args=(tpu_features, ))

  for i in distributed_dataset:
    print("DIST : ", i)

  res = evalution_step(distributed_dataset, batch_size)
  print("Final res: ", res)


