#!/usr/bin/env python

import numpy as np


def base_features(X):
  return X

def average_memory(X):
  cycle_time = X[:,0][:,None]
  min_memory = X[:,1][:,None]
  max_memory = X[:,2][:,None]
  cache_size = X[:,3][:,None]
  min_channels = X[:,4][:,None]
  max_channels = X[:,5][:,None]

  avg_memory = ((min_memory + max_memory)/2)
  return np.hstack((cycle_time, avg_memory, cache_size, min_channels, max_channels))

def average_channels(X):
  cycle_time = X[:,0][:,None]
  min_memory = X[:,1][:,None]
  max_memory = X[:,2][:,None]
  cache_size = X[:,3][:,None]
  min_channels = X[:,4][:,None]
  max_channels = X[:,5][:,None]

  avg_channels = ((min_channels + max_channels)/2)
  return np.hstack((cycle_time, min_memory, max_memory, cache_size, avg_channels))

def cycle_time_to_clock_speed(X):
  cycle_time = X[:,0][:,None]
  min_memory = X[:,1][:,None]
  max_memory = X[:,2][:,None]
  cache_size = X[:,3][:,None]
  min_channels = X[:,4][:,None]
  max_channels = X[:,5][:,None]

  clock_speed = (1/X[:,0])[:,None]
  return np.hstack((clock_speed, min_memory, max_memory, cache_size, min_channels, max_channels))


# def cache_average_memory_ratio(X):
#   cycle_time = X[:,0][:,None]
#   min_memory = X[:,1][:,None]
#   max_memory = X[:,2][:,None]
#   cache_size = X[:,3][:,None]
#   min_channels = X[:,4][:,None]
#   max_channels = X[:,5][:,None]

#   avg_memory = ((min_memory + max_memory)/2)
#   ratio = cache_size/avg_memory
#   return np.hstack((cycle_time, ratio, min_channels, max_channels))

def average_memory_and_average_channels(X):
  cycle_time = X[:,0][:,None]
  min_memory = X[:,1][:,None]
  max_memory = X[:,2][:,None]
  cache_size = X[:,3][:,None]
  min_channels = X[:,4][:,None]
  max_channels = X[:,5][:,None]

  avg_channels = ((min_channels + max_channels)/2)
  avg_memory = ((min_memory + max_memory)/2)

  return np.hstack((cycle_time, avg_memory, cache_size, avg_channels))

def average_memory_and_cycle_time_to_clock_speed(X):
  cycle_time = X[:,0][:,None]
  min_memory = X[:,1][:,None]
  max_memory = X[:,2][:,None]
  cache_size = X[:,3][:,None]
  min_channels = X[:,4][:,None]
  max_channels = X[:,5][:,None]

  clock_speed = (1000000000/cycle_time)
  avg_memory = ((min_memory + max_memory)/2)

  return np.hstack((clock_speed, avg_memory, cache_size, min_channels, max_channels))


def average_memory_and_cycle_time_to_clock_speed_and_average_channels(X):
  cycle_time = X[:,0][:,None]
  min_memory = X[:,1][:,None]
  max_memory = X[:,2][:,None]
  cache_size = X[:,3][:,None]
  min_channels = X[:,4][:,None]
  max_channels = X[:,5][:,None]

  clock_speed = (1000000000/cycle_time)
  avg_memory = ((min_memory + max_memory)/2)
  avg_channels = ((min_channels + max_channels)/2)

  return np.hstack((clock_speed, avg_memory, cache_size, avg_channels))

def average_channels_and_cycle_time_to_clock_speed(X):
  cycle_time = X[:,0][:,None]
  min_memory = X[:,1][:,None]
  max_memory = X[:,2][:,None]
  cache_size = X[:,3][:,None]
  min_channels = X[:,4][:,None]
  max_channels = X[:,5][:,None]

  avg_channels = ((min_channels + max_channels)/2)
  clock_speed = (1000000000/cycle_time)

  return np.hstack((clock_speed, min_memory, max_memory, cache_size, avg_channels))

# def average_channels_and_cache_average_memory_ratio(X):
#   cycle_time = X[:,0][:,None]
#   min_memory = X[:,1][:,None]
#   max_memory = X[:,2][:,None]
#   cache_size = X[:,3][:,None]
#   min_channels = X[:,4][:,None]
#   max_channels = X[:,5][:,None]

#   avg_channels = ((min_channels + max_channels)/2)
#   avg_memory = ((min_memory + max_memory)/2)
#   ratio = cache_size/avg_memory

#   return np.hstack((cycle_time, ratio, avg_channels))

# def average_channels_and_cycle_time_to_clock_speed_and_cache_average_memory_ratio(X):
#   cycle_time = X[:,0][:,None]
#   min_memory = X[:,1][:,None]
#   max_memory = X[:,2][:,None]
#   cache_size = X[:,3][:,None]
#   min_channels = X[:,4][:,None]
#   max_channels = X[:,5][:,None]

#   avg_channels = ((min_channels + max_channels)/2)
#   avg_memory = ((min_memory + max_memory)/2)
#   ratio = cache_size/avg_memory
#   clock_speed = (1000000000/cycle_time)

#   return np.hstack((clock_speed, ratio, avg_channels))

# def cycle_time_to_clock_speed_and_cache_average_memory_ratio(X):
#   cycle_time = X[:,0][:,None]
#   min_memory = X[:,1][:,None]
#   max_memory = X[:,2][:,None]
#   cache_size = X[:,3][:,None]
#   min_channels = X[:,4][:,None]
#   max_channels = X[:,5][:,None]

#   avg_memory = ((min_memory + max_memory)/2)
#   ratio = cache_size/avg_memory
#   clock_speed = (1000000000/cycle_time)

#   return np.hstack((clock_speed, ratio, min_channels, max_channels))

## Ratio combinations were removed as they consistently produced poor predictions.

possible_combinations = {tuple(['MYCT','MMIN','MMAX','CACH','CHMIN','CHMAX']):base_features, 
                        tuple(['MYCT','MAVG','CACH','CHMIN','CHMAX']):average_memory, 
                        tuple(['MYCT','MMIN','MMAX','CACH','CHAVG']):average_channels, 
                        tuple(['SPEED','MMIN','MMAX','CACH','CHMIN','CHMAX']):cycle_time_to_clock_speed, 
                        # tuple(['MYCT','RATIO','CHMIN','CHMAX']):cache_average_memory_ratio,
                        tuple(['MYCT','MAVG','CACH','CHAVG']):average_memory_and_average_channels, 
                        tuple(['SPEED','MAVG','CACH','CHMIN','CHMAX']):average_memory_and_cycle_time_to_clock_speed, 
                        tuple(['SPEED','MAVG','CACH','CHAVG']):average_memory_and_cycle_time_to_clock_speed_and_average_channels,
                        tuple(['SPEED','MMIN','MMAX','CACH','CHAVG']):average_channels_and_cycle_time_to_clock_speed}
                        # tuple(['MYCT','RATIO','CHAVG']):average_channels_and_cache_average_memory_ratio,
                        # tuple(['SPEED','RATIO','CHAVG']):average_channels_and_cycle_time_to_clock_speed_and_cache_average_memory_ratio,
                        # tuple(['SPEED','RATIO','CHMIN','CHMAX']):cycle_time_to_clock_speed_and_cache_average_memory_ratio} 

print(type(possible_combinations))
print(possible_combinations)

