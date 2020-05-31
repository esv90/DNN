import numpy as np
import random

n_features = 10
batch_size = 3

division = n_features / batch_size  # Will seldom be a natural number
feature_indices = np.arange(n_features)
random.shuffle(feature_indices)

# Create the mini batches
mini_batches = [feature_indices[round(division*i):round(division * (i+1))] for i in range(batch_size)]
#random.shuffle(mini_batches)
print(mini_batches)
