from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity
import numpy as np
from scipy.spatial import distance

output2_fhe = []
pool2_fhe = []

with open("./muld2/output2.txt") as file:
  for line in file.readlines():
    output2_fhe.append(float(line[1:].split(',')[0]))

with open("./muld2/pool2.txt") as file:
  for line in file.readlines():
    pool2_fhe.append(float(line[1:].split(',')[0]))

output2 = []
pool2 = []

with open("./nonfheouts/pool2.txt") as file:
  for line in file.readlines():
    pool2.append(float(line.strip()))

with open("./nonfheouts/output2.txt") as file:
  for line in file.readlines():
    output2.append(float(line.strip()))

euclidean_dist = distance.euclidean(output2, output2_fhe)
print("Euclidean Distance output2:", euclidean_dist)

cosine_sim = 1 - distance.cosine(output2, output2_fhe)
print("Cosine Similarity output2:", cosine_sim)

euclidean_dist = distance.euclidean(pool2, pool2_fhe)
print("Euclidean Distance pool2:", euclidean_dist)

cosine_sim = 1 - distance.cosine(pool2, pool2_fhe)
print("Cosine Similarity pool2:", cosine_sim)
