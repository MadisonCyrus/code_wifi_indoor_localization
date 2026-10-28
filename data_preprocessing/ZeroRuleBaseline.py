import os
import kagglehub 
import numpy as np 
import pandas as pd

# Download latest version
path = kagglehub.dataset_download("giantuji/UjiIndoorLoc")

print("Path to dataset files:", path)

for root, _, files in os.walk(path):
    for name in files:
        print(os.path.join(root, name))

# UJIIndoorLoc ships as trainingData.csv and validationData.csv
train_path = os.path.join(path, "trainingData.csv")
val_path = os.path.join(path, "validationData.csv")

train = pd.read_csv(train_path)
val = pd.read_csv(val_path)

print(train.shape, val.shape)

# Columns 0-519 are WAP RSSI readings (100 = signal not detected)
X = train.iloc[:, :520].to_numpy()
X = np.where(X == 100, -110, X)  # replace "not detected" with a very weak signal

# Targets: LONGITUDE, LATITUDE, FLOOR, BUILDINGID
y = train[["LONGITUDE", "LATITUDE", "FLOOR", "BUILDINGID"]].to_numpy()
  
def zero_rule_algorithm_classification(train, test):
  # Extract target column (assumed to be the last column)
  output_values = [row[-1] for row in train]

  # Find the mode (most common class value)
  prediction = max(set(output_values), key=output_values.count)

  # Predict the majority class for all test instances
  predicted = [prediction for _ in range(len(test))]
  return predicted


