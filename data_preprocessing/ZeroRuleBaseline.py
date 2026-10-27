
import kagglehub 
import numpy as np 

# Download latest version
path = kagglehub.dataset_download("giantuji/UjiIndoorLoc")

print("Path to dataset files:", path)


def zero_rule_algorithm_classification(train, test):
  # Extract target column (assumed to be the last column)
  output_values = [row[-1] for row in train]

  # Find the mode (most common class value)
  prediction = max(set(output_values), key=output_values.count)

  # Predict the majority class for all test instances
  predicted = [prediction for _ in range(len(test))]
  return predicted


