import numpy as np
import pandas as pd

def extract_labels(input = '/home/group4/Challenge3/vtp_analysis/outputs/dataset/dataset.csv', output = '/home/group4/Challenge3/vtp_analysis/outputs/dataset/outcomes.csv'):
    labels = pd.read_csv(input)["complication_raw"].to_numpy().reshape([-1, 1])
    print(labels)
    print(labels.shape)
    output_labels = (labels != 'none').astype(int)
    print(output_labels.shape)
    print(output_labels)
    np.save(output, output_labels)
    return output_labels