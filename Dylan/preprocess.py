import pandas as pd
import numpy as np
from PIL import Image
import os
from sklearn.model_selection import train_test_split


def load_images(dataframe, folder):
    images = []
    labels = []
    for i, row in dataframe.iterrows():
        print(i)
        filename = row['image'] 
        label = row['level']  
        file_path = os.path.join(folder, filename + '.jpeg')
        image = Image.open(file_path)
        image = image.resize((256, 256))
        images.append(np.array(image))
        labels.append(label)
    return np.array(images), np.array(labels)

train_df, test_df = train_test_split(pd.read_csv('Dataset/trainLabels_cropped.csv'), test_size=0.2)
print(train_df)
X_train, y_train = load_images(train_df, 'Dataset/resized_train_cropped/resized_train_cropped')
X_test, y_test = load_images(test_df, 'Dataset/resized_train_cropped/resized_train_cropped')

np.save('Dataset/numpy_files/X_train.npy', X_train)
np.save('Dataset/numpy_files/y_train.npy', y_train)
np.save('Dataset/numpy_files/X_test.npy', X_test)
np.save('Dataset/numpy_files/y_test.npy', y_test)