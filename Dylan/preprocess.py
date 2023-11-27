import pandas as pd
import numpy as np
from PIL import Image
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def load_images(dataframe, folder, n=1):
    images = []
    labels = []
    count = 0  
    for _, row in dataframe.iterrows():
        if count % n == 0:  # Take every nth image
            print(count)
            filename = row['image']
            label = row['level']
            file_path = os.path.join(folder, filename + '.jpeg')
            image = Image.open(file_path)
            image = image.resize((256, 256))
            #show image
            plt.imshow(image)
            print(label)
            plt.show()
            images.append(np.array(image))
            labels.append(label)
        count += 1  # Increment the counter
    return np.array(images), np.array(labels)

train_df, test_df = train_test_split(pd.read_csv('Dataset/trainLabels.csv'), test_size=0.2)

#find the number of images in each class as proportion
print(train_df['level'].value_counts(normalize=True))
# print(train_df['level'].value_counts())

# print(train_df)
# X_train, y_train = load_images(train_df, 'Dataset/resized_train_cropped/resized_train_cropped',5)
# X_test, y_test = load_images(test_df, 'Dataset/resized_train_cropped/resized_train_cropped', 5)

# np.save('Dataset/numpy_files/X_train_5.npy', X_train)
# np.save('Dataset/numpy_files/y_train_5.npy', y_train)
# np.save('Dataset/numpy_files/X_test_5.npy', X_test)
# np.save('Dataset/numpy_files/y_test_5.npy', y_test)