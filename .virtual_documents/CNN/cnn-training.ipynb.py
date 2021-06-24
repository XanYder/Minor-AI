# Save this file in the name of the ML model you are using
import pandas as pd
import numpy as np
import os


# Website for the dataset:
# https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data?select=icml_face_data.csv
CLASS_NAMES = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
CLASS_NAMES_WITHOUT_DISGUST = ["Angry", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
FILE_NAME = "emotions.csv" # Insert file name
WHITE_IMAGES = [6458,  7629, 10423, 11286, 13148, 13402, 13988, 15894, 22198, 22927, 28601, 59]


data_path = ["../data"] # Insert data file path
file_path = os.sep.join(data_path + [FILE_NAME])
data = pd.read_csv(file_path)
data = data.drop(index=WHITE_IMAGES, axis=0)

data.drop('Usage', axis=1, inplace=True)

data.head()


data.shape


data.info()


data.describe()


from collections import Counter

print('number of samples: ', len(data))
print('number of unique samples: ', len(data[data.columns[1]].unique()))
print('keys: ', list(data.keys()))
print('\n')

for i in range(len(CLASS_NAMES)):
    print(CLASS_NAMES[i] + ' ', ((data['emotion'].value_counts())[i]))



import matplotlib.pyplot as plt
get_ipython().run_line_magic("matplotlib", " inline")

# (0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral)
data_Angry = data[data['emotion'] == 0]
data_Disgust = data[data['emotion'] == 1]
data_Fear = data[data['emotion'] == 2]
data_Happy = data[data['emotion'] == 3]
data_Sad = data[data['emotion'] == 4]
data_Surprise = data[data['emotion'] == 5]
data_Neutral = data[data['emotion'] == 6]

sizes = [data_Angry.shape[0], data_Disgust.shape[0], data_Fear.shape[0], data_Happy.shape[0], data_Sad.shape[0], data_Surprise.shape[0], data_Neutral.shape[0]]

fig, ax = plt.subplots()
ax.pie(sizes, labels=CLASS_NAMES, autopct='get_ipython().run_line_magic("1.1f%%')", "")
ax.axis('equal')  # Equal aspect ratio ensures the pie chart is circular.
ax.set_title('Emotions')


plt.show()

sizes = [data_Angry.shape[0], data_Fear.shape[0], data_Happy.shape[0], data_Sad.shape[0], data_Surprise.shape[0], data_Neutral.shape[0]]
labels = ['Angry', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral' ]

fig, ax = plt.subplots()
ax.pie(sizes, labels=labels, autopct='get_ipython().run_line_magic("1.1f%%')", "")
ax.axis('equal')  # Equal aspect ratio ensures the pie chart is circular.
ax.set_title('Emotions without Disgust')


plt.show()


data_exclude_disgust = data[data['emotion'] get_ipython().getoutput("= 1] # Drops the emotion Disgust")
data_exclude_disgust.shape


data_exclude_disgust = data_exclude_disgust.replace({
    2 : 1,
    3 : 2,
    4 : 3,
    5 : 4,
    6 : 5
})


data_exclude_disgust.emotion.unique()


def data_X_y(data):
    # Split data into X & y
    X = data.drop('emotion', axis='columns')
    y = data['emotion']

    # Reshapes X into 3D array
    X = [pixels.split(" ") for pixels in data["pixels"]]
    X = np.array(X)
    X = X.astype("int32")
    X = np.array([image.reshape(48, 48) for image in X])
    X = X/255.0
    X = X.reshape(len(X), 48, 48, 1)
    
    return X,y


X,y = data_X_y(data_exclude_disgust)


print(X.max())
print(X.shape)
print(X.min(), X.max())
print(y.shape)
print(y.unique())


from skimage.io import imread
from skimage.transform import resize

def show_samples(X):
    # Array with all the unique emotions
    labels = np.unique(data_exclude_disgust['emotion'])

    fig, axes = plt.subplots(6, len(labels))
    fig.set_size_inches(15,10)
    fig.tight_layout()

    for ax, label in zip(axes, labels):
        i = 0 # starting point
        for a in ax:
            # A list with a the index for the corresponding emotion
            data_label_index_list = data_exclude_disgust.index[data_exclude_disgust['emotion'] == label].tolist()

            a.imshow(X[data_label_index_list[i]])
            a.axis('off')
            a.set_title(CLASS_NAMES_WITHOUT_DISGUST[label])
            i=i+1



show_samples(X)


from skimage.feature import hog
from skimage.io import imread

def show_hog_samples(X):
    labels = np.unique(data_exclude_disgust['emotion'])

    fig, axes = plt.subplots(1, len(labels))
    fig.set_size_inches(15,4)
    fig.tight_layout()

    for ax, label in zip(axes, labels):

        data_label_index_list = data_exclude_disgust.index[data_exclude_disgust['emotion'] == label].tolist()
        
        # The values below can be changes to decrease or increase the amount of details
        emotion_hog, emotion_hog_img = hog(
        X[data_label_index_list[0]],
        pixels_per_cell=(2, 2),
        cells_per_block=(2, 2), 
        orientations=9, 
        visualize=True, 
        block_norm='L2-Hys')
        
        ax.imshow(emotion_hog_img)
        ax.axis('off')
        ax.set_title(CLASS_NAMES_WITHOUT_DISGUST[label])


show_hog_samples(X)


import tensorflow as tf
from tensorflow.keras import layers

# For more you can visit:
# https://www.tensorflow.org/api_docs/python/tf/keras/layers/experimental/preprocessing
data_augmentation = tf.keras.Sequential([
    layers.experimental.preprocessing.RandomFlip("horizontal"),
    layers.experimental.preprocessing.RandomRotation(0.25),
    layers.experimental.preprocessing.RandomZoom(0.25),
])

image = tf.expand_dims(X[0], 0) # X[0] can be changed to view different images

plt.figure(figsize=(10, 10))
for i in range(9):
    augmented_image = data_augmentation(image)
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(augmented_image[0])
    plt.axis("off")


from tqdm import tqdm
def join_list(list_pixels):
    # Creates a string from the list of pixels
    final_str = ' '.join(str(int(v)) for v in list_pixels)
    return final_str

def create_csv_data_augmentation(X, y, copies):
    # Creates new dataframe
    column_names = ["emotion", "pixels", "Original"]
    data_augmentated = pd.DataFrame(columns = column_names)
    
    for image,emotion in tqdm(zip(X,y)):
        # Adds the original image to the dataframe converted to a string
        image_array = list((np.array(image)).flat)
        image_string = join_list(image_array)
        new_row = {'emotion':emotion, 'pixels':image_string, 'Original':True}
        data_augmentated = data_augmentated.append(new_row,ignore_index=True)
        image_expand = tf.expand_dims(image, 0)

        for i in range(copies): # tqdm to view progress
            # Adds the augmented image to the dataframe converted to a string
            augmented_image = data_augmentation(image_expand)
            augmented_image = list((np.array(augmented_image[0])).flat)
            augmented_image_string = join_list(augmented_image)
            new_row = {'emotion':emotion, 'pixels':augmented_image_string, 'Original':False}
            data_augmentated = data_augmentated.append(new_row, ignore_index=True)
    
    # Saves the dataframe to a csv file and in the title the amount of corresponding data augmentations
    save_data_augmentated_filepath = ('data/face_augmentated_{}'.format(copies))
    data_augmentated.to_csv(save_data_augmentated_filepath + '.csv', index = False)

    return data_augmentated


# # You can adjust the last value to choose how many unique copies you want to make
# # The higher the copies the longer the run time
# data_aug = create_csv_data_augmentation((X*255.0), y, 1) 


# # For when the file is created
FILE_NAME_AUG = "face_augmentated_1.csv" # Update name
file_path_aug = os.sep.join(data_path + [FILE_NAME_AUG])
data_aug = pd.read_csv(file_path_aug)


data_aug.shape


data_aug.head()


data_aug.drop('Original', axis=1, inplace=True)
X_aug, y_aug = data_X_y(data_aug)
y_aug = y_aug.astype('int32')

print(X_aug.shape)
print(y_aug.shape)


plt.figure(figsize=(10, 10))
for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(X_aug[i])
    plt.axis("off")


from sklearn.base import BaseEstimator, TransformerMixin

class HogTransformer(BaseEstimator, TransformerMixin):
    """
    Expects an array of 2d arrays (1 channel images)
    Calculates hog features for each img
    """
 
    def __init__(self, y=None, orientations=9,
                 pixels_per_cell=(8, 8),
                 cells_per_block=(3, 3), block_norm='L2-Hys'):
        self.y = y
        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block
        self.block_norm = block_norm
 
    def fit(self, X, y=None):
        return self
 
    def transform(self, X, y=None):
 
        def local_hog(X):
            return hog(X,
                       orientations=self.orientations,
                       pixels_per_cell=self.pixels_per_cell,
                       cells_per_block=self.cells_per_block,
                       block_norm=self.block_norm)
 
        try: # parallel
            return np.array([local_hog(img) for img in X])
        except:
            return np.array([local_hog(img) for img in X])



from skimage.feature import hog
from sklearn.preprocessing import StandardScaler
import skimage

def hogify_X(X):
    hogify = HogTransformer(
        pixels_per_cell=(14, 14), 
        cells_per_block=(2,2), 
        orientations=9, 
        block_norm='L2-Hys'
    )

    X_hog = hogify.fit_transform(X)

    return X_hog


X_hog = hogify_X(X) # This can be used for fitting a model faster with less data but still good in quality, test it with the other data samples
print(X_hog.shape)


from builtins import range
from builtins import object

def reshape_X(X):
    num_training = X.shape[0]
    mask = list(range(num_training))
    X_reshape = X[mask]

    # Reshape the image data into rows
    X_reshape = np.reshape(X, (X.shape[0], -1))
    
    return X_reshape


print(X.shape)
X = reshape_X(X)
print(X.shape)


print(X_aug.shape)
X_aug = reshape_X(X_aug)
print(X_aug.shape)


# pip install -U imbalanced-learn
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE

rus = RandomUnderSampler()
ros = RandomOverSampler()
smote = SMOTE()

X_rus, y_rus = rus.fit_resample(X,y) # This data is undersampled
X_ros, y_ros = ros.fit_resample(X,y) # This data is oversampled
X_smote, y_smote = smote.fit_resample(X,y) # This data is oversampled with smote

print(y_rus.value_counts())
print(y_ros.value_counts())
print(y_smote.value_counts())


# Checks the unique rows in the array for smote, ros & rus
print(len(X_smote))
print(len(np.unique(X_smote, axis=0)))
print(len(X_ros))
print(len(np.unique(X_ros, axis=0)))
print(len(X_rus))
print(len(np.unique(X_rus, axis=0)))


X_aug_smote, y_aug_smote = smote.fit_resample(X_aug,y_aug) # This data is oversampled with smote
X_aug_ros, y_aug_ros = ros.fit_resample(X_aug,y_aug) # This data is oversampled
X_aug_rus, y_aug_rus = rus.fit_resample(X_aug,y_aug) # This data is undersampled

print(y_aug_rus.value_counts())
print(y_aug_ros.value_counts())
print(y_aug_smote.value_counts())


def plot_train_test_distribution(y,loc='left', relative=True):
    width = 0.35
    CLASS_NAMES_WITHOUT_DISGUST = ["Angry", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

    if loc == 'left':
        n = -0.5
    elif loc == 'right':
        n = 0.5
     
    # calculate counts per type and sort, to ensure their order
    unique, counts = np.unique(y, return_counts=True)
    sorted_index = np.argsort(unique)

    if relative:
        # plot as a percentage
        counts = 100*counts[sorted_index]/len(y)
        ylabel_text = '% count'
    else:
        # plot counts
        counts = counts[sorted_index]
        ylabel_text = 'count'
         
    xtemp = np.arange(len(unique))
    
    plt.bar(xtemp + n*width, counts, align='center', alpha=.7, width=width)
    plt.xticks(xtemp, CLASS_NAMES_WITHOUT_DISGUST, rotation=45)
    plt.xlabel('equipment type')
    plt.ylabel(ylabel_text)
    plt.suptitle('relative amount of images per type')
    


from sklearn.model_selection import train_test_split
 
X_train, X_test, y_train, y_test = train_test_split(
    X, 
    y, 
    test_size=0.2, 
    shuffle=True,
    random_state=42,
)

plot_train_test_distribution(y_train, loc='left')
plot_train_test_distribution(y_test, loc='right')
plt.legend([
    'train ({0} photos)'.format(len(y_train)), 
    'test ({0} photos)'.format(len(y_test))
]);


X_train_rus, X_test_rus, y_train_rus, y_test_rus = train_test_split(
    X_rus, 
    y_rus, 
    test_size=0.2, 
    shuffle=True,
    random_state=42,
)

plot_train_test_distribution(y_train_rus, loc='left')
plot_train_test_distribution(y_test_rus, loc='right')
plt.legend([
    'train ({0} photos)'.format(len(y_train_rus)), 
    'test ({0} photos)'.format(len(y_test_rus))
]);


X_train_ros, X_test_ros, y_train_ros, y_test_ros = train_test_split(
    X_ros, 
    y_ros, 
    test_size=0.2, 
    shuffle=True,
    random_state=42,
)

plot_train_test_distribution(y_train_ros, loc='left')
plot_train_test_distribution(y_test_ros, loc='right')
plt.legend([
    'train ({0} photos)'.format(len(y_train_ros)), 
    'test ({0} photos)'.format(len(y_test_ros))
]);


X_train_smote, X_test_smote, y_train_smote, y_test_smote = train_test_split(
    X_smote, 
    y_smote, 
    test_size=0.2, 
    shuffle=True,
    random_state=42,
)

plot_train_test_distribution(y_train_smote, loc='left')
plot_train_test_distribution(y_test_smote, loc='right')
plt.legend([
    'train ({0} photos)'.format(len(y_train_smote)), 
    'test ({0} photos)'.format(len(y_test_smote))
]);


# These are the X & y that can be used for fitting a ML model
X_train, X_test, y_train, y_test
X_train_rus, X_test_rus, y_train_rus, y_test_rus
X_train_ros, X_test_ros, y_train_ros, y_test_ros
X_train_smote, X_test_smote, y_train_smote, y_test_smote
X_hog, y # This does still have to be ros, rus or smote for balance and then split in to train & test


# This is the augmented dataset
# Has to be train & test split before use
X_aug_smote, y_aug_smote
X_aug_ros, y_aug_ros
X_aug_rus, y_aug_rus


from keras.models import Model
import keras
import csv
from PIL import Image    
from sklearn.model_selection import train_test_split
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D,BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D, Dropout
from keras import regularizers
import numpy as np # linear algebra
from numpy import asarray
from sklearn.preprocessing import OneHotEncoder
from sklearn.datasets import make_classification
from imblearn.over_sampling import RandomOverSampler
import collections
import pandas as pd 
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.preprocessing import image
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.layers import Dropout
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split 
import os


checkpoint_path = "training/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True)
es_callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)

CLASS_NAMES_WITHOUT_DISGUST = ["Angry", "Fear", "Happy", "Sad", "Surprise", "Neutral"]


X_train = X_train.reshape(len(X_train), 48, 48, 1)
X_test = X_test.reshape(len(X_test), 48, 48, 1)


tf.random.set_seed(42)
input_shape = (48, 48, 1)
model = models.Sequential([
    Conv2D(64, (1, 1), activation = "relu", input_shape = input_shape),
    Conv2D(128, (3, 3), activation = "relu"),
    MaxPooling2D((2, 2)),
    Flatten(),
    Activation("relu"),
    Dense(128),
    Dense(len(CLASS_NAMES_WITHOUT_DISGUST), activation = "softmax")
])
model.summary()


epochs = 50
model.compile(optimizer = tf.keras.optimizers.Adam(lr = 0.001),  
              loss=tf.keras.losses.sparse_categorical_crossentropy,
              metrics = ["accuracy"])

history = model.fit(X_train, y_train, 
                    batch_size = 32, 
                    epochs = epochs, 
                    steps_per_epoch = len(X_train)/128,
                    validation_data = (X_test, y_test),
                    callbacks = [es_callback, cp_callback])

model.save("CNNModelx-1")


tf.random.set_seed(42)
input_shape = (48, 48, 1)
model = models.Sequential([
    Conv2D(64, (1, 1), activation = "relu", input_shape = input_shape),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation = "relu"),
    MaxPooling2D((2, 2)),
    Conv2D(256, (3, 3), activation = "relu"),
    MaxPooling2D((2, 2)),
    Flatten(),
    Activation("relu"),
    Dense(128),
    Dense(len(CLASS_NAMES_WITHOUT_DISGUST), activation = "softmax")
])
model.summary()


epochs = 50
model.compile(optimizer = tf.keras.optimizers.Adam(lr = 0.001),  
              loss=tf.keras.losses.sparse_categorical_crossentropy,
              metrics = ["accuracy"])

history = model.fit(X_train, y_train, 
                    batch_size = 32, 
                    epochs = epochs, 
                    steps_per_epoch = len(X_train)/128,
                    validation_data = (X_test, y_test),
                    callbacks = [es_callback, cp_callback])

model.save("CNNModelx-2")


tf.random.set_seed(42)
input_shape = (48, 48, 1)
model = models.Sequential([
    Conv2D(64, (1, 1), padding = "same", activation = "relu", input_shape = input_shape),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), padding = "same", activation = "relu"),
    MaxPooling2D((2, 2)),
    Conv2D(256, (3, 3), padding = "same", activation = "relu"),
    MaxPooling2D((2, 2)),
    Flatten(),
    Activation("relu"),
    Dense(128),
    Dense(len(CLASS_NAMES_WITHOUT_DISGUST), activation = "softmax")
])
model.summary()


epochs = 50
model.compile(optimizer = tf.keras.optimizers.Adam(lr = 0.001),  
              loss=tf.keras.losses.sparse_categorical_crossentropy,
              metrics = ["accuracy"])

history = model.fit(X_train, y_train, 
                    batch_size = 32, 
                    epochs = epochs, 
                    steps_per_epoch = len(X_train)/128,
                    validation_data = (X_test, y_test),
                    callbacks = [es_callback, cp_callback])

model.save("CNNModelx-3")


tf.random.set_seed(42)
input_shape = (48, 48, 1)
model = models.Sequential([
    Conv2D(64, (1, 1), padding = "same", activation = "relu", input_shape = input_shape),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), padding = "same", activation = "relu"),
    MaxPooling2D((2, 2)),
    Conv2D(256, (3, 3), padding = "same", activation = "relu"),
    MaxPooling2D((2, 2)),
    Flatten(),
    Activation("relu"),
    Dense(256),
    Dense(len(CLASS_NAMES_WITHOUT_DISGUST), activation = "softmax")
])
model.summary()


epochs = 50
model.compile(optimizer = tf.keras.optimizers.Adam(lr = 0.001),  
              loss=tf.keras.losses.sparse_categorical_crossentropy,
              metrics = ["accuracy"])

history = model.fit(X_train, y_train, 
                    batch_size = 32, 
                    epochs = epochs, 
                    steps_per_epoch = len(X_train)/128,
                    validation_data = (X_test, y_test),
                    callbacks = [es_callback, cp_callback])

model.save("CNNModelx-4")


tf.random.set_seed(42)
input_shape = (48, 48, 1)
model = models.Sequential([
    Conv2D(64, (1, 1), padding = "same", activation = "relu", input_shape = input_shape),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), padding = "same", activation = "relu"),
    MaxPooling2D((2, 2)),
    Conv2D(256, (3, 3), padding = "same", activation = "relu"),
    MaxPooling2D((2, 2)),
    Flatten(),
    Activation("relu"),
    Dense(256),
    Dense(len(CLASS_NAMES_WITHOUT_DISGUST), activation = "softmax")
])
model.summary()


epochs = 50
model.compile(optimizer = tf.keras.optimizers.Adam(lr = 0.001),  
              loss=tf.keras.losses.sparse_categorical_crossentropy,
              metrics = ["accuracy"])

history = model.fit(X_train, y_train, 
                    batch_size = 64, 
                    epochs = epochs, 
                    steps_per_epoch = len(X_train)/128,
                    validation_data = (X_test, y_test),
                    callbacks = [es_callback, cp_callback])

model.save("CNNModelx-5")


tf.random.set_seed(42)
input_shape = (48, 48, 1)
model = models.Sequential([
    Conv2D(64, (1, 1), padding = "same", activation = "relu", input_shape = input_shape),
    Dropout(.25),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), padding = "same", activation = "relu"),
    Dropout(.25),
    MaxPooling2D((2, 2)),
    Conv2D(256, (3, 3), padding = "same", activation = "relu"),
    Dropout(.25),
    MaxPooling2D((2, 2)),
    Flatten(),
    Activation("relu"),
    Dense(256),
    Dense(len(CLASS_NAMES_WITHOUT_DISGUST), activation = "softmax")
])
model.summary()


epochs = 50
model.compile(optimizer = tf.keras.optimizers.Adam(lr = 0.001),  
              loss=tf.keras.losses.sparse_categorical_crossentropy,
              metrics = ["accuracy"])

history = model.fit(X_train, y_train, 
                    batch_size = 64, 
                    epochs = epochs, 
                    steps_per_epoch = len(X_train)/128,
                    validation_data = (X_test, y_test),
                    callbacks = [es_callback, cp_callback])

model.save("CNNModelx-6")


tf.random.set_seed(42)
input_shape = (48, 48, 1)
model = models.Sequential([
    Conv2D(64, (1, 1), padding = "same", activation = "relu", input_shape = input_shape),
    Dropout(.25),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), padding = "same", activation = "relu"),
    Dropout(.25),
    MaxPooling2D((2, 2)),
    Conv2D(256, (3, 3), padding = "same", activation = "relu"),
    Dropout(.25),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128),
    Activation("relu"),
    Dropout(.25),
    Dense(256),
    Dense(len(CLASS_NAMES_WITHOUT_DISGUST), activation = "softmax")
])
model.summary()


epochs = 50
model.compile(optimizer = tf.keras.optimizers.Adam(lr = 0.001),  
              loss=tf.keras.losses.sparse_categorical_crossentropy,
              metrics = ["accuracy"])

history = model.fit(X_train, y_train, 
                    batch_size = 64, 
                    epochs = epochs, 
                    steps_per_epoch = len(X_train)/128,
                    validation_data = (X_test, y_test),
                    callbacks = [es_callback, cp_callback])

model.save("CNNModelx-7")


tf.random.set_seed(42)
input_shape = (48, 48, 1)
model = models.Sequential([
    Conv2D(64, (1, 1), padding = "same", activation = "relu", input_shape = input_shape),
    BatchNormalization(),
    Dropout(.25),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), padding = "same", activation = "relu"),
    BatchNormalization(),
    Dropout(.25),
    MaxPooling2D((2, 2)),
    Conv2D(256, (3, 3), padding = "same", activation = "relu"),
    BatchNormalization(),
    Dropout(.25),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128),
    Activation("relu"),
    Dropout(.25),
    Dense(256),
    Dense(len(CLASS_NAMES_WITHOUT_DISGUST), activation = "softmax")
])
model.summary()


epochs = 50
model.compile(optimizer = tf.keras.optimizers.Adam(lr = 0.001),  
              loss=tf.keras.losses.sparse_categorical_crossentropy,
              metrics = ["accuracy"])

history = model.fit(X_train, y_train, 
                    batch_size = 64, 
                    epochs = epochs, 
                    steps_per_epoch = len(X_train)/128,
                    validation_data = (X_test, y_test),
                    callbacks = [es_callback, cp_callback])

model.save("CNNModelx-8")


tf.random.set_seed(42)
input_shape = (48, 48, 1)
model = models.Sequential([
    Conv2D(64, (1, 1), padding = "same", activation = "relu", input_shape = input_shape, kernel_regularizer = regularizers.l2(l = 0.01)),
    Dropout(.25),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), padding = "same", activation = "relu", kernel_regularizer = regularizers.l2(l = 0.01)),
    Dropout(.25),
    MaxPooling2D((2, 2)),
    Conv2D(256, (3, 3), padding = "same", activation = "relu", kernel_regularizer = regularizers.l2(l = 0.01)),
    Dropout(.25),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, kernel_regularizer = regularizers.l2(l = 0.01)),
    Activation("relu"),
    Dropout(.25),
    Dense(256, kernel_regularizer = regularizers.l2(l = 0.01)),
    Dense(len(CLASS_NAMES_WITHOUT_DISGUST), activation = "softmax")
])
model.summary()


epochs = 50
model.compile(optimizer = tf.keras.optimizers.Adam(lr = 0.001),  
              loss=tf.keras.losses.sparse_categorical_crossentropy,
              metrics = ["accuracy"])

history = model.fit(X_train, y_train, 
                    batch_size = 64, 
                    epochs = epochs, 
                    steps_per_epoch = len(X_train)/128,
                    validation_data = (X_test, y_test),
                    callbacks = [es_callback, cp_callback])

model.save("CNNModelx-9")


X_train_ros = X_train_ros.reshape(len(X_train_ros), 48, 48, 1)


X_test_ros = X_test_ros.reshape(len(X_test_ros), 48, 48, 1)


tf.random.set_seed(42)
input_shape = (48, 48, 1)
model = models.Sequential([
    Conv2D(64, (1, 1), padding = "same", activation = "relu", input_shape = input_shape),
    BatchNormalization(),
    Dropout(.25),
    Conv2D(128, (3, 3), padding = "same", activation = "relu"),
    BatchNormalization(),
    Dropout(.25),
    Conv2D(256, (5, 5), padding = "same", activation = "relu"),
    BatchNormalization(),
    MaxPooling2D((2, 2), padding = "same"),
    Dropout(.25),
    Flatten(),
    Dense(128),
    BatchNormalization(),
    Activation("relu"),
    Dropout(.25),
    Dense(256),
    Dense(len(CLASS_NAMES_WITHOUT_DISGUST), activation = "softmax")
])
model.summary()


epochs = 50
model.compile(optimizer = tf.keras.optimizers.Adam(lr = 0.006),  
              loss=tf.keras.losses.sparse_categorical_crossentropy,
              metrics = ["accuracy"])

history = model.fit(X_train_ros, y_train_ros, 
                    batch_size = 64, 
                    epochs = epochs, 
                    steps_per_epoch = len(X_train_ros)/128,
                    validation_data = (X_test_ros, y_test_ros),
                    callbacks = [es_callback, cp_callback])

model.save("CNNModel3")


model = tf.keras.models.load_model("models/CNNModel3")


# pip install python-resize-image
from resizeimage import resizeimage
from PIL import Image, ImageOps

def import_test_sample(test_path):
    with open(test_path, 'r+b') as f:
        with Image.open(f) as image:
            test_image = resizeimage.resize_cover(image, [48,48])

    test_image = ImageOps.grayscale(test_image)

    test_image = np.array(test_image)
    test_image = test_image.astype("int32")
    test_image = test_image/255.0
    test_image = test_image.reshape(48, 48, 1)
    plt.imshow(test_image)
    print(test_image.shape)
    
    return test_image



X_train_smote = X_train_smote.reshape(len(X_train_smote), 48, 48, 1)
X_test_smote = X_test_smote.reshape(len(X_test_smote), 48, 48, 1)


tf.random.set_seed(42)
input_shape = (48, 48, 1)
model = models.Sequential([
    Conv2D(64, (1, 1), padding = "same", activation = "relu", input_shape = input_shape),
    BatchNormalization(),
    Dropout(.25),
    Conv2D(128, (3, 3), padding = "same", activation = "relu"),
    BatchNormalization(),
    Dropout(.25),
    Conv2D(256, (5, 5), padding = "same", activation = "relu"),
    BatchNormalization(),
    MaxPooling2D((2, 2), padding = "same"),
    Dropout(.25),
    Flatten(),
    Dense(128),
    BatchNormalization(),
    Activation("relu"),
    Dropout(.25),
    Dense(256),
    Dense(len(CLASS_NAMES_WITHOUT_DISGUST), activation = "softmax")
])


epochs = 50
model.compile(optimizer = tf.keras.optimizers.Adam(lr = 0.006),  
              loss=tf.keras.losses.sparse_categorical_crossentropy,
              metrics = ["accuracy"])

history = model.fit(X_train_smote, y_train_smote, 
                    batch_size = 64, 
                    epochs = epochs, 
                    steps_per_epoch = len(X_train_smote)/128,
                    validation_data = (X_test_smote, y_test_smote),
                    callbacks = [es_callback, cp_callback])

model.save("CNNModel4")


model.evaluate(X_test_smote, y_test_smote)


y_aug_ros = y_aug_ros.replace({
    2 : 1,
    3 : 2,
    4 : 3,
    5 : 4,
    6 : 5
})


X_train_aug_ros, X_test_aug_ros, y_train_aug_ros, y_test_aug_ros = train_test_split(
    X_aug_ros, 
    y_aug_ros, 
    test_size=0.2, 
    shuffle=True,
    random_state=42,
)


X_train_aug_ros.shape


X_train_aug_ros = X_train_aug_ros.reshape(len(X_train_aug_ros), 48, 48, 1)
X_test_aug_ros = X_test_aug_ros.reshape(len(X_test_aug_ros), 48, 48, 1)


X_train_aug_ros.shape


tf.random.set_seed(42)
input_shape = (48, 48, 1)
model = models.Sequential([
    Conv2D(64, (1, 1), padding = "same", activation = "relu", input_shape = input_shape),
    BatchNormalization(),
    Dropout(.25),
    Conv2D(128, (3, 3), padding = "same", activation = "relu"),
    BatchNormalization(),
    Dropout(.25),
    Conv2D(256, (5, 5), padding = "same", activation = "relu"),
    BatchNormalization(),
    MaxPooling2D((2, 2), padding = "same"),
    Dropout(.25),
    Flatten(),
    Dense(128),
    BatchNormalization(),
    Activation("relu"),
    Dropout(.25),
    Dense(256),
    Dense(len(CLASS_NAMES_WITHOUT_DISGUST), activation = "softmax")
])


epochs = 50
model.compile(optimizer = tf.keras.optimizers.Adam(lr = 0.006),  
              loss=tf.keras.losses.sparse_categorical_crossentropy,
              metrics = ["accuracy"]) 

history = model.fit(X_train_aug_ros, y_train_aug_ros, 
                    batch_size = 64,
                    epochs = epochs, 
                    steps_per_epoch = len(X_train_aug_ros)/128,
                    validation_data = (X_test_aug_ros, y_test_aug_ros),
                    callbacks = [es_callback, cp_callback])

model.save("CNNModel5")


model.evaluate(X_test_aug_ros, y_test_aug_ros)


from keras.applications.resnet50 import ResNet50
import cv2


def get_rgb_X(X):
    X = [cv2.cvtColor(np.float32(image), cv2.COLOR_GRAY2RGB) for image in X]
    return X


rgb_X_train_aug_ros = get_rgb_X(X_train_aug_ros)
rgb_X_test_aug_ros = get_rgb_X(X_test_aug_ros)


rgb_X_train_aug_ros = np.array(rgb_X_train_aug_ros)
rgb_X_test_aug_ros = np.array(rgb_X_test_aug_ros)


model = ResNet50(weights = "imagenet", include_top = False,
                 input_shape = (48, 48, 3))
X = model.output
X = Flatten()(X)
X = Dense(len(CLASS_NAMES_WITHOUT_DISGUST), activation = "softmax")(X)
model = Model(inputs = model.input, outputs = X)


model.compile(optimizer = "adam",
             loss = tf.keras.losses.sparse_categorical_crossentropy,
             metrics = ["accuracy"])


history = model.fit(rgb_X_train_aug_ros, y_train_aug_ros, batch_size = 32,
                    epochs = 50,
                    steps_per_epoch = len(rgb_X_train_aug_ros)/128,
                    validation_data = (rgb_X_test_aug_ros, y_test_aug_ros),
                    callbacks = [es_callback, cp_callback])
model.save("CNNModel6")
