
"""
Feature extraction by Autoencoder

@author: Hbzh
"""

import numpy as np
import pandas as pd
from PIL import Image
#from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping



# clinical data
data = pd.read_csv('spaTIL_pca_TCGA.csv')
data['LN_metastasis'] = data['LN_metastasis'].replace({'Yes': 1, 'No': 0})
data['T'] = data['T'].replace({'T4': 3, 'T3': 2, 'T2': 1, 'T1': 0})
data_train, data_val = train_test_split(data, test_size=0.3, random_state=0)
print(data.dtypes)
data_train=data_train[["sample", "Age", "T", "TILscore","LN_metastasis"]]
data_val=data_val[["sample", "Age", "T", "TILscore","LN_metastasis"]]

# image data
def load_images(data, image_folder):
    images = []
    labels = []
    for _, row in data.iterrows():
        try:
            image_path = f"{image_folder}/{row['sample']}.png"
            image = Image.open(image_path).convert('RGB')
            image = image.resize((1024, 1024))  
            image_array = np.array(image) / 255.0  
            images.append(image_array)
            labels.append(row['LN_metastasis'])
        except FileNotFoundError:
            print(f"File not found: {image_path}. Skipping...")
    return np.array(images), np.array(labels)


# input data
image_folder = 'TCGA_cohort'
images, labels = load_images(data, image_folder)

# split dataset
X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.3, random_state=0)


# Define the encoder
def build_encoder(input_img):
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(4, (3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)
    return encoded

def build_decoder(encoded):
    x = Conv2D(4, (3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)
    return decoded

# Input placeholder
input_img = Input(shape=(1024, 1024, 3))  # Adjust dimensions as necessary

# Build the autoencoder using Functional API
encoded = build_encoder(input_img)
decoded = build_decoder(encoded)

# Create the autoencoder model
autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

# Model summary
autoencoder.summary()


# Train the autoencoder
model_checkpoint = ModelCheckpoint('Autoencoder_best_model.h5', save_best_only=True, monitor='val_loss')
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
# callback
autoencoder.fit(X_train, X_train,
                epochs=50,
                batch_size=8,
                shuffle=True,
                validation_data=(X_val, X_val),
                callbacks=[early_stopping, model_checkpoint])





