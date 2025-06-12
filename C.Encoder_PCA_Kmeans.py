
"""

encoder--pca--kmeans

@author: Hbzh
"""

import numpy as np
import pandas as pd
from PIL import Image
#from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from tensorflow import keras
from sklearn.decomposition import PCA



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


### train val set
# clinical data
data = pd.read_csv('spaTIL_pca_TCGA.csv')
data['LN_metastasis'] = data['LN_metastasis'].replace({'Yes': 1, 'No': 0})
data['T'] = data['T'].replace({'T4': 3, 'T3': 2, 'T2': 1, 'T1': 0})
data=data[["sample", "Age", "T", "TILscore","LN_metastasis"]]


# input data
image_folder = 'TCGA_cohort'
#images, labels = load_images(data, image_folder)
X_data, y_data = load_images(data, image_folder)



### test set
test = pd.read_csv('spaTIL_pca_XY.csv')
test['LN_metastasis'] = test['LN_metastasis'].replace({'Yes': 1, 'No': 0})
test['T'] = test['T'].replace({'T4': 3, 'T3': 2, 'T2': 1, 'T1': 0})
data_test =test[["sample", "Age", "T", "TILscore","LN_metastasis"]]
# input data
image_folder2 = 'XY_cohort'
X_test, y_test = load_images(test, image_folder2)


### Encoder Model prediction

from tensorflow.keras.models import load_model
best_model = load_model('Autoencoder_best_model.h5')
encoder_output_layer = best_model.layers[8].output  # 9th layer
encoder_model = keras.Model(inputs=best_model.input, outputs=encoder_output_layer)

# feature extraction
encoded_data_imgs = encoder_model.predict(X_data)
encoded_test_imgs = encoder_model.predict(X_test)

encoded_data_df = pd.DataFrame(encoded_data_imgs.reshape(encoded_data_imgs.shape[0], -1))  # Recast into 2D data
#encoded_data_df['label'] = y_data  
encoded_test_df = pd.DataFrame(encoded_test_imgs.reshape(encoded_test_imgs.shape[0], -1))  # Recast into 2D data
#encoded_test_df['label'] = y_test 


"""
####### PCA analysis #######
"""
### train and val set

# Initialize PCA, setting it to retain 95% of the variance
pca = PCA(n_components=0.95)
# Fit and transform the training data
pca_data_features = pca.fit_transform(encoded_data_imgs.reshape(encoded_data_imgs.shape[0], -1)) # Recast into 2D data
# Use the PCA parameters obtained from the training set to transform the validation data
pca_test_features = pca.transform(encoded_test_imgs.reshape(encoded_test_imgs.shape[0], -1)) # Recast into 2D data
print(f"Number of PCA components (95% variance retained): {pca.n_components_}")


"""
####### Kmeans clustering #######
"""

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
pca_data_scaled = pca_data_features
pca_test_scaled = pca_test_features


### 1.Silhouette Score Plot

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np

# Assuming pca_data_scaled is your data after PCA and scaling
cluster_range = range(2, 12)
silhouette_scores = []

for k in cluster_range:
    kmeans = KMeans(n_clusters=k, random_state=0)
    cluster_labels = kmeans.fit_predict(pca_data_scaled)
    silhouette_avg = silhouette_score(pca_data_scaled, cluster_labels)
    silhouette_scores.append(silhouette_avg)

optimal_k = cluster_range[np.argmax(silhouette_scores)]

plt.figure(figsize=(6, 6))
plt.plot(cluster_range, silhouette_scores, marker='o')
plt.title('Silhouette Score for Various Cluster Counts')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.axvline(x=optimal_k, color='r', linestyle='--', label=f'Optimal k: {optimal_k}')
plt.legend()
plt.savefig('Silhouette_Score.pdf')  
plt.show()



### 2. Davies-Bouldin Index Plot
from sklearn.metrics import davies_bouldin_score

dbi_scores = []

for k in cluster_range:
    kmeans = KMeans(n_clusters=k, random_state=0)
    cluster_labels = kmeans.fit_predict(pca_data_scaled)
    dbi = davies_bouldin_score(pca_data_scaled, cluster_labels)
    dbi_scores.append(dbi)

optimal_k = cluster_range[np.argmin(dbi_scores)]

plt.figure(figsize=(6, 6))
plt.plot(cluster_range, dbi_scores, marker='o')
plt.title('Davies-Bouldin Index for Various Cluster Counts')
plt.xlabel('Number of Clusters')
plt.ylabel('Davies-Bouldin Index')
plt.axvline(x=optimal_k, color='r', linestyle='--', label=f'Optimal k: {optimal_k}')
plt.legend()
plt.savefig('Davies_Bouldin_Index.pdf')  
plt.show()


### 3. Calinski-Harabasz Index Plot
from sklearn.metrics import calinski_harabasz_score

ch_scores = []

for k in cluster_range:
    kmeans = KMeans(n_clusters=k, random_state=0)
    cluster_labels = kmeans.fit_predict(pca_data_scaled)
    ch_score = calinski_harabasz_score(pca_data_scaled, cluster_labels)
    ch_scores.append(ch_score)

optimal_k = cluster_range[np.argmax(ch_scores)]

plt.figure(figsize=(6, 6))
plt.plot(cluster_range, ch_scores, marker='o')
plt.title('Calinski-Harabasz Index for Various Cluster Counts')
plt.xlabel('Number of Clusters')
plt.ylabel('Calinski-Harabasz Index')
plt.axvline(x=optimal_k, color='r', linestyle='--', label=f'Optimal k: {optimal_k}')
plt.legend()
plt.savefig('Calinski_Harabasz_Index.pdf')  # Save as PDF
plt.show()



# Determine the optimal K=2 based on the above three methods

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=2, random_state=0)

kmeans.fit(pca_data_scaled)
data_labels = kmeans.labels_
print("cluster center:\n", kmeans.cluster_centers_)

test_labels = kmeans.predict(pca_test_scaled)

# Optional: View the distribution of clustering labels in the test dataset
print("Cluster label distribution in the test dataset:\n", np.bincount(test_labels))
print("Cluster label distribution in the training dataset:\n", np.bincount(data_labels))

data['label'] = data_labels
data_test['label'] = test_labels

# Finally save the datasets containing sTILCs information
data.to_csv('data_set_final.csv', index=False)
data_test.to_csv('test_set_final.csv', index=False)
