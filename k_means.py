# Created to Identify K_Means clusters for Mechanical Behavior Testing
# Created on 07JULY2022 by Benjamin Randoing

from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np

### Import Mechanical Behavior Data
data_raw = pd.read_csv("KMeans_Data.csv", skipfooter=2)
age = data_raw['Age']
data = data_raw[['Proximal Kink Radius', 'Proximal Compliance 50/90', 'Proximal Compliance 80/120', 'Proximal Compliance 110/150', 'Proximal Elastic Modulus', 'Proximal UTS']]
data_dist = data_raw[['Distal Kink Radius', 'Distal Compliance 50/90', 'Distal Compliance 80/120', 'Distal Compliance 110/150', 'Distal Elastic Modulus', 'Distal UTS']]

# Convert Proximal Data into a Numpy Array where each row contains all data characterizing a single vessel
formatted_data = []
for i in range(len(data)-2):
    point = data.iloc[i].to_list()
    formatted_data.append(point)
formatted_data = np.array(formatted_data)


# Convert Distal Data into a Numpy Array where each row contains all data characterizing a single vessel
formatted_data_dist = []
for i in range(len(data_dist)-2):
    point = data_dist.iloc[i].to_list()
    formatted_data_dist.append(point)
formatted_data_dist = np.array(formatted_data_dist)

# Combine Proximal and Distal Data
proximal = formatted_data[0:15]
distal = formatted_data_dist[0:15]
data_local = np.vstack((proximal, distal))

# ## Identify Number of Clusters to use via Inertia (Identified 4 Clusters)
# num_clusters = list(range(1,9))
# inertias = []
#
# for num in num_clusters:
#     model = KMeans(n_clusters = num)
#     model.fit(formatted_data)
#     inertias.append(model.inertia_)
#
# plt.plot(num_clusters, inertias, '-o')
# plt.xlabel('Number of Clusters (k)')
# plt.ylabel('Inertia')
# plt.savefig("Inertia for Age")
# plt.show()

#
# ### Model 4 Clusters
# model = KMeans(n_clusters = 4)
# model.fit(formatted_data)
# labels = model.predict(formatted_data)
# centers = model.cluster_centers_
# print(centers)
#
# kr = formatted_data[:,0]
# comp5090 = formatted_data[:,1]
# comp80120 = formatted_data[:,2]
# comp110150 = formatted_data[:,3]
#
# kr_label = centers[:,0]
# comp5090_label = centers[:,1]
# comp80120_label = centers[:,2]
# comp110150_label = centers[:,3]
#
# plt.figure()
# plt.scatter(kr, comp5090, c=labels, alpha = 0.5)
# plt.scatter(kr_label, comp5090_label, c = 'r', marker='s')
# plt.show()


# ### Perform Principle Component Analysis (PCA) to Plot Clusters in 2D Space
#
# # 1. Standardize the data matrix
# mean = data.mean(axis=0)
# sttd = data.std(axis=0)
# data_matrix_standardized = (data - mean) / sttd
# print(data_matrix_standardized.head())
#
# # 2. Find the principal components
# pca = PCA(n_components = 2)
# components = pca.fit(data_matrix_standardized).components_
# components = pd.DataFrame(components).transpose()
# components.index = data.columns
# print(components)
#
# # 3. Calculate the variance/info ratios
# var_ratio = pca.explained_variance_ratio_
# var_ratio = pd.DataFrame(var_ratio).transpose()
# print(var_ratio)
#
# # 4. Transform the data into 4 new features using the first PCs
# data_pcomp = pca.fit_transform(data_matrix_standardized)
# data_pcomp_array = data_pcomp
# data_pcomp = pd.DataFrame(data_pcomp)
# data_pcomp.columns = ['PC1', 'PC2']
# print(data_pcomp.head())
#
#
#
# ### Model Clusters using new Principle Components
# model_PCA = KMeans(n_clusters = 4)
# model_PCA.fit(data_pcomp_array)
# labels_PCA = model_PCA.predict(data_pcomp_array)
# centers_PCA = model_PCA.cluster_centers_
# print(centers_PCA)
#
# # Step size of the mesh. Decrease to increase the quality of the VQ.
# h = 0.02  # point in the mesh [x_min, x_max]x[y_min, y_max].
#
# # Plot the decision boundary. For that, we will assign a color to each
# x_min, x_max = data_pcomp_array[:, 0].min() - 1, data_pcomp_array[:, 0].max() + 1
# y_min, y_max = data_pcomp_array[:, 1].min() - 1, data_pcomp_array[:, 1].max() + 1
# xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
#
# # Obtain labels for each point in mesh. Use last trained model.
# Z = model_PCA.predict(np.c_[xx.ravel(), yy.ravel()])
#
# # Put the result into a color plot
# Z = Z.reshape(xx.shape)
# plt.figure(1)
# plt.clf()
# plt.imshow(
#     Z,
#     interpolation="nearest",
#     extent=(xx.min(), xx.max(), yy.min(), yy.max()),
#     cmap="Blues",
#     aspect="auto",
#     origin="lower",
#     alpha = 0.3
# )
#
# PC1_data = data_pcomp_array[:,0]
# PC2_data = data_pcomp_array[:,1]
#
# PC1_center = centers_PCA[:,0]
# PC2_center = centers_PCA[:,1]
#
# scatter = plt.scatter(PC1_data, PC2_data, c=age, cmap ='prism', alpha = 1)
# scatter2 = plt.scatter(PC1_center, PC2_center, c = 'black', marker='s', label = "Cluster Centers")
# plt.xlabel('PC1')
# plt.ylabel('PC2')
# plt.legend(handles=scatter.legend_elements()[0],
#            labels=["Old", "New"],
#            title="Age")
# plt.title("HAV Clustered by Principle Components of Mechanical Behaviors")
# plt.savefig("PCA_age.pdf")
# plt.show()



### Perform Principle Component Analysis (PCA) to Plot Clusters in 2D Space

## 1. Standardize the data matrix
mean = data_local.mean(axis=0)
sttd = np.array(data_local.std(axis=0))
data_matrix_standardized = (data_local - mean) / sttd


## 2. Find the principal components
pca = PCA(n_components = 2)
components = pca.fit(data_matrix_standardized).components_
components = pd.DataFrame(components).transpose()
# components.index = data.columns


## 3. Calculate the variance/info ratios
var_ratio = pca.explained_variance_ratio_
var_ratio = pd.DataFrame(var_ratio).transpose()

## 4. Transform the data into 4 new features using the first PCs
data_pcomp = pca.fit_transform(data_matrix_standardized)
data_pcomp_array = data_pcomp
data_pcomp = pd.DataFrame(data_pcomp)
data_pcomp.columns = ['PC1', 'PC2']



### Model Clusters using new Principle Components

## 1. Create KMeans Model and save Predicted Labels and Cluster Center
model_PCA = KMeans(n_clusters = 2)
model_PCA.fit(data_pcomp_array)
labels_PCA = model_PCA.predict(data_pcomp_array)
centers_PCA = model_PCA.cluster_centers_

## 2. Create a Plot of the Data Points with Visualized Clusters (Shaded) and Distal/Proximal Status (Legend)
# Step size of the mesh. Decrease to increase the quality of the VQ.
h = 0.02  # point in the mesh [x_min, x_max]x[y_min, y_max].

# Plot the decision boundary. For that, we will assign a color to each
x_min, x_max = data_pcomp_array[:, 0].min() - 1, data_pcomp_array[:, 0].max() + 1
y_min, y_max = data_pcomp_array[:, 1].min() - 1, data_pcomp_array[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Obtain labels for each point in mesh. Use last trained model.
Z = model_PCA.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure(1)
plt.clf()
plt.imshow(
    Z,
    interpolation="nearest",
    extent=(xx.min(), xx.max(), yy.min(), yy.max()),
    cmap="Blues",
    aspect="auto",
    origin="lower",
    alpha = 0.3
)

## 3. Isolate Data from numpy Array to be Plotted
PC1_data = data_pcomp_array[:,0]
PC2_data = data_pcomp_array[:,1]

PC1_center = centers_PCA[:,0]
PC2_center = centers_PCA[:,1]

## 4. Plot and Save Figure of Shaded Regions with Points

# Make Locations Labels for Proximal (1st 0.5*total entries) and Distal
locations = []
for i in range(30):
    if i < 15:
        locations.append(0)
    else:
        locations.append(1)


scatter = plt.scatter(PC1_data, PC2_data, c=locations, cmap ='prism', alpha = 1)
scatter2 = plt.scatter(PC1_center, PC2_center, c = 'black', marker='s', label = "Cluster Centers")
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend(handles=scatter.legend_elements()[0],
           labels=["Proximal", "Distal"],
           title="Location")
plt.title("HAV Clustered by Principle Components of Mechanical Behaviors")
# plt.savefig("PCA_location.pdf")
plt.show()
