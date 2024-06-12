# The Algorithm****
# Fuzzy c-means (FCM) is a method of clustering which allows one
# piece of data to belong to two or more clusters. This method (developed by
# Dunn in 1973 and improved by Bezdek in 1981) is frequently used in pattern
# recognition.

# # Loading modules and training data
import pandas as pd
import numpy as np
import random
import operator
import math
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

df_full = pd.read_csv("Input.csv")
df_full.head()
df_full = df_full.drop(['Id'], axis=1)
df_full.shape
columns = list(df_full.columns)
features = columns[:len(columns)-1]
class_labels = list(df_full[columns[-1]])
df = df_full[features]

# # Defining parameters
# Number of Clusters
k = 3
# Maximum number of iterations
MAX_ITER = 100
# Number of data points
n = len(df)
# Fuzzy parameter
m = 1.7 #Select a value greater than 1 else it will be knn
# # Scatter Plots
plt.figure(figsize=(10,10))
plt.scatter(list(df.iloc[:,0]), list(df.iloc[:,1]), marker='o')
plt.axis('equal')
plt.xlabel('intelligence', fontsize=16)
plt.ylabel('talent', fontsize=16)
plt.title('Chance Plot', fontsize=22)
plt.grid()
plt.show()

# # Calculating accuracy
def accuracy(cluster_labels, class_labels):
    correct_pred = 0
    #print(cluster_labels)
    low = max(set(labels[0:50]), key=labels[0:50].count)
    medium = max(set(labels[50:100]), key=labels[50:100].count)
    high = max(set(labels[100:]), key=labels[100:].count)
    for i in range(len(df)):
        if cluster_labels[i] == low and class_labels[i] == 'poor;;':
            correct_pred = correct_pred + 1
        if cluster_labels[i] == medium and class_labels[i] == 'average;;' and medium!=low:
            correct_pred = correct_pred + 1
        if cluster_labels[i] == high and class_labels[i] == 'good;;' and high!=low and high!=medium:
            correct_pred = correct_pred + 1
    accuracy = (correct_pred/len(df))*100
    return accuracy

# # Initialize membership matrix
def initializeMembershipMatrix():
    membership_mat = []
    for i in range(n):
        random_num_list = [random.random() for i in range(k)]
        summation = sum(random_num_list)
        temp_list = [x/summation for x in random_num_list]
        flag = temp_list.index(max(temp_list))
        for j in range(0,len(temp_list)):
            if(j == flag):
                temp_list[j] = 1
            else:
                temp_list[j] = 0
        membership_mat.append(temp_list)
    return membership_mat

membership_mat = initializeMembershipMatrix()

# # Calculating Cluster Center
def calculateClusterCenter(membership_mat):
    cluster_mem_val = list(zip(*membership_mat))
    cluster_centers = []
    for j in range(k):
        x = list(cluster_mem_val[j])
        xraised = [p ** m for p in x]
        denominator = sum(xraised)
        temp_num = []
        for i in range(n):
            data_point = list(df.iloc[i])
            prod = [xraised[i] * val for val in data_point]
            temp_num.append(prod)
        numerator = map(sum, list(zip(*temp_num)))
        center = [z / denominator for z in numerator]
        cluster_centers.append(center)
    return cluster_centers

calculateClusterCenter(membership_mat)

# # Updating Membership Value
def updateMembershipValue(membership_mat, cluster_centers):
    p = float(2/(m-1))
    for i in range(n):
        x = list(df.iloc[i])
        distances = [np.linalg.norm(np.array(list(map(operator.sub, x, cluster_centers[j])))) for j in range(k)]
        for j in range(k):
            den = sum([math.pow(float(distances[j]/distances[c]), p) for c in range(k)])
            membership_mat[i][j] = float(1/den)
    return membership_mat

# # Getting the clusters
def getClusters(membership_mat):
    cluster_labels = list()
    for i in range(n):
        max_val, idx = max((val, idx) for (idx, val) in enumerate(membership_mat[i]))
        cluster_labels.append(idx)
    return cluster_labels

# **Below are three different initializations. When the
# initialization is at the origin all points converge into one cluster and
# for the other 2 cases we get the clusters as we have initialized before(3
# in this code**)**

# # Fuzzy C-Means with cluster centres at the origin
def fuzzyCMeansClustering(): #First Iteration with centers at 0
    # Membership Matrix
    membership_mat = initializeMembershipMatrix()
    curr = 0
    acc=[]
    cent_temp = [[0, 0, 0, 0],[0, 0, 0, 0],[0, 0, 0, 0]]
    while curr < MAX_ITER:
        if (curr == 0):
            cluster_centers = cent_temp
            print("Cluster Centers:")
            print(np.array(cluster_centers))
        else:
            cluster_centers = calculateClusterCenter(membership_mat)
        # cluster_centers = calculateClusterCenter(membership_mat)
        membership_mat = updateMembershipValue(membership_mat, cluster_centers)
        cluster_labels = getClusters(membership_mat)
        acc.append(cluster_labels)
        curr += 1
    print("---------------------------")
    print("Membership Matrix:")
    print(np.array(membership_mat))
    return cluster_labels, cluster_centers, acc

# # Fuzzy C-Means with with cluster centers at random locations
# within a multi-variate Gaussian distribution with zero-mean and unit-
# variance.

def fuzzyCMeansClustering(): #Second Iteration Multivariate Gaussian
    # Membership Matrix
    membership_mat = initializeMembershipMatrix()
    curr = 0
    acc=[]
    mean = [0, 0]
    cov = [[1, 0], [0, 1]]
    lis1,cent_temp=[],[]
    for i in range(0,k):
        Z = list(np.random.multivariate_normal(mean, cov))
        Z1 = list(np.random.multivariate_normal(mean, cov))
        lis1 = Z+Z1
        cent_temp.append(lis1)
    while curr < MAX_ITER:
        if(curr == 0):
            cluster_centers = cent_temp
            print("Cluster Centers:")
            print(np.array(cluster_centers))
        else:
            cluster_centers = calculateClusterCenter(membership_mat)
        #cluster_centers = calculateClusterCenter(membership_mat)
        membership_mat = updateMembershipValue(membership_mat, cluster_centers)
        cluster_labels = getClusters(membership_mat)
        acc.append(cluster_labels)
        curr += 1
    print("---------------------------")
    print("Membership Matrix:")
    print(np.array(membership_mat))
    return cluster_labels, cluster_centers, acc

# # Fuzzy C-Means with cluster centers at random vectors chosen from
# the data.
def fuzzyCMeansClustering(): #Third iteration Random vectors from data
    # Membership Matrix
    membership_mat = initializeMembershipMatrix()
    curr = 0
    acc=[]
    while curr < MAX_ITER:
        cluster_centers = calculateClusterCenter(membership_mat)
        membership_mat = updateMembershipValue(membership_mat, cluster_centers)
        cluster_labels = getClusters(membership_mat)
        acc.append(cluster_labels)
        if(curr == 0):
            print("Cluster Centers:")
            print(np.array(cluster_centers))
        curr += 1
    print("---------------------------")
    print("Partition matrix:")
    print(np.array(membership_mat))
    #return cluster_labels, cluster_centers
    return cluster_labels, cluster_centers, acc

# # Calculating the Accuracy
labels, centers, acc = fuzzyCMeansClustering()
a = accuracy(labels, class_labels)
acc_lis = []
for i in range(0,len(acc)):
    val = accuracy(acc[i], class_labels)
    acc_lis.append(val)
acc_lis = np.array(acc_lis) #calculating accuracy and std deviation 100 times
print("mean=",np.mean(acc_lis))
print("Std dev=",np.std(acc_lis))
print("Accuracy = " + str(round(a, 2)))
print("Cluster center vectors:") #final cluster centers
print(np.array(centers))

# # Plotting the data
sepal_df = df_full.iloc[:,0:2]
sepal_df = np.array(sepal_df)

m1 = random.choice(sepal_df)
m2 = random.choice(sepal_df)
m3 = random.choice(sepal_df)
cov1 = np.cov(np.transpose(sepal_df))
cov2 = np.cov(np.transpose(sepal_df))
cov3 = np.cov(np.transpose(sepal_df))
x1 = np.linspace(70,150,150)
x2 = np.linspace(0,11,150)
X, Y = np.meshgrid(x1,x2)
Z1 = multivariate_normal(m1, cov1)
Z2 = multivariate_normal(m2, cov2)
Z3 = multivariate_normal(m3, cov3)
pos = np.empty(X.shape + (2,))
pos[:, :, 0] = X; pos[:, :, 1] = Y
plt.figure(figsize=(10,10))
plt.scatter(sepal_df[:,0], sepal_df[:,1], marker='o')
plt.contour(X, Y, Z1.pdf(pos), colors="r" ,alpha = 0.5)
plt.contour(X, Y, Z2.pdf(pos), colors="b" ,alpha = 0.5)
plt.contour(X, Y, Z3.pdf(pos), colors="g" ,alpha = 0.5)
plt.axis('equal')
plt.xlabel('intelligence', fontsize=16)
plt.ylabel('talent', fontsize=16)
plt.title('Initial Random Clusters (Chance Evaluation)', fontsize=22)
plt.grid()
plt.show()

#finding mode
low = max(set(labels[0:50]), key=labels[0:50].count)
medium = max(set(labels[50:100]), key=labels[50:100].count)
high = max(set(labels[100:]), key=labels[100:].count)

#features
s_mean_clus1 = np.array([centers[low][0],centers[low][1]])
s_mean_clus2 = np.array([centers[medium][0],centers[medium][1]])
s_mean_clus3 = np.array([centers[high][0],centers[high][1]])

values = np.array(labels) #label

#search all 3 chance level
searchval_low = low
searchval_medium = medium
searchval_high = high

#index of all 3 chance level
ii_low = np.where(values == searchval_low)[0]
ii_medium = np.where(values == searchval_medium)[0]
ii_high = np.where(values == searchval_high)[0]
ind_low = list(ii_low)
ind_medium = list(ii_medium)
ind_high = list(ii_high)

sepal_df = df_full.iloc[:,0:2]

low_df = sepal_df[sepal_df.index.isin(ind_low)]
medium_df = sepal_df[sepal_df.index.isin(ind_medium)]
high_df = sepal_df[sepal_df.index.isin(ind_high)]

cov_low = np.cov(np.transpose(np.array(low_df)))
cov_medium = np.cov(np.transpose(np.array(medium_df)))
cov_high = np.cov(np.transpose(np.array(high_df)))

sepal_df = np.array(sepal_df)

x1 = np.linspace(70,160,150)
x2 = np.linspace(0,11,150)
X, Y = np.meshgrid(x1,x2)

Z1 = multivariate_normal(s_mean_clus1, cov_low)
Z2 = multivariate_normal(s_mean_clus2, cov_medium)
Z3 = multivariate_normal(s_mean_clus3, cov_high)

pos = np.empty(X.shape + (2,))
pos[:, :, 0] = X; pos[:, :, 1] = Y

plt.figure(figsize=(10,10))
plt.scatter(sepal_df[:,0], sepal_df[:,1], marker='o')
plt.contour(X, Y, Z1.pdf(pos), colors="r" ,alpha = 0.5)
plt.contour(X, Y, Z2.pdf(pos), colors="b" ,alpha = 0.5)
plt.contour(X, Y, Z3.pdf(pos), colors="g" ,alpha = 0.5)
plt.axis('equal')
plt.xlabel('intelligence', fontsize=16)
plt.ylabel('talent', fontsize=16)
plt.title('Final Clusters (Chance Evaluation)', fontsize=22)
plt.grid()
plt.show()