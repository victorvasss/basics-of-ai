from sklearn.datasets import load_digits

digits = load_digits()
digits_data = digits.data
digits_labels = digits.target

import numpy as np

labels = np.reshape(digits_labels,(1797,1))
final_digits_data = np.concatenate([digits_data,labels],axis=1)

import pandas as pd

digits_dataset = pd.DataFrame(final_digits_data)
features = digits.feature_names
features_labels = np.append(features,'label')
digits_dataset.columns = features_labels

print(digits_dataset.head())

digits_dataset['label'].replace(0, 'Zero',inplace=True)
digits_dataset['label'].replace(1, 'One',inplace=True)
digits_dataset['label'].replace(2, 'Two',inplace=True)
digits_dataset['label'].replace(3, 'Three',inplace=True)
digits_dataset['label'].replace(4, 'Four',inplace=True)
digits_dataset['label'].replace(5, 'Five',inplace=True)
digits_dataset['label'].replace(6, 'Six',inplace=True)
digits_dataset['label'].replace(7, 'Seven',inplace=True)
digits_dataset['label'].replace(8, 'Eight',inplace=True)
digits_dataset['label'].replace(9, 'Nine',inplace=True)

print(digits_dataset.tail())

from sklearn.preprocessing import StandardScaler
x = digits_dataset.loc[:, features].values
x = StandardScaler().fit_transform(x) # normalizing the features

print(x.shape)

feat_cols = ['feature'+str(i) for i in range(x.shape[1])]
normalised_digits = pd.DataFrame(x,columns=feat_cols)

print(normalised_digits.tail())

from sklearn.decomposition import PCA
pca_digits = PCA(n_components=2)
principalComponents_digits = pca_digits.fit_transform(x)
principal_digits_Df = pd.DataFrame(data = principalComponents_digits , columns = ['principal component 1', 'principal component 2'])

print(principal_digits_Df.tail())

print('Explained variation per principal component: {}'.format(pca_digits.explained_variance_ratio_))

import matplotlib.pyplot as plt
plt.figure()
plt.figure(figsize=(10,10))
plt.xticks(fontsize=12)
plt.yticks(fontsize=14)
plt.xlabel('Principal Component - 1',fontsize=20)
plt.ylabel('Principal Component - 2',fontsize=20)
plt.title("Principal Component Analysis of Handwritten Digits Dataset",fontsize=20)
targets = ['Zero', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine']
colors = ['red', 'green', 'black', 'yellow', 'white', 'brown', 'blue', 'purple', 'grey', 'pink']
for target, color in zip(targets,colors):
    indicesToKeep = digits_dataset['label'] == target
    plt.scatter(principal_digits_Df.loc[indicesToKeep, 'principal component 1'], principal_digits_Df.loc[indicesToKeep, 'principal component 2'], c = color, s = 50)

plt.legend(targets,prop={'size': 15})
plt.show()