import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

data = {
    'Age' : [25, 30, 35, 40, 45,50],
    'Income' : [50000, 60000, 80000, 90000, 120000,150000],
    'Spending' : [2000, 3000, 4000, 5000, 7000,9000],
    'Savings' : [10000, 15000, 20000, 25000, 30000,40000]
}

df = pd.DataFrame(data)

scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

print(f'"Scaled data"{scaled_data}')

pca = PCA(n_components = 2)
pca_data = pca.fit_transform(scaled_data)

pca_df = pd.DataFrame(pca_data, columns = ['PC1', 'PC2'])
explained_variance = pca.explained_variance_ratio_
print(np.round(explained_variance*100,2))

# print(pca_df)

plt.figure(figsize=(8,6))

plt.scatter(pca_df['PC1'], pca_df['PC2'], color = 'Black', s=80)
plt.title('PCa Projection (2D View)')
plt.xlabel('PCA1 Main Patter')
plt.ylabel('PCA2 Main PAttern')
plt.grid()
plt.show()

print(pca_df)

