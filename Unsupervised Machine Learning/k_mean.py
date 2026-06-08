import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

data = {
    "Customer" : ["Riya", "Sonia", "Ankit", "Rahul", "Priya", "Amit", "Neha", "Vikram", "Sneha", "Karan"],
    "Age" : [25, 30, 35, 40, 45, 50, 55, 60, 65, 70],
    "Spending" : [200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100]
}

df = pd.DataFrame(data)

X = df[["Age", "Spending"]]

model = KMeans(n_clusters=3, random_state = 42, n_init = 10)

df["Group"] = model.fit_predict(X)

plt.figure(figsize=(10, 6))

for group in df["Group"].unique():
    group_data = df[df["Group"] == group]
    plt.scatter(group_data["Age"], group_data["Spending"], label=f"Group {group}")

plt.title("Customer Segmentation using K-Means")
plt.xlabel("Age")
plt.ylabel("Spending")
plt.legend()
plt.grid()
plt.show()

print(df)