import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

# First I load the dataset
file_path = r'C:\Users\Green\portfolio_restaurant\restaurant_data.csv'
data = pd.read_csv(file_path)

# And here I perform Data Preprocessing
data['Cuisines'] = data['Cuisines'].fillna('Unknown')
label_encoders = {}
for column in ['City', 'Cuisines', 'Has Table booking', 'Has Online delivery']:
    label_encoders[column] = LabelEncoder()
    data[column] = label_encoders[column].fit_transform(data[column])

# Here I select the features and target variable for prediction
features = ['City', 'Cuisines', 'Price range', 'Votes', 'Average Cost for two', 'Has Table booking', 'Has Online delivery']
target = 'Aggregate rating'
X = data[features]
y = data[target]

# Here I split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Here I build the Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

# Here I evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

# Here I visualize the feature importances
importances = model.feature_importances_
feature_names = X.columns
plt.figure(figsize=(10, 6))
sns.barplot(x=importances, y=feature_names)
plt.title('Feature Importances for Customer Satisfaction')
plt.xlabel('Importance')
plt.ylabel('Features')
plt.xticks([0, 0.5, 1], ['Not important', 'So so', 'Important'])
plt.grid(axis='both', linestyle='--')
plt.show()

# Here I perform Customer Segmentation using K-means Clustering
customer_features = ['Votes', 'Average Cost for two', 'Aggregate rating']
X_customers = data[customer_features]
X_customers_scaled = scaler.fit_transform(X_customers)

pca = PCA(n_components=2)
principal_components = pca.fit_transform(X_customers_scaled)

kmeans = KMeans(n_clusters=5, random_state=42)
clusters = kmeans.fit_predict(principal_components)
data['Cluster'] = clusters

# Here I visualize the clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(x=principal_components[:, 0], y=principal_components[:, 1], hue=clusters, palette='viridis')
plt.title('Customer Segmentation using K-means Clustering')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(title='Cluster')
plt.grid(axis='both', linestyle='--')
plt.show()

# Here I analyze each cluster and print the statistics
for i in range(5):
    print(f"Cluster {i}:")
    cluster_data = data[data['Cluster'] == i]
    print(cluster_data[customer_features].describe())
    print("\n")

# Here I analyze the impact of table booking and online delivery on aggregate ratings
plt.figure(figsize=(10, 6))
sns.boxplot(x='Has Table booking', y='Aggregate rating', data=data)
plt.title('Impact of Table Booking on Aggregate Rating')
plt.xlabel('Has Table booking (0: Without Table Booking, 1: With Table Booking)')  # Add note to the x-axis
plt.grid(axis='y', linestyle='--')
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(x='Has Online delivery', y='Aggregate rating', data=data)
plt.title('Impact of Online Delivery on Aggregate Rating')
plt.xlabel('Has Online delivery (0: Without Online Delivery, 1: With Online Delivery)')
plt.grid(axis='y', linestyle='--')
plt.show()

# Here I analyze pricing strategy
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Average Cost for two', y='Aggregate rating', data=data)
plt.title('Pricing Strategy Analysis')
plt.xlabel('Average Cost for two')
plt.ylabel('Aggregate Rating')
plt.grid(axis='both', linestyle='--')
plt.show()
