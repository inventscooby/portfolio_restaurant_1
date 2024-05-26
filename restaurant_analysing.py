import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Here I load the CSV file
file_path = r'C:\Users\Green\portfolio_restaurant\restaurant_data.csv'
data = pd.read_csv(file_path)

# Here I analyse the data
print(data.head())
print(data.info())

# And correlation matrix for numeric columns
correlation_matrix = data[['Average Cost for two', 'Price range', 'Aggregate rating', 'Votes']].corr()

# Here I plot a heatmap to see the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix of Numeric Features')
plt.show()

# Here I plot the distribution of key features
plt.figure(figsize=(15, 5))

# Here I build a histogram plot for the distribution of Aggregate rating
plt.subplot(1, 3, 1)
sns.histplot(data['Aggregate rating'], bins=20, kde=True, color='blue')
plt.title('Distribution of Aggregate Rating')
plt.xlabel('Aggregate Rating')
plt.ylabel('Count of Restaurants')

# Here I build a histogram plot for the distribution of Votes
plt.subplot(1, 3, 2)
sns.histplot(data['Votes'], bins=20, kde=True, color='green')
plt.title('Distribution of Votes')
plt.xlabel('Votes')
plt.ylabel('Count of Restaurants')

# Here I build a histogram plot for the distribution of Price range
plt.subplot(1, 3, 3)
sns.histplot(data['Price range'], bins=10, kde=True, color='red')
plt.title('Distribution of Price Range')
plt.xlabel('Price Range')
plt.ylabel('Count of Restaurants')

plt.tight_layout()
plt.show()

# Here I extract the top cuisines
cuisine_counts = data['Cuisines'].str.split(', ').explode().value_counts().head(10)
print(cuisine_counts)

# Here I plot the top 10 cuisines
top_cuisines = ['Italian', 'Chinese', 'North Indian', 'Mexican', 'Cafe', 'Fast Food', 'South Indian', 'Mughlai', 'Bakery', 'Continental']
plt.figure(figsize=(12, 6))
sns.barplot(x=cuisine_counts.index, y=cuisine_counts.values, hue=cuisine_counts.index, palette='viridis', legend=False)
plt.title('Top 10 Cuisines')
plt.xlabel('Cuisine')
plt.ylabel('Count  of Restaurants')
plt.gca().yaxis.grid(True)

# Here I need to remove rows with NaN values in 'Cuisines' column
top_cuisines_data = data[data['Cuisines'].notna() & data['Cuisines'].apply(lambda x: isinstance(x, str) and any(cuisine in x for cuisine in top_cuisines))]

# Here I melt the dataframe to have cuisines in one column
cuisines_ratings = top_cuisines_data[['Cuisines', 'Aggregate rating']].copy()
cuisines_ratings = cuisines_ratings.assign(Cuisines=cuisines_ratings['Cuisines'].str.split(', ')).explode('Cuisines')
cuisines_ratings = cuisines_ratings[cuisines_ratings['Cuisines'].isin(top_cuisines)]

# Here I plot the average rating for top cuisines
plt.figure(figsize=(12, 6))
sns.boxplot(x='Cuisines', y='Aggregate rating', hue='Cuisines', data=cuisines_ratings, palette='viridis', legend=False)
plt.title('Aggregate Rating Distribution for Top Cuisines')
grouped_data = data.groupby('City')['Aggregate rating'].mean().reset_index()
plt.xlabel('Cuisine')
plt.ylabel('Aggregate Rating')
plt.xticks(rotation=45)
plt.gca().yaxis.grid(True)
plt.show()

# Here I group by City to analyze the average rating and total votes per city
city_analysis = data.groupby('City').agg({'Aggregate rating': 'mean', 'Votes': 'sum'}).reset_index()

# Here I sort the data by Aggregate rating and Votes
city_analysis = city_analysis.sort_values(by='Aggregate rating', ascending=False)

# Since the data set contains many cities, I list down top 20 cities to make the plot readable
top_cities_by_rating = city_analysis.sort_values(by='Aggregate rating', ascending=False).head(20)
top_cities_by_votes = city_analysis.sort_values(by='Votes', ascending=False).head(20)

plt.figure(figsize=(14, 10))

# Here I plot average rating by top cities
plt.subplot(1, 2, 1)
sns.barplot(x='Aggregate rating', y='City', hue='City', data=top_cities_by_rating, palette='coolwarm', legend=False)
plt.title('Average Rating by Top 20 Cities')
plt.xlabel('Average Rating')
plt.ylabel('City')

# Here I plot total votes by top cities
plt.subplot(1, 2, 2)
sns.barplot(x='Votes', y='City', hue='City', data=top_cities_by_votes, palette='coolwarm', legend=False)
plt.title('Total Votes by Top 20 Cities')
plt.xlabel('Total Votes')
plt.ylabel('City')

plt.tight_layout()
plt.show()

