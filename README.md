Report on Restaurant Data Analysis and Insights

Objective:

The objective of this analysis is to gain insights into customer satisfaction, segment customers based on their behavior, and optimize operational strategies using the restaurant dataset.

Data Preprocessing:

Handling Missing Values: Filled missing values in the 'Cuisines' column with 'Unknown'.
Label Encoding: Encoded categorical variables ('City', 'Cuisines', 'Has Table booking', 'Has Online delivery') using LabelEncoder.
Feature Scaling: Standardized numeric features ('Average Cost for two', 'Votes') using StandardScaler.

Predictive Modeling:

Model Used: Random Forest Regressor
Features: 'City', 'Cuisines', 'Price range', 'Votes', 'Average Cost for two', 'Has Table booking', 'Has Online delivery'
Target: 'Aggregate rating'

Evaluation Metrics:

Mean Squared Error: 0.10271160080481714
R-squared: 0.9548740410753127

Key Insights from Feature Importance:

Votes: The most significant predictor of customer satisfaction.
Average Cost for two: The second most important feature.
City and Price Range: Also important, but to a lesser extent.
Has Online delivery, Cuisines, Has Table booking: Less influential but still relevant.
Customer Segmentation
Method: K-means clustering (5 clusters)
Features: 'Votes', 'Average Cost for two', 'Aggregate rating'
Dimensionality Reduction: PCA for visualization

Cluster Descriptions:

Cluster 0: Lower votes, lower average cost, and lower aggregate rating.
Cluster 1: Moderate votes, moderate average cost, and moderate aggregate rating.
Cluster 2: Higher votes, higher average cost, and higher aggregate rating.
Cluster 3: Very high votes, very high average cost, and very high aggregate rating.
Cluster 4: Extremely high votes, extremely high average cost, and highest aggregate rating.

Insights:

Table Booking:

Restaurants with table booking options generally have higher aggregate ratings.
This suggests that offering table booking can improve customer satisfaction.

Online Delivery:

Restaurants offering online delivery tend to have better aggregate ratings.
This indicates that online delivery services are valued by customers and can enhance satisfaction.
Pricing Strategy Analysis

Findings:

There is a positive correlation between the average cost for two and aggregate ratings.
Higher-priced restaurants generally receive better ratings, suggesting that customers perceive higher value or quality at these establishments.

Summary of Findings:
Key Factors for Customer Satisfaction:
Votes and average cost for two are significant predictors.
City and price range also play crucial roles.

Customer Segmentation:
Five distinct customer segments were identified, which can be targeted with specific marketing strategies.

Operational Strategies:
Table Booking: Offering this option is associated with higher satisfaction.
Online Delivery: Providing online delivery services correlates with better ratings.

Pricing Strategy:
Higher-priced restaurants often receive better ratings, indicating a possible perceived value or quality.
