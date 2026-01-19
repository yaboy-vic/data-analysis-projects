"""
Apartment Rent Analysis
Coursework Exercise

Explores apartment rental pricing patterns using Python,
including data cleaning and exploratory data analysis.
"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# read the dataset in, use sep=';' because this file uses semicolons instead of commas)
df = pd.read_csv('apartments_for_rent.csv', sep=';', quotechar='"', encoding='utf-8', on_bad_lines='skip') 
print(df.shape) #gets how many rows and columns
print(df.head(5)) #prints first 5 rows
print(df.tail(5)) #prints last 5 rows
print(df.info()) #prints the data types

print(df.isnull().sum())  # Checks for any missing (null) values in the dataset
# based on the output, amenities 35%, pets_allowed 41%, and address 33%
# have high missing counts and might be dropped later. smaller ones will be filled.

# Clean the dataset

df = df.drop(['id', 'category', 'title', 'body', 'amenities', 'pets_allowed', 'address', 'source'], axis=1)

# filling smaller missing values with median
df['bathrooms'] = df['bathrooms'].fillna(df['bathrooms'].median())
df['bedrooms'] = df['bedrooms'].fillna(df['bedrooms'].median())
df['latitude'] = df['latitude'].fillna(df['latitude'].median())
df['longitude'] = df['longitude'].fillna(df['longitude'].median())
df['state'] = df['state'].fillna('Unknown')
df['cityname'] = df['cityname'].fillna('Unknown')

print("\nMissing values after cleaning:")
print(df.isnull().sum())

# For the part, i changed words (like city/state names) into numbers
# and then plot a heatmap to see which variables are related to price

df['cityname'] = df['cityname'].astype('category').cat.codes     # each city gets its own number
df['state'] = df['state'].astype('category').cat.codes           # each state gets a number too

plt.figure(figsize=(10, 8))                                     # make the plot bigger so it's easier to read
sns.heatmap(df.corr(numeric_only=True), cmap='coolwarm', annot=True)  
plt.title('Correlation Heatmap of Numerical Features')           # title of heatmap
plt.tight_layout()                                               
plt.show()                                                       # display the heatmap

# Insights
# A high positive value (close to +1) means when one increases, the other does too.
# A negative value (close to -1) means when one goes up, the other goes down.
# from the heatmap, it’s clear that bathrooms, bedrooms, and square_feet have the strongest positive relationships with price. 
# This makes sense because bigger apartments with more rooms usually cost more to rent. 
# Among these, bathrooms showed the highest correlation which is around 0.41, followed by square_feet (0.39) and bedrooms (0.31).
# These will probably be the most useful independent variables in the regression model.

# Some variables like cityname, latitude, and longittude show very low correlation  with price, which means they might not help much in predicting rent.
# Also, bathrooms and bedrooms are highly correlated with each other (about 0.71).


from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import statsmodels.api as sm

train_df, test_df = train_test_split(df, test_size= 0.2, random_state=72) # keeps 80% for training, 20% for testing

train_X = train_df[['bathrooms', 'bedrooms', 'square_feet']] # these are the independent variables (predictors)
train_y = train_df['price'] # dependent variable (target)

test_X = test_df[['bathrooms', 'bedrooms', 'square_feet']] # for test
test_y = test_df['price']

# Use add constant to retrieve the Y intercept
train_X = sm.add_constant(train_X)
test_X = sm.add_constant(test_X)

model = sm.OLS(train_y, train_X).fit() # OSL model
print(model.summary()) # shows coefficients, p-values, R-squared, etc.

predictions = model.predict(test_X) # predictions to test data

# calculate the accuracy metrics
mse = mean_squared_error(test_y, predictions)   # mean squared error
rmse = np.sqrt(mse)  # square root mse
mae = mean_absolute_error(test_y, predictions)  # mean absolute error
r2 = r2_score(test_y, predictions)      # R-squared for test data

# adjusted R-squared (for test set)
n = test_X.shape[0]          # total number of test samples (rows)
p = test_X.shape[1] - 1      # number of predictors (columns, minus the constant)
adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

# Adjusted r2 tells how well the model fits after accounting for the number of predictors.


# Result
print("\nEvaluation Metrics on Test Data:")
print(f"Mean Squared Error (MSE): {mse:.2f}")   
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}") 
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Adjusted R-squared: {adj_r2:.4f}")     

# graph showing the predicted vs actual prices
plt.figure(figsize=(7,5))
plt.scatter(test_y, predictions, alpha=0.6, color='teal')      
plt.xlabel("Actual Prices") #
plt.ylabel("Predicted Prices")
plt.title("Predicted vs Actual Apartment Prices")
plt.tight_layout()
plt.show()

# Insights
# The plot helps us see how close the predictions are to the actual rent prices.
# If the points line up along the diagonal, our model did well.
# rmse and mae show the average prediction error in price units.
# Adjusted R-squared tells how much of the variation in rent our model explains.
# the higher the Adjusted R-squared, the better our model fits the data.

# Summary/Analysis
# This model predicts rent using bathrooms, bedrooms, and square footage.
# The adjusted R-squared of around 0.19 means these features explain about 19% of rent price differences, so there’s still room to improve. 
# Bathrooms and square footage had the biggest positive impact on price, while bedrooms were slightly negative, probably because of overlap with bathrooms. 
# Overall, it makes sense that bigger apartments with more bathrooms cost more. 
# The model could get better if we added more details like location or amenities.

