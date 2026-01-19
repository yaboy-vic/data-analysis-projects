# Load the libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder   # Scikit–learn requires numeric values
# this library provides labels for category values without using dummy variables
from sklearn.tree import plot_tree               # used to draw a picture of the decision tree
from sklearn.tree import DecisionTreeClassifier     # model we will be using
from sklearn.ensemble import RandomForestClassifier  # model we will be using
from sklearn.metrics import classification_report, confusion_matrix

# read the dataset in
df = pd.read_excel('marketing_problem.xlsx') 
print(df.shape) #gets how many rows and columns
print(df.head(5)) #prints first 5 rows
print(df.tail(5)) #prints last 5 rows
print(df.info()) #prints the data types

print("\nMissing values per column:")
print(df.isnull().sum())  # Checks for any missing (null) values in the dataset

# from the output in the terminal, it seems like there are no missing values
# so I don’t need to fill or drop anything here

df_encoded = df.copy() # made a copy to maintain original dataset
for col in df_encoded.select_dtypes(include='object').columns:
    le = LabelEncoder() # this gives each category a numeric code
    df_encoded[col] = le.fit_transform(df_encoded[col])

print("\nThis is what the data looks like after encoding:")
print(df_encoded.head())

plt.figure(figsize=(10,8))                                   # make the plot bigger so it's easier to read
sns.heatmap(df_encoded.corr(), cmap='coolwarm', annot=True, fmt=".2f")  # shows correlation numbers (2 decimals)
plt.title('Correlation Heatmap (Encoded Features)')              # title of heatmap
plt.tight_layout()
plt.show()                              # display the heatmap

# check the actual correlation values for all features vs y
correlation_with_y = df_encoded.corr()['y'].sort_values(ascending=False)
print(correlation_with_y)

# From the heatmap, duration stands out with the strongest positive link to y.
# That means the longer the call, the more likely the customer says yes to a term deposit.
# pdays is negatively correlated, so customers who haven't been contacted in a long time usually say no.
# campaign and previous also show some relationship, but it's weaker.
# Most of the other features like job, marital, or housing don’t show strong connections to y.

# Split the data into training and testing sets
X = df_encoded.drop('y', axis=1)   # X = features (all columns except y)
y = df_encoded['y']                # y = target (who said yes/no)

# use 80% for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=72)
print(X_train.shape, X_test.shape)  # just checking how many rows ended up in each set

# Step 4: Build the Decision Tree model

model = DecisionTreeClassifier(max_depth=4, random_state=72)  
model.fit(X_train, y_train)                       # train the model with training data

# check accuracy on both training and testing sets
train_acc = model.score(X_train, y_train)   # how well it fits the training data
test_acc = model.score(X_test, y_test)      # how well it performs on unseen data

print(f"Training Accuracy: {train_acc:.3f}")   
print(f"Testing Accuracy: {test_acc:.3f}")     

# visualize the decision tree
plt.figure(figsize=(18,8))   # make it big enough to read clearly
plot_tree(model, feature_names=X.columns, class_names=['No', 'Yes'], filled=True, fontsize=8)
plt.title("Decision Tree (Max Depth = 4)")  # title of the plot
plt.show()  # show the tree

# The top split (root) is duration, which means call length matters most.
# short calls (low duration) almost always lead to "No", while longer calls increase "Yes"
# month also shows up near the top, meaning when the call was made plays a smaller role
# features like contact type and previous calls also help decide the outcome.
# The blue boxes show higher chance of "Yes", orange shows "No".
# So in short, the model thinks longer calls + certain months (like later in the year) are linked with customers saying yes to term deposits.

# model evaluation


# make predictions on the test set
y_pred = model.predict(X_test)   # predicted values
print("\nSample predictions:", y_pred[:10])  # just checking the first few

# confusion matrix shows how many were correctly/incorrectly predicted
cm = confusion_matrix(y_test, y_pred)

# visualize the confusion matrix
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['No', 'Yes']))

# Based on the confusion matrix:
# The model correctly predicted 797 customers who said No (true negatives).
# It also correctly predicted 16 customers who said Yes (true positives).
# There were 15 false positives (predicted Yes but actually No)
# and 77 false negatives (predicted No but actually Yes).

# So, the model is much better at spotting customers who will say No.
# It struggles a bit with predicting the Yes customers — meaning it misses some potential buyers.
# For marketing, that means the bank could end up ignoring some people who might have said Yes.
#
# In business terms: it's safer (less wasted calls), but less aggressive in finding new customers.
# If the bank wants to capture more "Yes" cases, they might prefer to adjust the model
# to focus more on recall, even if it means more false positives.

# Summary: The decision tree gives decent accuracy and helps the bank avoid wasting calls,
# but it needs some tuning if the goal is to target more customers who are likely to say Yes.

print("\n======================== PART 2 ===================================")

# Part 2

# read the dataset in
df2 = pd.read_csv('Liver_disease_data.csv') 
print(df2.shape) #gets how many rows and columns
print(df2.head(5)) #prints first 5 rows
print(df2.tail(5)) #prints last 5 rows
print(df2.info()) #prints the data types

print("\nMissing values per column:")
print(df2.isnull().sum())  # Checks for any missing (null) values in the dataset


plt.figure(figsize=(10,8))  #  make the plot bigger so it's easier to read
sns.heatmap(df2.corr(), cmap='coolwarm', annot=True, fmt=".2f")  # shows correlation numbers for each feature
plt.title('Correlation Heatmap (Liver Disease Data)') # title of heatmap
plt.tight_layout()
plt.show()  # display the heatmap

# checking which features are most related to the Diagnosis column (target)
corr_with_diag = df2.corr()['Diagnosis'].sort_values(ascending=False)
print("\nCorrelation with Diagnosis:")
print(corr_with_diag)

# Split the data into training and testing sets
# I’ll use 80% for training and 20% for testing like the instructions said


X = df2.drop('Diagnosis', axis=1)   # features (everything except Diagnosis)
y = df2['Diagnosis']                # target column

# split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=72)

print(X_train.shape, X_test.shape)  # just checking how many rows went into each set

model = RandomForestClassifier(n_estimators=100, random_state=72)  # 100 trees, same seed for consistency
model.fit(X_train, y_train)  # train the model

# check accuracy on both sets
train_acc = model.score(X_train, y_train) # accuracy on training data
test_acc = model.score(X_test, y_test)  # accuracy on unseen test data

print(f"Training Accuracy: {train_acc:.3f}")
print(f"Testing Accuracy: {test_acc:.3f}")

y_pred = model.predict(X_test)

# Calculate the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

# Display confusion matrix with labels
labels = ['No Disease', 'Has Disease']
plt.figure(figsize=(6,4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - Random Forest Model')
plt.show()

# The confusion matrix shows the model predicted 139 people correctly as having no liver disease and 173 correctly as having liver disease (true positives).
# it made 16 false positives which is itpredicted disease when they didn’t have it and 12 false negatives (missed actual disease cases).
# that means the model does a good job overall especially at catching people who actually have liver disease.
# Since the hospital prefers more false positives over false negatives,
# this model fits their goal well because it only missed 12 real cases.
# In business terms, it’s safer because extra tests (false positives) are better than missing patients who need care.

# Summary:
# The Random Forest model reached around 92% accuracy, showing it predicts liver disease pretty well.
# it performed slightly better at identifying patients who actually have liver disease than those who don’t.
# this is good because the hospital prefers catching possible cases, even if that means a few extra false alarms.
# Overall, the model is reliable and could be a helpful tool for early detection.
# with more data or tuning like adjusting tree depth or sample weights, it could get even better at spotting all positive cases.