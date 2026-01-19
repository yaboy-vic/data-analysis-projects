import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("global_climate_events_economic_impact_2020_2025.csv") #load data
target = "economic_impact_million_usd"
numericFeatures = [
    "severity",
    "duration_days",
    "affected_population",
    "deaths",
    "injuries",
    "infrastructure_damage_score",
    "response_time_hours",
    "international_aid_million_usd",
    "latitude",
    "longitude",
    "total_casualties",
    "impact_per_capita",
    "aid_percentage",
    "year",
    "month",
]

categoricalFeatures = ["event_type", "country"]

print("Data Overview")
print("-"* 200)
print(df.head(), "\n") #shows first four columns of the data set

#univariate analysis (target + candidate numeric predictors)
print("Central Tendency & Dispersion (Key Numeric Variables)")
print("-" * 160)
keyUnivariate = df[[target] + numericFeatures].describe().T
keyUnivariate["range"] = keyUnivariate["max"] - keyUnivariate["min"]
print(keyUnivariate, "\n")

print("Mode (Key Numeric Variables)")
print("-" * 50)
for col in [target] + numericFeatures:
    print(f"{col}: {df[col].mode().iloc[0]}")
print()

#distribution of the target
sns.set(style="whitegrid")
plt.figure(figsize=(9, 4.5))
sns.histplot(df[target], bins=40, kde=True)
plt.title("Distribution of Economic Impact (USD millions)")
plt.xlabel("Economic Impact (USD millions)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# Boxplot of the target to visualize spread/outliers
plt.figure(figsize=(9, 2.8))
sns.boxplot(x=df[target])
plt.title("Boxplot: Economic Impact (USD millions)")
plt.xlabel("Economic Impact (USD millions)")
plt.tight_layout()
plt.show()


print("Correlation Between Numeric Variables and Economic Impact")  
print("-" * 100)
corr_data = df[numericFeatures + [target]].corr(numeric_only=True)  # gets correlation between all numeric columns
print(corr_data[target].sort_values(ascending=False))  # shows which features are most related to the target

# heatmap that shows correlation between numeric variables and target
plt.figure(figsize=(10, 6))  # makes the plot bigger so it’s easy to read
sns.heatmap(corr_data, annot=True, cmap='coolwarm', fmt=".2f")  # heatmap with color scale and correlation numbers
plt.title('Correlation Heatmap - Climate Impact Dataset')  # title of plot
plt.tight_layout()
plt.grid(True)  # adds grid for readability
plt.show()
# Analysis: We can use this heatmap to see which numeric columns are most related to economic impact.

# find which variable is most correlated with economic impact
top_corr_var = corr_data[target].drop(target).abs().idxmax()  # finds the variable with the strongest relationship
print(f"Most correlated variable with {target}: {top_corr_var}")  # prints it out

# scatterplot to show relationship between top correlated variable and target
plt.figure(figsize=(8, 5))
sns.scatterplot(x=top_corr_var, y=target, data=df, hue='event_type', alpha=0.7)  # colors by event type
plt.title(f'{top_corr_var} vs Economic Impact')  # title for plot
plt.xlabel(top_corr_var)  # label x-axis
plt.ylabel('Economic Impact (USD millions)')  # label y-axis
plt.grid(True)
plt.tight_layout()
plt.show()
# Analysis: if the scatter shows an upward trend, that means as the variable increases, so does the impact.

# 1. Average impact by event type
plt.figure(figsize=(10, 5))
sns.barplot(x='event_type', y=target, data=df, estimator=np.mean, errorbar=None)  # bar plot showing average impact per event
plt.title('Average Economic Impact by Event Type')  # shorter, clean title
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()
# My Analysis: This shows which event types (like floods, droughts, etc.) cause the biggest average economic loss.

# 2. Top 10 countries by average economic impact
top_countries = df.groupby('country')[target].mean().nlargest(10).index  # gets top 10 countries with highest average impact
plt.figure(figsize=(10, 5))
sns.barplot(x='country', y=target, data=df[df['country'].isin(top_countries)], estimator=np.mean, errorbar=None)
plt.title('Top 10 Countries by Avg Economic Impact')
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()
# My Analysis: Helps us see which countries face the most financial damage from climate events.

# 3. Total economic impact trend by year
plt.figure(figsize=(10, 5))
sns.lineplot(x='year', y=target, data=df, estimator=np.sum, errorbar=None, marker='o')  # line chart showing total yearly impact
plt.title('Total Economic Impact by Year (2020–2025)')
plt.xlabel('Year')
plt.ylabel('Total Impact (USD millions)')
plt.grid(True)
plt.tight_layout()
plt.show()
# My Analysis: If the line goes down, it means total impact is decreasing over time — could show better disaster management or fewer large events.

# --- Summary print at the end ---
print("\nSummary:")
print(f"- The variable most correlated with {target} is: {top_corr_var}.")
print("- Heatwaves and droughts show the highest average impacts among event types.")
print("- The United States and China experience the highest average economic losses.")
print("- Total global impact seems to decline after 2020, possibly due to improved response efforts.\n")

# PART 3 – REGRESSION MODELING WITH P-VALUES + TUNING (Predicting Economic Impact (USD millions))

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor
import statsmodels.api as sm

# Encode Categorical Features

# make a copy of dataset
data = df.copy()  # to keep original and avoid changes

# drop ID/time string columns before modeling
data = data.drop(columns=["event_id", "date"])

# convert event_type and country into dummy variables
data = pd.get_dummies(data, columns=categoricalFeatures, drop_first=True) # creates dummy/indicators variable like event_type_Flood etc
# goal to to turn all features into numeric inputs for the model

# Separate features and target
X_full = data.drop(target, axis=1) # our predictors
y = data[target] # what we want to predict (economic impact)

# log transform to make the target less skewed
y_log = np.log1p(y)   # log(impact + 1), keeps 0 safe

# we used log transform because economic impact is highly skewed with huge outliers. it helps the model fit better

# Train/Test split (80/20 as required)
X_train, X_test, y_train, y_test = train_test_split(
    X_full, y_log, test_size=0.2, random_state=42
)
# standard train test split we've been using

print("Train/Test Shapes:", X_train.shape, X_test.shape) # to see if split correctly

# OLS regression to get p-values (we used NumPy arrays to avoid dtype issues - runtime errors from terminal)

# convert X and y to NumPy float arrays
X_train_np = X_train.values.astype(float)
y_train_np = y_train.values.astype(float)

# add intercept column manually
X_train_sm = sm.add_constant(X_train_np)

# fit OLS
ols_model = sm.OLS(y_train_np, X_train_sm).fit()

print("\n=== OLS SUMMARY (for p-values) ===")
print(ols_model.summary())

# p-values for each feature (skip the intercept at index 0)
p_values = pd.Series(ols_model.pvalues[1:], index=X_train.columns)

# Variables with p < 0.05 and p < 0.01
sig_005 = p_values[p_values < 0.05].index.tolist()
sig_001 = p_values[p_values < 0.01].index.tolist()

# Using p < 0.05 lets us keep predictors that are meaningful and drop noisy ones, which reduces overfitting and improves stability.

print("\nNumber of predictors with p < 0.05:", len(sig_005)) # gets the length 
print("Predictors (p < 0.05):", sig_005) # prints the predictors that satisfy the command

print("\nNumber of predictors with p < 0.01:", len(sig_001)) # gets the lenght
print("Predictors (p < 0.01):", sig_001) # predictors for p < 0.01

# use the p<0.05 features for the rest of the models
X_train_sig = X_train[sig_005]
X_test_sig  = X_test[sig_005]


# Scale Features (needed for Ridge/Lasso/ElasticNet/KNN)

scaler = StandardScaler()
X_train_sig_scaled = scaler.fit_transform(X_train_sig)
X_test_sig_scaled = scaler.transform(X_test_sig)

# we scaled the data because some models works better when all features are on the same scale

# functions we'll reuse
def evaluate_model(name, model, Xtr, ytr, Xte, yte):
    #This function trains a model and calculates how good its predictions are.
    #Fits a regression model, predicts on test data, and returns MSE/RMSE/MAE
    #on the original (unlogged) scale of the target.
    
    model.fit(Xtr, ytr)
    preds_log = model.predict(Xte)

    # convert back to actual USD (undo log1p)
    preds = np.expm1(preds_log)
    actual = np.expm1(yte)
 # we used expm1 because we trained on log1p of the target. expm1 reverses that so we can evaluate errors in the original dollar units.
    mse = mean_squared_error(actual, preds)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actual, preds)

    return {
        "Model": name,
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae
    }

def tune_alpha(model_class, alphas, Xtr_scaled, ytr, name_prefix):
   # This function tries different alpha values (strength of regularization). 
   # Simple manual hyperparameter tuning:
   # we split the training set again into train/validation and
   # pick the alpha with the lowest validation RMSE.
    
    X_sub_tr, X_val, y_sub_tr, y_val = train_test_split(
        Xtr_scaled, ytr, test_size=0.2, random_state=42
    )

    best_alpha = None
    best_rmse = np.inf

    for a in alphas:
        model = model_class(alpha=a)
        model.fit(X_sub_tr, y_sub_tr)

        preds_val_log = model.predict(X_val)
        preds_val = np.expm1(preds_val_log)
        actual_val = np.expm1(y_val)

        rmse_val = np.sqrt(mean_squared_error(actual_val, preds_val))
        print(f"{name_prefix} alpha={a}: Validation RMSE = {rmse_val:.2f}")

        if rmse_val < best_rmse:
            best_rmse = rmse_val
            best_alpha = a

    print(f"Best alpha for {name_prefix}: {best_alpha} (Val RMSE = {best_rmse:.2f})\n")
    return best_alpha

def tune_elasticnet(alphas, l1_ratios, Xtr_scaled, ytr):
    # purpose of this function is to tune both hyperparameters and get the best ElasticNet model.
     #Tune both alpha and l1_ratio for ElasticNet using a validation split.
    
    X_sub_tr, X_val, y_sub_tr, y_val = train_test_split(
        Xtr_scaled, ytr, test_size=0.2, random_state=42
    )

    best_params = None
    best_rmse = np.inf

    for a in alphas:
        for l1 in l1_ratios:
            model = ElasticNet(alpha=a, l1_ratio=l1)
            model.fit(X_sub_tr, y_sub_tr)

            preds_val_log = model.predict(X_val)
            preds_val = np.expm1(preds_val_log)
            actual_val = np.expm1(y_val)

            rmse_val = np.sqrt(mean_squared_error(actual_val, preds_val))
            print(f"ElasticNet alpha={a}, l1_ratio={l1}: Validation RMSE = {rmse_val:.2f}")

            if rmse_val < best_rmse:
                best_rmse = rmse_val
                best_params = (a, l1)

    print(f"\nBest ElasticNet params: alpha={best_params[0]}, l1_ratio={best_params[1]} (Val RMSE = {best_rmse:.2f})\n")
    return best_params

def tune_k(k_values, Xtr_scaled, ytr):
   # purpose of this is to find the number of neighbors that gives the most accurate predictions. 
   # tuning for KNN: try different k values and select the best one.
    
    X_sub_tr, X_val, y_sub_tr, y_val = train_test_split(
        Xtr_scaled, ytr, test_size=0.2, random_state=42
    )

    best_k = None
    best_rmse = np.inf

    for k in k_values:
        model = KNeighborsRegressor(n_neighbors=k)
        model.fit(X_sub_tr, y_sub_tr)

        preds_val_log = model.predict(X_val)
        preds_val = np.expm1(preds_val_log)
        actual_val = np.expm1(y_val)

        rmse_val = np.sqrt(mean_squared_error(actual_val, preds_val))
        print(f"KNN k={k}: Validation RMSE = {rmse_val:.2f}")

        if rmse_val < best_rmse:
            best_rmse = rmse_val
            best_k = k

    print(f"\nBest k for KNN: {best_k} (Val RMSE = {best_rmse:.2f})\n")
    return best_k

# train the models and tune them

results = []

# We used part of the training data to test different versions of each model (different α values or k values) 
# so that we could choose the best model before evaluating on the test set. 
# This makes sure our test accuracy is honest and unbiased.


# Baseline Linear Regression model (without tuning)
lin = LinearRegression()
results.append(
    evaluate_model("Linear Regression (p<0.05 vars)", lin, X_train_sig_scaled, y_train, X_test_sig_scaled, y_test)
)

# values to try for tuning
alphas = [0.001, 0.01, 0.1, 1, 5, 10]
l1_ratios = [0.2, 0.5, 0.8]

# Ridge model (tuned alpha)
best_alpha_ridge = tune_alpha(Ridge, alphas, X_train_sig_scaled, y_train, "Ridge")
ridge_best = Ridge(alpha=best_alpha_ridge)
results.append(
    evaluate_model(f"Ridge (alpha={best_alpha_ridge})", ridge_best, X_train_sig_scaled, y_train, X_test_sig_scaled, y_test)
)

# Lasso model (tuned alpha)
best_alpha_lasso = tune_alpha(Lasso, alphas, X_train_sig_scaled, y_train, "Lasso")
lasso_best = Lasso(alpha=best_alpha_lasso)
results.append(
    evaluate_model(f"Lasso (alpha={best_alpha_lasso})", lasso_best, X_train_sig_scaled, y_train, X_test_sig_scaled, y_test)
)

# ElasticNet model (tuned alpha + l1_ratio)
best_alpha_en, best_l1 = tune_elasticnet(alphas, l1_ratios, X_train_sig_scaled, y_train)
elastic_best = ElasticNet(alpha=best_alpha_en, l1_ratio=best_l1)
results.append(
    evaluate_model(f"ElasticNet (alpha={best_alpha_en}, l1={best_l1})", elastic_best, X_train_sig_scaled, y_train, X_test_sig_scaled, y_test)
)

# KNN Regression (tuned k)
k_values = [3, 5, 7, 9]
best_k = tune_k(k_values, X_train_sig_scaled, y_train)
knn_best = KNeighborsRegressor(n_neighbors=best_k)
results.append(
    evaluate_model(f"KNN Regression (k={best_k})", knn_best, X_train_sig_scaled, y_train, X_test_sig_scaled, y_test)
)

# Compare All Models

results_df = pd.DataFrame(results)
print("\n MODEL PERFORMANCE SUMMARY (Original USD Scale) ")
print(results_df.sort_values("RMSE"))

# Linear regression ended up the best once we cleaned the predictors using p-values, the remaining variables worked really well with Linear Regression. 
# Since the relationships were mostly linear, regularization didn’t improve anything. 
# KNN struggled because it doesn’t work great with many features and mixed data types.”


# Plots for the Best Model

# Choose the model with lowest RMSE
best_row = results_df.sort_values("RMSE").iloc[0]
best_model_name = best_row["Model"]
print(f"\nBest Model Based on RMSE: {best_model_name}")

# Map name back to model object
model_map = {
    "Linear Regression (p<0.05 vars)": lin,
    f"Ridge (alpha={best_alpha_ridge})": ridge_best,
    f"Lasso (alpha={best_alpha_lasso})": lasso_best,
    f"ElasticNet (alpha={best_alpha_en}, l1={best_l1})": elastic_best,
    f"KNN Regression (k={best_k})": knn_best
}
best_model = model_map[best_model_name]

# Fit on full training data and predict
best_model.fit(X_train_sig_scaled, y_train)
best_preds_log = best_model.predict(X_test_sig_scaled)

#  Coefficients for the best model (only works for linear-type models) 
if best_model_name.startswith("Linear Regression") or \
   best_model_name.startswith("Ridge") or \
   best_model_name.startswith("Lasso") or \
   best_model_name.startswith("ElasticNet"):
    
    coef_table = pd.DataFrame({
        "Feature": X_train_sig.columns,
        "Coefficient": best_model.coef_
    }).sort_values(by="Coefficient", ascending=False)

    print("\nTop 15 Coefficients for Best Model:")
    print(coef_table.head(15))

    print("\nBottom 15 Coefficients for Best Model:")
    print(coef_table.tail(15))


# Back-transform to original scale
y_test_actual = np.expm1(y_test)
best_preds_actual = np.expm1(best_preds_log)

# True vs Predicted
plt.figure(figsize=(7, 6))
sns.scatterplot(x=y_test_actual, y=best_preds_actual, alpha=0.6)
max_val = max(y_test_actual.max(), best_preds_actual.max())
plt.plot([0, max_val], [0, max_val], 'r--')
plt.title(f"True vs Predicted Economic Impact ({best_model_name})")
plt.xlabel("Actual Impact (USD millions)")
plt.ylabel("Predicted Impact (USD millions)")
plt.grid(True)
plt.tight_layout()
plt.show()

# Residual Plot
residuals = y_test_actual - best_preds_actual

plt.figure(figsize=(7, 5))
sns.scatterplot(x=best_preds_actual, y=residuals, alpha=0.6)
plt.axhline(0, color='red', linestyle='--')
plt.title(f"Residual Plot ({best_model_name})")
plt.xlabel("Predicted Impact")
plt.ylabel("Residuals (Actual - Predicted)")
plt.grid(True)
plt.tight_layout()
plt.show()

# Analysis

print("\n--- Analysis ---")
print("- We first ran an OLS regression with statsmodels and used the p-values to filter predictors,")
print("  keeping only variables with p < 0.05. This removed noisy features and made the models more stable.")
print("- We then trained and compared Linear Regression, Ridge, Lasso, ElasticNet, and KNN using the same")
print("  filtered feature set. Hyperparameters were tuned using a validation split inside the training data.")
print(f"- Linear Regression (p < 0.05 predictors) achieved the lowest RMSE, meaning it predicted economic")
print("  impact more accurately than the regularized models or KNN.")
print("- After selecting the best model, we examined its coefficients. Variables such as infrastructure_damage_score,")
print("  aid_percentage, impact_per_capita, and affected_population had the strongest positive coefficients,")
print("  meaning these factors increase predicted economic loss the most.")
print("- Event-type dummy variables (such as heatwaves and droughts) also had positive coefficients, matching the")
print("  earlier data visualization trends showing these events cause higher average damages.")
print("- KNN performed the worst because it struggles with high-dimensional numeric + dummy variable datasets,")
print("  and it does not model linear relationships as effectively.")
print("- The True vs Predicted plot shows the model fits small and medium events well but underestimates extreme")
print("  mega-disasters, which is expected due to the heavy-tailed distribution of climate losses.")
print("- Overall, the model is highly useful for forecasting typical disaster costs and supporting budgeting,")
print("  early planning, and risk assessment, but extreme events remain challenging to predict.")
