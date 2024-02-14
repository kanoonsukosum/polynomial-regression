import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Import the dataset
data = pd.read_csv("top_six_economies.csv")  # load data set
df = pd.DataFrame(data)

# Filtering the DataFrame for entries where Country is Japan
japanese = df[df['Country Name'] == 'Japan']

# Sort the DataFrame based on 'Year'
japanese = japanese.sort_values(by='Year')

###########Part 1##############
# Plotting the filtered data using a line plot
japanese.plot(x='Year', y='GDP (current US$)', kind='scatter', marker='o', label='Data Points')

# Adding labels and title
plt.xlabel('Year')
plt.ylabel('GDP (current US$)')
plt.title('Best-Fitted Polynomial to GDP Data with Year as X-axis')

# Initialize variables for best R-squared and corresponding degree
best_r_squared = -1
best_degree = -1

# Initialize variable to store coefficients
best_coefficients = None

# Iterate over different polynomial degrees
for degree in range(1, 31):  # You can adjust the range as needed
    X = np.arange(len(japanese)).reshape(-1, 1)
    y = japanese['GDP (current US$)'].values.reshape(-1, 1)

    poly_features = PolynomialFeatures(degree=degree)
    X_poly = poly_features.fit_transform(X)

    model = LinearRegression()
    model.fit(X_poly, y)
    y_poly_pred = model.predict(X_poly)

    # Calculate R-squared
    current_r_squared = r2_score(y, y_poly_pred)
    print(f"The Current R-squared of the 1st model with degree {degree}: {current_r_squared:.2f}")

    # Update best R-squared, degree, and coefficients if the current one is better
    if current_r_squared > best_r_squared:
        best_r_squared = current_r_squared
        best_degree = degree
        best_coefficients = model.coef_[0]

# Apply polynomial regression with the best degree
poly_features = PolynomialFeatures(degree=best_degree)
X_poly = poly_features.fit_transform(X)

model = LinearRegression()
model.fit(X_poly, y)
y_poly_pred = model.predict(X_poly)

# Plotting the regression curve as a line
plt.plot(japanese['Year'], y_poly_pred, label=f'Best Polynomial Regression (Degree {best_degree}, R²={best_r_squared:.2f})', color='red')

# Display best R-squared in the legend
plt.legend()

# Print coefficients
print(f"Coefficients for the best polynomial regression (Degree {best_degree}): {best_coefficients}")

year_to_predict = 2025  # Change this to the year you want to predict
X_new = np.array([[year_to_predict]])
X_poly_new = poly_features.transform(X_new)
predicted_gdp = model.predict(X_poly_new)
print(f"Predicted GDP for the year {year_to_predict}: {predicted_gdp[0][0]:,.2f} (in current US$)")

year_to_predict = 2020  # Change this to the year you want to predict
X_new = np.array([[year_to_predict]])
X_poly_new = poly_features.transform(X_new)
predicted_gdp = model.predict(X_poly_new)
print(f"Predicted GDP for the year {year_to_predict}: {predicted_gdp[0][0]:,.2f} (in current US$)")

year_to_predict = 2035  # Change this to the year you want to predict
X_new = np.array([[year_to_predict]])
X_poly_new = poly_features.transform(X_new)
predicted_gdp = model.predict(X_poly_new)
print(f"Predicted GDP for the year {year_to_predict}: {predicted_gdp[0][0]:,.2f} (in current US$)")


###########Part 2##############
# Plotting the filtered data using a scatter plot
japanese.plot(x='Unemployment, total (% of total labor force) (modeled ILO estimate)', y='GDP (current US$)', kind='scatter', marker='o', label='Data Points')

# Adding labels and title
plt.xlabel('Unemployment, total (% of total labor force)')
plt.ylabel('GDP (current US$)')
plt.title('Best-Fitted Polynomial to GDP Data with Unemployment as X-axis')

# Initialize variables for best R-squared and corresponding degree
best_r_squared = -1
best_degree = -1

# Initialize variable to store coefficients
best_coefficients = None

# Iterate over different polynomial degrees
for degree in range(1, 31):  # You can adjust the range as needed
    X = japanese['Unemployment, total (% of total labor force) (modeled ILO estimate)'].values.reshape(-1, 1)
    y = japanese['GDP (current US$)'].values.reshape(-1, 1)

    poly_features = PolynomialFeatures(degree=degree)
    X_poly = poly_features.fit_transform(X)

    model = LinearRegression()
    model.fit(X_poly, y)
    y_poly_pred = model.predict(X_poly)

    # Calculate R-squared
    current_r_squared = r2_score(y, y_poly_pred)
    print(f"The Current R-squared of the {degree}-degree model: {current_r_squared:.2f}")

    # Update best R-squared, degree, and coefficients if the current one is better
    if current_r_squared > best_r_squared:
        best_r_squared = current_r_squared
        best_degree = degree
        best_coefficients = model.coef_[0]

# Apply polynomial regression with the best degree
poly_features = PolynomialFeatures(degree=best_degree)
X_poly = poly_features.fit_transform(X)

model = LinearRegression()
model.fit(X_poly, y)
y_poly_pred = model.predict(X_poly)

# Sort the predicted values based on the feature values for a smooth curve
sorted_zip = sorted(zip(X[:, 0], y_poly_pred))
X_sort, y_poly_pred_sort = zip(*sorted_zip)

# Plotting the regression curve as a line
plt.plot(X_sort, y_poly_pred_sort, label=f'Best Polynomial Regression (Degree {best_degree}, R²={best_r_squared:.2f})', color='red')

# Display best R-squared in the legend
plt.legend()

# Print coefficients
print(f"Coefficients for the best polynomial regression (Degree {best_degree}): {best_coefficients}")

unemployment_rate_to_predict = 3  # Change this to the unemployment rate you want to predict
X_new = np.array([[unemployment_rate_to_predict]])
X_poly_new = poly_features.transform(X_new)
predicted_gdp = model.predict(X_poly_new)
print(f"Predicted GDP for an unemployment rate of {unemployment_rate_to_predict}%: {predicted_gdp[0][0]:,.2f} (in current US$)")

unemployment_rate_to_predict = 5  # Change this to the unemployment rate you want to predict
X_new = np.array([[unemployment_rate_to_predict]])
X_poly_new = poly_features.transform(X_new)
predicted_gdp = model.predict(X_poly_new)
print(f"Predicted GDP for an unemployment rate of {unemployment_rate_to_predict}%: {predicted_gdp[0][0]:,.2f} (in current US$)")

unemployment_rate_to_predict = 7  # Change this to the unemployment rate you want to predict
X_new = np.array([[unemployment_rate_to_predict]])
X_poly_new = poly_features.transform(X_new)
predicted_gdp = model.predict(X_poly_new)
print(f"Predicted GDP for an unemployment rate of {unemployment_rate_to_predict}%: {predicted_gdp[0][0]:,.2f} (in current US$)")

########################################
# Show the plot
plt.show()
