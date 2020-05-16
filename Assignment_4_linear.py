import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import datasets
from yellowbrick.cluster import KElbowVisualizer

# 1. Use a linear regression model with the Boston housing data set. Your code should then return
# which factor has the largest effect on the price of housing in Boston. (This is not the correlation coefficient.
# This is the absolute value of the slope.)

def linear_boston():
    boston = datasets.load_boston()

    my_y = boston.target
    my_x = boston.data

    lr = LinearRegression()

    lr.fit(my_x, my_y)

    results = pd.DataFrame(
        abs(lr.coef_), index=boston.feature_names, columns=["Slope"]
    ).sort_values(by=["Slope"], ascending=False)


    print(
        "The factor having the largest effect on housing prices is",
        results.index[0],
        "with a coefficient of",
        results["Slope"].iloc[0],
    )

if __name__ == "__main__":
    linear_boston()

