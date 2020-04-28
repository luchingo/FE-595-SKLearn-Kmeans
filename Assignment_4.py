import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import datasets
from yellowbrick.cluster import KElbowVisualizer

# 1. Use a linear regression model with the Boston housing data set. Your code should then return
# which factor has the largest effect on the price of housing in Boston. (This is not the correlation coefficient.
# This is the absolute value of the slope.)

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


# 2. Use a KMeans regression model with the Iris data set. Graph the fit when using differing numbers of clusters.
# Graph the result and either corroborate or refute the assumption that the data set represents 3 different
# varieties of iris.

iris = datasets.load_iris()

iris_x = pd.DataFrame(iris.data, columns=iris.feature_names)
iris_y = iris.target

model = KMeans()

visualizer = KElbowVisualizer(model, k=(1, 20))
visualizer.fit(iris_x)
visualizer.show()

# As we can see from the graph plotted, the optimal number of cluster calculated using the elbow method
# is 3. To check this assumption I fit the KMean to my data and plot with different colors the various features
# among themselves:

fin_model = KMeans(n_clusters=visualizer.elbow_value_)

kmeans_fin = KMeans(n_clusters=visualizer.elbow_value_, random_state=10).fit(iris_x)

plt.subplot(3, 2, 1)
plt.scatter(
    iris_x.iloc[:, [0]], iris_x.iloc[:, [1]], c=kmeans_fin.labels_, cmap="rainbow"
)
plt.title("".join([iris_x.columns[0], " vs ", iris_x.columns[1]]))

plt.subplot(3, 2, 2)
plt.scatter(
    iris_x.iloc[:, [0]], iris_x.iloc[:, [2]], c=kmeans_fin.labels_, cmap="rainbow"
)
plt.title("".join([iris_x.columns[0], " vs ", iris_x.columns[2]]))

plt.subplot(3, 2, 3)
plt.scatter(
    iris_x.iloc[:, [0]], iris_x.iloc[:, [3]], c=kmeans_fin.labels_, cmap="rainbow"
)
plt.title("".join([iris_x.columns[0], " vs ", iris_x.columns[3]]))

plt.subplot(3, 2, 4)
plt.scatter(
    iris_x.iloc[:, [1]], iris_x.iloc[:, [2]], c=kmeans_fin.labels_, cmap="rainbow"
)
plt.title("".join([iris_x.columns[1], " vs ", iris_x.columns[2]]))

plt.subplot(3, 2, 5)
plt.scatter(
    iris_x.iloc[:, [1]], iris_x.iloc[:, [3]], c=kmeans_fin.labels_, cmap="rainbow"
)
plt.title("".join([iris_x.columns[1], " vs ", iris_x.columns[3]]))

plt.subplot(3, 2, 6)
plt.scatter(
    iris_x.iloc[:, [2]], iris_x.iloc[:, [3]], c=kmeans_fin.labels_, cmap="rainbow"
)
plt.title("".join([iris_x.columns[2], " vs ", iris_x.columns[3]]))

plt.show()
