import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import datasets
from yellowbrick.cluster import KElbowVisualizer

# 2. Use a KMeans regression model with the Iris data set. Graph the fit when using differing numbers of clusters.
# Graph the result and either corroborate or refute the assumption that the data set represents 3 different
# varieties of iris.

def iris_cluster():
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


if __name__ == "__main__":
    iris_cluster()