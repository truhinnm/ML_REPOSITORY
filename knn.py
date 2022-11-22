import numpy as np
from sklearn import datasets
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

iris = datasets.load_iris()
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_df['target'] = list(map(lambda x: iris.target_names[x], iris.target))
target = pd.Series(iris.target, name='target')

sns.pairplot(iris_df, hue='target', height=3, diag_kind="kde")
plt.show()

iris_df_norm = pd.DataFrame(data=StandardScaler().fit_transform(iris_df.drop('target', axis=1)),
                            columns=iris.feature_names)
iris_df_norm['target'] = iris_df['target']

sns.pairplot(iris_df_norm, hue='target', height=3, diag_kind="kde")
plt.show()

iris_df = iris_df.drop('target', axis=1)


class KNNClassifier(BaseEstimator):
    def __init__(self, k_neighbors=5):
        self.k_neighbors = k_neighbors

    def __to_numpy(self, d):
        if isinstance(d, pd.DataFrame) or isinstance(d, pd.Series):
            return d.to_numpy()
        if isinstance(d, list):
            return np.array(d)

        return d

    def fit(self, X, y):
        self.X, self.y = self.__to_numpy(X), self.__to_numpy(y)
        self.classes = np.sort(np.unique(y))

        return self

    def predict_proba(self, X):
        points = self.__to_numpy(X)
        results = []
        for p in points:
            probs = []
            neighbors = self.y[np.sum((self.X - p) ** 2, axis=1).argsort()[:self.k_neighbors]]
            for c in self.classes:
                probs.append((neighbors == c).sum() / self.k_neighbors)
            results.append(probs)

        return np.array(results)

    def predict(self, X):
        probs = self.predict_proba(X)
        return probs.argmax(axis=1)


pipe = Pipeline([
    ('std', StandardScaler()),
    ('knn', KNNClassifier())
])

param_grid = {
    "knn__k_neighbors": np.arange(1, 24, 1)
}

grid = GridSearchCV(pipe, param_grid, cv=5, n_jobs=-1, scoring="accuracy", error_score='raise', verbose=True)

X_train, X_test, y_train, y_test = train_test_split(iris_df, target, shuffle=True)
knn = KNNClassifier(k_neighbors=1)
knn.fit(X_train, y_train)


X_test['pred_target'] = list(map(lambda x: iris.target_names[x], knn.predict(X_test)))
X_test['real_target'] = list(map(lambda x: iris.target_names[x], y_test))

sns.pairplot(X_test, hue='real_target', height=3, diag_kind="kde", hue_order=['versicolor', 'virginica', 'setosa'])
plt.show()

sns.pairplot(X_test, hue='pred_target', height=3, diag_kind="kde", hue_order=['versicolor', 'virginica', 'setosa'])
plt.show()