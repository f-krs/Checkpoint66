import pandas as pd

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

iris = load_iris()

X = iris.data
y = iris.target

print(X.shape)
print(y.shape)

X_df = pd.DataFrame(X, columns=iris.feature_names)

print(X_df.head())
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print(X_train.shape)
print(X_test.shape)
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(y_pred[:10])

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy :", accuracy)

cm = confusion_matrix(y_test, y_pred)
print("Matrice de confusion :")
print(cm)
print(classification_report(y_test, y_pred))

scores = cross_val_score(
    model,
    X,
    y,
    cv=5,
    scoring='accuracy'
)

print("Scores par pli :", scores)
print("Accuracy moyenne :", scores.mean())

