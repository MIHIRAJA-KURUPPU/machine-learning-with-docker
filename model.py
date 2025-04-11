from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import pickle

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=43, test_size=0.2)

# Initialize and train the Random Forest model
clf = RandomForestClassifier(n_estimators=10)
clf.fit(X_train, y_train)

# Make predictions
predicted = clf.predict(X_test)

# Print accuracy score
print("Accuracy:", accuracy_score(y_test, predicted))

# Save the model to a file
with open("rf.pkl", "wb") as pkl:
    pickle.dump(clf, pkl)
