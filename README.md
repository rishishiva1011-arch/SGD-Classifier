# SGD-Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

Step 1: Load & Split Dataset
Load the Iris dataset using load_iris. Extract features X (sepal/petal length & width) and target y (species). Split into 80% train and 20% test using train_test_split.

Step 2: Scale Features
Apply StandardScaler on X_train using fit_transform and on X_test using transform only to normalize all feature values for stable SGD training.

Step 3: Train SGDClassifier
Initialize SGDClassifier with max_iter=1000 and random_state=42. Fit the model on the scaled training data to learn decision boundaries between the 3 flower species.

Step 4: Evaluate the Model
Predict species on the scaled test set. Print Accuracy Score and Confusion Matrix to measure how well the model classifies each species.

Step 5: Predict & Visualize
Scale the new flower input and predict its species using target_names for a readable output. Plot a scatter graph of Sepal Length vs Sepal Width colored by species class.

## Program:
```
/*
Program to implement the prediction of iris species using SGD Classifier.
Developed by: Rishikesh S
RegisterNumber:  212225240118
*/
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

iris = load_iris()

X = iris.data

y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = SGDClassifier()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

new_flower = [[5.1, 3.5, 1.4, 0.2]]

prediction = model.predict(new_flower)

print("Predicted Species:", iris.target_names[prediction][0])

plt.scatter(X[:,0], X[:,1], c=y)

plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.title("Iris Flower Classification")

plt.show()
```

## Output:

<img width="365" height="131" alt="image" src="https://github.com/user-attachments/assets/ac07e198-e238-4843-b7f1-729747d82eb1" />

<img width="568" height="455" alt="image" src="https://github.com/user-attachments/assets/af50fd6c-20cf-4e46-8e6e-861b1a0b3f27" />


## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
