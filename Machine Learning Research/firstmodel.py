from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import random

# 1. Get the data: X = measurements, y = species label
X, y = load_iris(return_X_y=True)

# 2. Split: 80% to learn from, 20% held back for testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=random.randint(0, 100)
)

# 3. Create and train the model
model = KNeighborsClassifier(n_neighbors=50)
model.fit(X_train, y_train)

# 4. Predict on data it has never seen, then score it
predictions = model.predict(X_test)
print(accuracy_score(y_test, predictions))