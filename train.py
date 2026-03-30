import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

X, y = load_iris(return_X_y=True)

model= RandomForestClassifier(n_estimators=100)
model.fit(X, y)

print("Training completed successfully")