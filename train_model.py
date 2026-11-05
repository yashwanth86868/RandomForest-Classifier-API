
import pandas as pd
import pickle
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load dataset
dataset = pd.read_csv(
    r"C:\Users\ybali\Downloads\logit classification.csv"
)

# Independent and dependent variables
x = dataset.iloc[:, 2:4].values
y = dataset.iloc[:, -1].values

# Split dataset
x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.20,
    random_state=0,
    stratify=y
)

# Create optimized Random Forest model
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=3,
    criterion="entropy",
    min_samples_split=2,
    random_state=0
)

# Train model
model.fit(x_train, y_train)

# Save model in current project folder
model_path = Path(__file__).parent / "random_forest_model.pkl"

with open(model_path, "wb") as file:
    pickle.dump(model, file)

print("Model saved successfully!")
print("Model size:", model_path.stat().st_size, "bytes")
print("Test Accuracy:", model.score(x_test, y_test))
