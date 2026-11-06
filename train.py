
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt

# from sklearn.model_selection import train_test_split, cross_val_score
# from sklearn.ensemble import RandomForestClassifier

# from sklearn.metrics import (
#     confusion_matrix,
#     accuracy_score,
#     classification_report,
#     ConfusionMatrixDisplay,
#     roc_auc_score
# )

# # 1. Load dataset
# dataset = pd.read_csv(
#     r"C:\Users\ybali\Downloads\logit classification.csv"
# )

# # 2. Separate independent and dependent variables
# X = dataset.iloc[:, 2:4].values
# y = dataset.iloc[:, -1].values

# # 3. Split dataset
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y,
#     test_size=0.20,
#     random_state=0,
#     stratify=y
# )

# # 4. Create Random Forest model
# classifier = RandomForestClassifier(
#     n_estimators=33,
#     max_depth=5,
#     criterion="entropy",
#     max_features=None,
#     random_state=0
# )

# # 5. Train model
# classifier.fit(X_train, y_train)

# # 6. Prediction
# y_pred = classifier.predict(X_test)

# # 7. Evaluation
# print("Confusion Matrix:")
# print(confusion_matrix(y_test, y_pred))

# print("\nAccuracy:")
# print(accuracy_score(y_test, y_pred))

# print("\nClassification Report:")
# print(classification_report(y_test, y_pred))

# # 8. Training and testing accuracy
# print("Training Accuracy:", classifier.score(X_train, y_train))
# print("Testing Accuracy:", classifier.score(X_test, y_test))

# # 9. Cross-validation
# scores = cross_val_score(
#     classifier, X_train, y_train, cv=5, scoring="accuracy"
# )

# print("Cross Validation Scores:", scores)
# print("Mean CV Accuracy:", scores.mean())

# # 10. Feature importance
# print("Feature Importance:", classifier.feature_importances_)

# # 11. Confusion matrix visualization
# ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
# plt.show()








# from sklearn.model_selection import GridSearchCV

# parameters = {
#     'n_estimators': [50, 100, 200],
#     'max_depth': [3, 5, 7, None],
#     'min_samples_split': [2, 5, 10],
#     'min_samples_leaf': [1, 2, 4]
# }

# grid = GridSearchCV(
#     RandomForestClassifier(
#         random_state=0,
#         criterion='entropy'
#     ),
#     parameters,
#     cv=5,
#     scoring='accuracy',
#     n_jobs=-1
# )

# grid.fit(x_train, y_train)

# print("Best Parameters:", grid.best_params_)
# print("Best CV Accuracy:", grid.best_score_)

# best_model = grid.best_estimator_

# print("Test Accuracy:", best_model.score(x_test, y_test))











# import pandas as pd
# import numpy as np

# from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score, classification_report

# # 1. Load dataset
# dataset = pd.read_csv(
#     r"C:\Users\ybali\Downloads\logit classification.csv"
# )

# # 2. Independent and dependent variables
# x = dataset.iloc[:, 2:4].values
# y = dataset.iloc[:, -1].values

# # 3. Split dataset
# x_train, x_test, y_train, y_test = train_test_split(
#     x, y,
#     test_size=0.20,
#     random_state=0,
#     stratify=y
# )

# # 4. Create Random Forest model
# classifier = RandomForestClassifier(random_state=0)

# # 5. Define hyperparameters

# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import RandomizedSearchCV
# from sklearn.metrics import accuracy_score

# # Parameter combinations
# parameters = {
#     'n_estimators': [50, 100, 150, 200, 300],
#     'max_depth': [3, 5, 7, 10, None],
#     'criterion': ['gini', 'entropy'],
#     'min_samples_split': [2, 5, 10],
#     'min_samples_leaf': [1, 2, 4]
# }

# # Create Random Forest
# classifier = RandomForestClassifier(random_state=0)

# # Randomized Search
# random_search = RandomizedSearchCV(
#     estimator=classifier,
#     param_distributions=parameters,
#     n_iter=20,
#     cv=5,
#     scoring='accuracy',
#     random_state=0,
#     n_jobs=-1
# )

# # Train
# random_search.fit(x_train, y_train)

# # Best parameters
# print("Best Parameters:", random_search.best_params_)

# # Best CV accuracy
# print("Best CV Accuracy:", random_search.best_score_)

# # Best model
# best_model = random_search.best_estimator_

# # Prediction
# y_pred = best_model.predict(x_test)

# # Test accuracy
# print("Test Accuracy:", accuracy_score(y_test, y_pred))


# # 11. Prediction
# y_pred = best_model.predict(x_test)

# # 12. Final evaluation
# print("Test Accuracy:", accuracy_score(y_test, y_pred))

# print("Classification Report:")
# print(classification_report(y_test, y_pred))





import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. Load dataset
dataset = pd.read_csv(
    r"C:\Users\ybali\Downloads\logit classification.csv"
)

# 2. Independent and dependent variables
x = dataset.iloc[:, 2:4].values
y = dataset.iloc[:, -1].values

# 3. Split dataset
x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.20,
    random_state=0,
    stratify=y
)

# 4. Create Random Forest model
classifier = RandomForestClassifier(random_state=0)

# 5. Define hyperparameters
parameters = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7, None],
    'criterion': ['gini', 'entropy'],
    'min_samples_split': [2, 5],
}

# 6. GridSearchCV
grid = GridSearchCV(
    estimator=classifier,
    param_grid=parameters,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

# 7. Train and find best parameters
grid.fit(x_train, y_train)

# 8. Best parameters
print("Best Parameters:", grid.best_params_)

# 9. Best cross-validation score
print("Best CV Accuracy:", grid.best_score_)

# 10. Best model
best_model = grid.best_estimator_

# 11. Prediction
y_pred = best_model.predict(x_test)

# 12. Final evaluation
print("Test Accuracy:", accuracy_score(y_test, y_pred))

print("Classification Report:")
print(classification_report(y_test, y_pred))





import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    auc
)

# Get positive-class probabilities
y_probability = best_model.predict_proba(x_test)[:, 1]

# Calculate ROC values
fpr, tpr, thresholds = roc_curve(
    y_test,
    y_probability
)

# Calculate AUC
roc_auc = roc_auc_score(
    y_test,
    y_probability
)

print("AUC Score:", roc_auc)

# Plot ROC curve
plt.figure(figsize=(7, 5))

plt.plot(
    fpr,
    tpr,
    label=f"Random Forest (AUC = {roc_auc:.3f})"
)

# Random classifier reference
plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid()
plt.show()


print("AUC Score:", roc_auc)




import pandas as pd
import matplotlib.pyplot as plt

importance = best_model.feature_importances_

feature_names = dataset.columns[2:4]

importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importance
})

print(importance_df)

importance_df.plot(
    x='Feature',
    y='Importance',
    kind='bar',
    legend=False
)

plt.title("Feature Importance")
plt.ylabel("Importance")
plt.show()



import pandas as pd
import matplotlib.pyplot as plt

importance = best_model.feature_importances_

feature_names = dataset.columns[2:4]

importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importance
})

print(importance_df)

importance_df.plot(
    x='Feature',
    y='Importance',
    kind='bar',
    legend=False
)

plt.title("Feature Importance")
plt.ylabel("Importance")
plt.show()








from sklearn.feature_selection import SelectKBest, f_classif

# Select top 2 features
selector = SelectKBest(
    score_func=f_classif,
    k=2
)

# Fit and transform training data
x_train_selected = selector.fit_transform(x_train, y_train)

# Transform testing data
x_test_selected = selector.transform(x_test)

# Selected feature names
selected_features = dataset.columns[2:4][
    selector.get_support()
]

print("Selected Features:", selected_features)
print("Feature Scores:", selector.scores_)







from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier

# Create model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=0
)

# Create RFE
selector = RFE(
    estimator=model,
    n_features_to_select=1
)

# Fit RFE
selector.fit(x_train, y_train)

# Selected feature names
selected_features = dataset.columns[2:4][
    selector.support_
]

print("Selected Features:", selected_features)
print("Feature Ranking:", selector.ranking_)



import pickle

# Save trained model
with open("random_forest_model.pkl", "wb") as file:
    pickle.dump(best_model, file)

print("Model saved successfully!")

import pickle

with open("random_forest_model.pkl", "rb") as file:
    loaded_model = pickle.load(file)

print("Model loaded successfully!")


new_data = [[30, 50000]]

prediction = loaded_model.predict(new_data)

print("Prediction:", prediction)