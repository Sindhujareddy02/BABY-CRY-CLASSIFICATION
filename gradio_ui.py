from frequency_features import extract_features
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
import joblib
import numpy as np

# 🔍 Load features and labels
X, y = extract_features(r"C:\Users\sindh\OneDrive\Desktop\cry")

# 🧪 Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 📊 Optional: Scale features (comment out if not needed)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 🔍 Hyperparameter search space
param_dist = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

# 🔁 Tune model
rf = RandomForestClassifier(random_state=42)
clf = RandomizedSearchCV(rf, param_distributions=param_dist, n_iter=10, cv=3, n_jobs=-1, verbose=1)
clf.fit(X_train, y_train)

# 🎯 Predict and evaluate
y_pred = clf.predict(X_test)
print("📈 Classification Report:")
print(classification_report(y_test, y_pred))

# 💾 Save model
joblib.dump(clf.best_estimator_, "model.pkl")
print("✅ Optimized model saved!")