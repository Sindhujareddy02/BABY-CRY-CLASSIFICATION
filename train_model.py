from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
from frequency_features import extract_features

X, y = extract_features("C:/Users/sindh/OneDrive/Desktop/cry/Baby Crying Sounds")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = RandomForestClassifier()
clf.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test)
print(f"✅ Model trained. Accuracy: {accuracy * 100:.2f}%")

joblib.dump(clf, "model.pkl")
print("💾 Model saved as model.pkl")
