import joblib
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

data = load_iris()
model = RandomForestClassifier(n_estimators=50, random_state=42)
model.fit(data.data, data.target)

artifact = {
    "model": model,
    "target_names": [str(name) for name in data.target_names],
}
joblib.dump(artifact, "model.joblib")
print("Saved model.joblib successfully.")