from fastapi.testclient import TestClient
from app.main import app

def test_health():
    with TestClient(app) as client:
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

def test_predict():
    payload = {
        "Pregnancies": 2,
        "Glucose": 120,
        "BloodPressure": 70,
        "SkinThickness": 20,
        "Insulin": 79,
        "BMI": 28.0,
        "DiabetesPedigreeFunction": 0.5,
        "Age": 33,
    }
    with TestClient(app) as client:
        r = client.post("/predict", json=payload)
        assert r.status_code == 200
        data = r.json()
        assert "prediction" in data
        assert "probability" in data
        assert 0.0 <= data["probability"] <= 1.0

