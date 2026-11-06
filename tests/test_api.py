from fastapi.testclient import TestClient
from app import app

client = TestClient(app)


def test_home():
    response = client.get("/")

    assert response.status_code == 200


def test_prediction():
    response = client.post(
        "/predict",
        json={
            "age": 30,
            "salary": 50000
        }
    )

    assert response.status_code == 200
    assert response.json()["prediction"] in [0, 1]