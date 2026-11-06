
# Random Forest Classification API

![Python](https://img.shields.io/badge/Python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-API-green)
![Docker](https://img.shields.io/badge/Docker-Containerized-blue)
![CI](https://github.com/yashwanth86868/RandomForest-Classifier-API/actions/workflows/python-tests.yml/badge.svg)

## Project Overview

This project implements a **Random Forest Classification Machine Learning model** and deploys it as a REST API using FastAPI.

The API accepts user input such as age and estimated salary and returns a binary classification prediction.

The project demonstrates machine learning, API development, model serialization, containerization, automated testing, and continuous integration.

## Technologies Used

- Python
- NumPy
- Pandas
- Scikit-learn
- FastAPI
- Uvicorn
- Docker
- Docker Compose
- Pytest
- Git and GitHub
- GitHub Actions

## Project Structure

```text
RandomForest_API/
│
├── .github/
│   └── workflows/
│       └── python-tests.yml
│
├── tests/
│   └── test_api.py
│
├── app.py
├── train_model.py
├── random_forest_model.pkl
├── requirements.txt
├── Dockerfile
├── compose.yaml
└── README.md
```

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/yashwanth86868/RandomForest-Classifier-API.git
```

### 2. Navigate to the project directory

```bash
cd RandomForest-Classifier-API
```

### 3. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```powershell
.\venv\Scripts\Activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

## Run the API

Start the FastAPI server:

```bash
uvicorn app:app --reload
```

Open your browser:

- API Home: http://127.0.0.1:8000/
- Swagger API Documentation: http://127.0.0.1:8000/docs

## API Usage

### Prediction Endpoint

**Method:** POST

**Endpoint:** `/predict`

Example request:

```json
{
    "age": 30,
    "salary": 50000
}
```

Example response:

```json
{
    "prediction": 1
}
```

The prediction is a binary classification result (0 or 1).

## Run with Docker

Build the Docker image:

```bash
docker build -t random-forest-api .
```

Run the container:

```bash
docker run -p 8000:8000 random-forest-api
```

Open:

http://127.0.0.1:8000/docs

### Docker Compose

Start the application:

```bash
docker compose up --build
```

Stop the application:

```bash
docker compose down
```

## Automated Testing

This project uses Pytest and FastAPI TestClient to test API endpoints.

Run tests:

```bash
python -m pytest tests -v
```

The tests verify:

- Home endpoint response.
- Prediction endpoint response.
- HTTP status codes.
- Prediction output.

## Continuous Integration (CI)

GitHub Actions automatically runs the automated tests when code is pushed to the `main` branch or a pull request is created against `main`.

**CI Workflow:** `.github/workflows/python-tests.yml`

[View GitHub Actions](https://github.com/yashwanth86868/RandomForest-Classifier-API/actions)

## Key Learning Outcomes

- Building a machine learning classification model.
- Saving and loading trained ML models.
- Developing REST APIs using FastAPI.
- Testing API endpoints using Pytest.
- Containerizing applications using Docker.
- Managing containers using Docker Compose.
- Implementing CI pipelines using GitHub Actions.
- Managing source code using Git and GitHub.

## Author

**Yashwanth Balija**

GitHub: https://github.com/yashwanth86868

---

⭐ If you find this project useful, consider giving it a star!
