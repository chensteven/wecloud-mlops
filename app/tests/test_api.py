import pytest
from fastapi.testclient import TestClient

from ..serve import app


@pytest.fixture
def client():
    # use "with" statement to run "startup" event of FastAPI
    with TestClient(app) as c:
        yield c


def test_serve_predict(client):
    """
    Test predction response
    """

    headers = {}
    body = {
        "sentence": "Deep learning and machine learning is super cool"
    }

    response = client.post("/api/v1/predict",
                           headers=headers,
                           json=body)

    try:
        assert response.status_code == 200
        reponse_json = response.json()
        assert reponse_json['error'] == False
        assert isinstance(reponse_json['results']['pred'], list)

    except AssertionError:
        print(response.status_code)
        print(response.json())
        raise