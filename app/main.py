from contextlib import asynccontextmanager

from fastapi import FastAPI
import os
import mlflow
import numpy as np
from pydantic import BaseModel


class FetalHealthData(BaseModel):
    accelerations: float
    fetal_movement: float
    uterine_contraction: float
    severe_decelerations: float


@asynccontextmanager
async def lifespan(app: FastAPI):
    global super_model
    super_model = load_model()
    yield


app = FastAPI(lifespan=lifespan, title="Super Fetal Health API",
              openapi_tags=[
                  {
                      "name": "Health",
                      "description": "Get app health"
                  },
                  {
                      "name": "Prediction",
                      "description": "Model prediction"
                  }
              ])


def load_model():
    MLFLOW_TRACKING_URI = 'https://dagshub.com/aggioxx/my-first-repo.mlflow'

    #i know this is bad practice, just doing it because is not harmful in this scenario
    os.environ['MLFLOW_TRACKING_USERNAME'] = 'aggioxx'
    os.environ['MLFLOW_TRACKING_PASSWORD'] = '4a10b84d0646dc3da014f67ab1efff6a6f071bdb'

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
    registered_model = client.get_registered_model('fetal_health')

    run_id = registered_model.latest_versions[-1].run_id
    logged_model = f'runs:/{run_id}/model'
    loaded_model = mlflow.pyfunc.load_model(logged_model)

    return loaded_model


@app.get(path="/healthz", tags=['Health'])
def healthz():
    return {"status": "healthy"}


@app.post(path="/predict", tags=['Prediction'])
def predict(req: FetalHealthData):
    global super_model
    sample_data = np.array([
        req.accelerations,
        req.fetal_movement,
        req.uterine_contraction,
        req.severe_decelerations,
    ], dtype=np.float32).reshape(1, -1)

    res = super_model.predict(sample_data)
    return {"prediction": str(res)}
