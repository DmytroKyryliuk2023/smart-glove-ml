from pydantic import BaseModel


class GesturePredictionData(BaseModel):
    gestureModelId: str
    rawData: list[list[float]]


class SequencePredictionData(BaseModel):
    gestureModelId: str
    divisionModelId: str = "default"
