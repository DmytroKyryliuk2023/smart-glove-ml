from pydantic import BaseModel


class GesturePredictionData(BaseModel):
    ModelId: str
    rawData: list[list[float]]


class SequencePredictionData(BaseModel):
    gestureModelId: str
    divisionModelId: str = "default"
