from numpy import float64
from pydantic import BaseModel


class Parameters(BaseModel):
	sepal_length: float64
	sepal_width: float64
	petal_length: float64
	petal_width: float64

