from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

app = FastAPI()

class History(BaseModel):
    history: List[int]

class PredictionResponse(BaseModel):
    prediction: int

class ErrorResponse(BaseModel):
    code: int
    message: str

@app.post("/predict", response_model=PredictionResponse, responses={422: {"model": ErrorResponse}})
async def predict(history: History):
    if not history.history:
        raise HTTPException(status_code=422, detail="History cannot be empty")

    # Example prediction logic
    prediction = sum(history.history) % 37  # Simple example: sum of history mod 37

    return {"prediction": prediction}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
