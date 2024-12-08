from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Annotated
import uvicorn
from Aicare import RecipeRecommender
aicare = RecipeRecommender()

def GetReceipe(userData):
    # Generate suggested recipes
    recomendation = []
    suggested_recipes = aicare.suggest_recipes(
        category=userData['gender'],
        body_weight=userData['weight'],
        body_height=userData['height'],
        age=userData['age'],
        activity_intensity=userData['activity'],
        objective=userData['objective']
    )
    required_calories = aicare.compute_daily_caloric_intake(
        bmr=aicare.compute_bmr(gender=userData['gender'], body_weight=userData['weight'], body_height=userData['height'], age=userData['age']),
        activity_intensity=userData['activity'],
        objective=userData['objective']
    )
    # Print the suggested recipes in a readable format
    for idx, recipe in suggested_recipes.iterrows():
        recomendation.append(
            {
                "Name": recipe['Name'],
                "Calories": recipe['Calories']
            }
        )
    return recomendation

app = FastAPI()
class BasicInformation(BaseModel):
    gender: str
    weight: int = Field(gt=0, description="Weight must be greater than zero!")
    height: int = Field(gt=0, description="Height must be greater than zero!")
    age: int = Field(gt=0, description="Age must be greater than zero!")
    activity: str
    objective: str

@app.post("/getRecomend")
async def GetRecomend(basicInfo: BasicInformation):
    userInfo = basicInfo.dict()
    GetReceipe(userData=userInfo)
    return GetReceipe(userData=userInfo)

# Custom error handler for validation errors
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={
            "detail": "Validation error occurred",
            "errors": exc.errors(),
            "body": exc.body
        },
    )

# Custom error handler for other unhandled exceptions
@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={
            "detail": "An unexpected error occurred",
            "error": str(exc)
        },
    )

if __name__ == '__main__':
    uvicorn.run("app:app", port=9090, log_level='info', reload=True)
