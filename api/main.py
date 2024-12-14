# Importing necessary libraries
import uvicorn
import pickle
from pydantic import BaseModel
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import RedirectResponse
from starlette.staticfiles import StaticFiles




# Initializing the fast API server
app = FastAPI()


origins = [
    "http://localhost",
    "http://localhost:8080",
    "http://0.0.0.0:8000",
    "http://0.0.0.0",
    "http://localhost:3000",
    "http://localhost:3001",
    "http://139.59.255.63",
    "http://139.59.255.63:8000",
    "http://139.59.255.63:8080",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Loading up the trained model
heart_model = pickle.load(open('../model/forest_heart.pkl', 'rb'))
pitch_model = pickle.load(open('../model/model_forest_Pitch_classification.pkl', 'rb'))
roll_model = pickle.load(open('../model/model_forest_Roll_classification.pkl', 'rb'))

# Defining the model input types
class Candidate(BaseModel):
    heartRate: float
    frequencyPitch: float
    frequencyRoll: float
    


@app.post("/prediction/")
async def get_predict(data: Candidate):
    
    
    heartStatus = heart_model.predict([[data.heartRate ]]).tolist()[0]
    pitchStatus = pitch_model.predict([[data.frequencyPitch ]]).tolist()[0]
    rollStatus = roll_model.predict([[data.frequencyRoll ]]).tolist()[0]
    
    if (heartStatus == 1):
        if ((pitchStatus == 0) or (rollStatus == 0)):
            fatigue = 1
        elif ((pitchStatus == 1) or (rollStatus == 1)):
            fatigue = 3
        elif ((pitchStatus == 2) or (rollStatus == 2)):
            fatigue = 3
        elif ((pitchStatus == 3) or (rollStatus == 3)):
            fatigue = 1
    elif (heartStatus == 2):
        if ((pitchStatus == 0) or (rollStatus == 0)):
            fatigue = 2
        elif ((pitchStatus == 1) or (rollStatus == 1)):
            fatigue = 3
        elif ((pitchStatus == 2) or (rollStatus == 2)):
            fatigue = 3
        elif ((pitchStatus == 3) or (rollStatus == 3)):
            fatigue = 1
    elif (heartStatus == 3):
        if ((pitchStatus == 0) or (rollStatus == 0)):
            fatigue = 2
        elif ((pitchStatus == 1) or (rollStatus == 1)):
            fatigue = 3
        elif ((pitchStatus == 2) or (rollStatus == 2)):
            fatigue = 3
        elif ((pitchStatus == 3) or (rollStatus == 3)):
            fatigue = 2
    elif (heartStatus == 4):
        fatigue = 3


    
    
    return {
        "prediction": {
            'fatigue': fatigue,
            "heartStatus": heartStatus,
            "pitchStatus": pitchStatus,
            "rollStatus": rollStatus 
        }
    }



# Setting up the home route
@app.get("/")
def read_root():
    return RedirectResponse(url="/index.html")

app.mount("/", StaticFiles(directory="front/"), name="front")
# Setting up the prediction route

# Configuring the server host and port
if __name__ == '__main__':
    uvicorn.run(app, port=8080, host="0.0.0.0")
