import pandas as pd
from fastapi import FastAPI
import joblib

app = FastAPI()


# define a root `/` endpoint
@app.get("/")
def index():
    return {"ok": True}


# Implement a /predict endpoint
@app.get("/predict/")
def predict(acousticness,
        danceability,
        duration_ms,
        energy,
        explicit,
        id,
        instrumentalness,
        key,   
        liveness,
        loudness,
        mode,
        name,
        release_date,
        speechiness,
        tempo,   
        valence,
        artist):

    # build X ⚠️ beware to the order of the parameters ⚠️
    X = pd.DataFrame(dict(
        acousticness=[float(acousticness)],
        danceability=[float(danceability)],
        duration_ms=[int(duration_ms)],
        energy=[float(energy)],
        explicit=[int(explicit)],
        id=id,
        instrumentalness=[float(instrumentalness)],
        key=[int(key)],   
        liveness=[float(liveness)],
        loudness=[float(loudness)],
        mode=[int(mode)],
        name=name,
        release_date=release_date,
        speechiness=[float(speechiness)],
        tempo=[float(tempo)],   
        valence=[float(valence)],
        artist=artist))

    """
        the pipeline expects to be trained with a DataFrame containing
        the following data types in that order
        ```
        acousticness        float64
        danceability        float64
        duration_ms           int64
        energy              float64
        explicit              int64
        id                   object
        instrumentalness    float64
        key                   int64
        liveness            float64
        loudness            float64
        mode                  int64
        name                 object
        release_date         object
        speechiness         float64
        tempo               float64
        valence             float64
        artist               object
        ```
        """


    # pipeline = get_model_from_joblib
    pipeline = joblib.load('model.joblib')

    # make prediction
    results = pipeline.predict(X)

    # convert response from numpy to python type
    pred = float(round(results[0],2))

    return dict(
        artist=artist,
        name=name,
        popularity_predicted=pred)
