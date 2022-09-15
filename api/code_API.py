import joblib
import pandas as pd
import shap
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()
lgbm_model = joblib.load("models/balanced_lgbm_model.sav")
explainer = shap.TreeExplainer(lgbm_model)

class ClientID(BaseModel):
    SK_ID_CURR: int
    threshold: float

@app.post("/predict")
def profile_and_predict(client: ClientID):
    data = client.dict()
    for i in range(0, 5):
        df_chunk = joblib.load("data/split_csv_pandas/chunk{}.sav".format(i))
        #df_chunk = pd.read_csv("data/split_csv_pandas/chunk{}.csv".format(i))
        if data["SK_ID_CURR"] not in list(df_chunk["SK_ID_CURR"]):
            del df_chunk
        else:
            test_df = df_chunk
            del df_chunk
            break
    feats = [f for f in test_df.columns if f not in [
        "TARGET",
        "SK_ID_BUREAU",
        "SK_ID_PREV",
        "index"
    ]]
    test_feats = test_df[feats]
    app_df = test_feats[test_feats["SK_ID_CURR"]==data["SK_ID_CURR"]]
    app_test = app_df.drop(columns="SK_ID_CURR")
    proba = lgbm_model.predict_proba(
        app_test,
        num_iteration=lgbm_model.best_iteration_
    ).item(1)
    if proba >= data["threshold"]:
        score = "B"
    else:
        score = "G"
    return {
        "client_ID": data["SK_ID_CURR"],
        "PROBA": proba,
        "SCORE": score
    }

class ClientID2(BaseModel):
    SK_ID_CURR: int

@app.post("/features")
def client_features(client: ClientID2):
    data = client.dict()
    for i in range(0, 5):
        df_chunk = joblib.load("data/split_csv_pandas/chunk{}.sav".format(i))
        #df_chunk = pd.read_csv("data/split_csv_pandas/chunk{}.csv".format(i))
        if data["SK_ID_CURR"] not in list(df_chunk["SK_ID_CURR"]):
            del df_chunk
        else:
            test_df = df_chunk
            del df_chunk
            break
    feats = [f for f in test_df.columns if f not in [
        "TARGET",
        "SK_ID_BUREAU",
        "SK_ID_PREV",
        "index"
    ]]
    test_feats = test_df[feats]
    app_df = test_feats[test_feats["SK_ID_CURR"]==data["SK_ID_CURR"]]
    app_test = app_df.drop(columns="SK_ID_CURR")
    shap_values = explainer.shap_values(app_test)
    return {
        "explain_value": explainer.expected_value[1],
        "shap_values": shap_values[1].tolist(),
        "app_values": app_test.values.tolist(),
        "features": app_test.columns.tolist()
    }

if __name__ == "__main__":
    uvicorn.run(app)