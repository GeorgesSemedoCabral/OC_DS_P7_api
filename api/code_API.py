import joblib
import shap
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()
lgbm_model = joblib.load("models/balanced_lgbm_model.sav")
explainer = shap.TreeExplainer(lgbm_model)

# On crée la classe de données qui permettront d'exposer l'API pour la prédiction du score.

class ClientID(BaseModel):
    SK_ID_CURR: int
    threshold: float

# L'API charge les données du client demandé, les prétraite pour le mettre au bon format pour le modèle, et calcule la probabilité du score de prédiction. En fonction du seuil de prédiction demandé, l'API retourne le score "G" ou "B" avec la probabilité de score positif.

@app.post("/predict")
def profile_and_predict(client: ClientID):
    data = client.dict()
    for i in range(0, 5):
        df_chunk = joblib.load("data/split_csv_pandas/chunk{}.sav".format(i))
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

# On crée la classe de données qui permettront d'exposer l'API pour l'interprétation du score de prédiction.

class ClientID2(BaseModel):
    SK_ID_CURR: int

# L'API charge les données du client demandé, les prétraite pour le mettre au bon format pour le modèle, et calcule les valeurs de Shapley des variables. L'API retourne la valeur de référence, les valeurs de Shapley du client demandé, et la matrice des variables passées à la fonction de SHAP. Ces données peuvent notamment servir à générer le graphique SHAP au format JavaScript sur un dashboard.

@app.post("/features")
def client_features(client: ClientID2):
    data = client.dict()
    for i in range(0, 5):
        df_chunk = joblib.load("data/split_csv_pandas/chunk{}.sav".format(i))
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