import joblib
import numpy as np
import pandas as pd

client = pd.read_csv("../../input/application_test.csv")
client.fillna("NaN", inplace=True)
client["DAYS_BIRTH"] = client["DAYS_BIRTH"].apply(
    lambda x: int(np.abs(x) / 365.25)
)
client["ORGANIZATION_TYPE"] = client["ORGANIZATION_TYPE"].apply(
    lambda x: x.split(":")[0]
)
client["ORGANIZATION_TYPE"] = client["ORGANIZATION_TYPE"].apply(
    lambda x: x.split(" Entity")[0]
)
client["AMT_INCOME_TOTAL"] = client["AMT_INCOME_TOTAL"].apply(
    lambda x: int(x / 73.3670)
)
client["AMT_ANNUITY"] = client["AMT_ANNUITY"].astype("float64")
#client.to_csv("data/client_test.csv", index=False)
joblib.dump(client, "data/client_test.sav")

"""for i, chunk in enumerate(pd.read_csv("preprocessed_test_df.csv",
                          chunksize=10000)):
    #chunk.to_csv("data/split_csv_pandas/chunk{}.csv".format(i), index=False)
    joblib.dump(chunk, "data/split_csv_pandas/chunk{}.sav".format(i))"""