import joblib
import pandas as pd

for i, chunk in enumerate(pd.read_csv("preprocessed_test_df.csv",
                          chunksize=10000)):
    joblib.dump(chunk, "data/split_csv_pandas/chunk{}.sav".format(i))
