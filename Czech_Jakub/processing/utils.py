import numpy as np
import pandas as pd
import pickle

def perform_processing(
        input_data: pd.DataFrame
) -> pd.DataFrame:
    # NOTE(MF): sample code
    # preprocessed_data = preprocess_data(input_data)
    # models = load_models()  # or load one model
    # please note, that the predicted data should be a proper pd.DataFrame with column names
    # predicted_data = predict(models, preprocessed_data)
    # return predicted_data

    loaded_model = pickle.load(open("processing/model.sav", 'rb'))
    predicted_data = loaded_model.predict(input_data)
    # column_names = ['ZASOB', 'MENAGER_ID', 'GRUPA_1', 'GRUPA_2', 'DZIAL', 'TYTUL', 'OPIS_RODZINY', 'NAZWA_RODZINY']
    # predicted_data = predicted_data.drop(columns="DECYZJA", axis=1)

    # for the simplest approach generate a random DataFrame with proper column names and size
    # column_names = ['ZASOB', 'MENAGER_ID', 'GRUPA_1', 'GRUPA_2', 'DZIAL', 'TYTUL', 'OPIS_RODZINY', 'NAZWA_RODZINY', 'KOD_STANOWISKA']
    # predicted_data = pd.DataFrame(
    #     np.random.randint(low=0, high=2, size=(len(input_data.index), len(column_names))),
    #     columns=column_names
    # )
    predicted_data = pd.DataFrame(predicted_data, index=False)
    predicted_data.style.hide_index()
    return predicted_data
