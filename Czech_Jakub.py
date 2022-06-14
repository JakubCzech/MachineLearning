import argparse
import pathlib

import pandas as pd
from sklearn.metrics import accuracy_score

from processing.utils import perform_processing


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', type=str)
    parser.add_argument('results_file', type=str)
    args = parser.parse_args()

    input_file = pathlib.Path(args.input_file)
    results_file = pathlib.Path(args.results_file)

    data = pd.read_csv(input_file, sep=',')

    output_column_names = ['DECYZJA']
    input_column_names = ['ZASOB', 'MENAGER_ID', 'GRUPA_1', 'GRUPA_2', 'DZIAL', 'TYTUL', 'OPIS_RODZINY', 'NAZWA_RODZINY', 'KOD_STANOWISKA']
    gt_data = data.drop(columns=input_column_names)
    input_data = data.drop(columns=output_column_names)

    predicted_data = perform_processing(input_data)
    # print(predicted_data.head())
    # sample for calculating accuracy of the prediction
    print(accuracy_score(gt_data, predicted_data))

    predicted_data.to_csv(results_file, sep='\t', encoding='utf-8')


if __name__ == '__main__':
    main()
