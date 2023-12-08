import numpy as np
import pandas as pd


def divide_data():
    """
    GOAL: Avoir 4 fichier de données à partir du gros fichier global:
        Train :
            - input.csv
            - output.csv
        Test :
            - input.csv
            - ouput.csv
    """

    # 1. Charger le fichier CSV
    data_frame = pd.read_csv('data/week_approach_maskedID_timeseries.csv')

    # 2. Mélanger les lignes (car les injury = 1 sont seulement à la fin du fichier), ce qui permet d'avoir des injury=1 dans notre train_input
    data_frame = data_frame.sample(frac=1).reset_index(drop=True)

    # 3. Calculer la moitié du nombre total de lignes
    half_file = len(data_frame) // 2

    # 4. Choix de colonnes à garder
    # -- Inputs
    columns_inputs = ['nr. sessions',
                      'nr. rest days',
                      'total kms',
                      'max km one day',
                      'nr. tough sessions (effort in Z5, T1 or T2)',
                      'nr. strength trainings']
    # -- Output
    column_output = ['injury']

    # -- All Shuffle
    columns_all = ['nr. sessions',
                      'nr. rest days',
                      'total kms',
                      'max km one day',
                      'nr. tough sessions (effort in Z5, T1 or T2)',
                      'nr. strength trainings',
                      'injury']

    # 5. Sélectionner uniquement les colonnes à garder
    data_frame_input = data_frame[columns_inputs]
    data_frame_output = data_frame[column_output]

    data_frame_shuffled = data_frame[columns_all]


    # 6. Diviser les DataFrame en deux parties (train and test)
    train_input_dataset = data_frame_input.iloc[:half_file]
    train_output_dataset = data_frame_output.iloc[:half_file]

    test_input_dataset = data_frame_input.iloc[half_file:]
    test_output_dataset = data_frame_output.iloc[half_file:]

    # 7. Sauvegarder les deux parties dans de nouveaux fichiers CSV
    train_input_dataset.to_csv('datasets/train/train_input_dataset.csv', index=False)
    train_output_dataset.to_csv('datasets/train/train_output_dataset.csv', index=False)

    test_input_dataset.to_csv('datasets/test/test_input_dataset.csv', index=False)
    test_output_dataset.to_csv('datasets/test/test_output_dataset.csv', index=False)

    data_frame_shuffled.to_csv('data_shuffled.csv', index=False)





def load_data_csv():
    # 1. Charger les différents fichier
    train_input_dataset = pd.read_csv('datasets/train/train_input_dataset.csv')
    train_output_dataset = pd.read_csv('datasets/train/train_output_dataset.csv')

    test_input_dataset = pd.read_csv('datasets/test/test_input_dataset.csv')
    test_output_dataset = pd.read_csv('datasets/test/test_output_dataset.csv')

    # 2. Convertir les dataframe en np array
    X_train = train_input_dataset.values  # your train set features
    y_train = train_output_dataset.values # your train set labels

    X_test = test_input_dataset.values  # your test set features
    y_test = test_output_dataset  # your test set labels

    return X_train, y_train, X_test, y_test


# divide_data()

