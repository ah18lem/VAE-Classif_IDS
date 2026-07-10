import os
import numpy as np
import pandas as pd
import constantes

def clean_df(df):
    df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed")]
    return df.replace([np.inf, -np.inf], np.nan).fillna(0.0)

def read_x(path):
    # Lecture features
    if not os.path.exists(path):
        raise FileNotFoundError(f"Fichier introuvable : {path}")

    return clean_df(pd.read_csv(path)).astype("float32").to_numpy(dtype=np.float32)

def read_y(path):
    # Lecture labels
    if not os.path.exists(path):
        raise FileNotFoundError(f"Fichier introuvable : {path}")

    df = clean_df(pd.read_csv(path))

    if df.shape[1] == constantes.NUM_CLASSES:
        return df.astype("float32").to_numpy(dtype=np.float32)

    if df.shape[1] == 1:
        y = df.iloc[:, 0].astype(int).to_numpy()
        one_hot = np.zeros((len(y), constantes.NUM_CLASSES), dtype=np.float32)
        one_hot[np.arange(len(y)), y] = 1.0
        return one_hot

    raise ValueError(f"Format labels non reconnu : {path}, shape={df.shape}")

def load_class_names():
    # Noms classes
    if not os.path.exists(constantes.CLASS_NAMES_FILE):
        return [str(i) for i in range(constantes.NUM_CLASSES)]

    df = clean_df(pd.read_csv(constantes.CLASS_NAMES_FILE))
    return df.iloc[:, 0].astype(str).tolist()

def make_fake_labels(n_samples):
    # Faux labels
    y = np.zeros((n_samples, constantes.NUM_CLASSES), dtype=np.float32)
    y[:, 0] = 1.0
    return y

def load_server_data():
    # Données serveur
    return read_x(constantes.SERVER_TRAIN_DATA), read_y(constantes.SERVER_TRAIN_LABELS)

def load_global_test_data():
    # Données test
    return read_x(constantes.GLOBAL_TEST_DATA), read_y(constantes.GLOBAL_TEST_LABELS)

def load_client_data(client_id):
    # Données client
    if client_id not in constantes.CLIENT_DATA_FILES:
        raise ValueError(f"client_id invalide : {client_id}")

    x = read_x(constantes.CLIENT_DATA_FILES[client_id])
    return x, make_fake_labels(len(x))
