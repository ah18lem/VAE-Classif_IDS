from pathlib import Path

# Dossiers projet
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
MODELS_DIR = BASE_DIR / "models"

for directory in [DATA_DIR, RESULTS_DIR, MODELS_DIR]:
    directory.mkdir(exist_ok=True)

# Paramètres FD
SEED = 42
NUM_CLIENTS = 3
SERVER_ADDRESS = "0.0.0.0:8080"

# Stratégie Flower
FRACTION_FIT = 1.0
FRACTION_EVALUATE = 0.0
MIN_FIT_CLIENTS = 3
MIN_EVALUATE_CLIENTS = 0
MIN_AVAILABLE_CLIENTS = 3

# Hyperparamètres
VAE = True
BATCH_SIZE = 64
LEARNING_RATE = 0.0001
NUM_CLASSES = 5
EPOCHS_CLIENT = 5
EPOCHS_SERVEUR = 20
NUM_ROUNDS = 30

# Architecture modèle
ENCODER_LAYERS = [14, 14, 10]
DECODER_LAYERS = [14, 14]

# Données serveur
SERVER_TRAIN_DATA = DATA_DIR / "server_train_data.csv"
SERVER_TRAIN_LABELS = DATA_DIR / "server_train_labels.csv"
GLOBAL_TEST_DATA = DATA_DIR / "global_test_data.csv"
GLOBAL_TEST_LABELS = DATA_DIR / "global_test_labels.csv"
CLASS_NAMES_FILE = DATA_DIR / "class_names.csv"

# Données clients
CLIENT_DATA_FILES = {
    0: DATA_DIR / "client_0_train.csv",
    1: DATA_DIR / "client_1_train.csv",
    2: DATA_DIR / "client_2_train.csv",
}
