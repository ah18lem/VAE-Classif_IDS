import os
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import random
from typing import Dict, List, Optional, Tuple

import flwr as fl
import numpy as np
import pandas as pd
import tensorflow as tf
from flwr.common import FitIns, Parameters, Scalar, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from sklearn.metrics import classification_report, confusion_matrix, f1_score

import constantes
import data_loader
import model

def set_seed(seed):
    # Fixer seed
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.keras.utils.set_random_seed(seed)


set_seed(constantes.SEED)
# Données globales
train_data_server, train_labels_server = data_loader.load_server_data()
test_data, test_labels_one_hot = data_loader.load_global_test_data()
class_names = data_loader.load_class_names()

input_dim = train_data_server.shape[1]
test_data_tensor = tf.convert_to_tensor(test_data, dtype=tf.float32)
test_labels_encoded = np.argmax(test_labels_one_hot, axis=1)

class ModifiedFedAvg(fl.server.strategy.FedAvg):
    def aggregate_fit(self, server_round, results, failures):
        # Agrégation poids
        aggregated = super().aggregate_fit(server_round, results, failures)

        if aggregated is None:
            return None

        parameters, metrics = aggregated
        self.latest_parameters = parameters

        return parameters, metrics

    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
    ) -> List[Tuple[ClientProxy, FitIns]]:

        # Entraînement serveur
        net = model.AutoencoderWithClassifier(
            input_dim,
            isServer=True,
            vae=constantes.VAE,
        )

        net.model.set_weights(parameters_to_ndarrays(parameters))

        net.train(
            train_data_server,
            train_labels_server,
            constantes.EPOCHS_SERVEUR,
        )

        fit_ins = FitIns(
            ndarrays_to_parameters(net.get_parameters()),
            {"server_round": server_round},
        )

        size, minimum = self.num_fit_clients(client_manager.num_available())
        clients = client_manager.sample(num_clients=size, min_num_clients=minimum)

        return [(client, fit_ins) for client in clients]

net = model.AutoencoderWithClassifier(input_dim, isServer=True, vae=constantes.VAE)

def evaluate(
    server_round: int,
    parameters: fl.common.NDArrays,
    config: Dict[str, Scalar],
) -> Optional[Tuple[float, Dict[str, Scalar]]]:

    # Évaluation globale
    net.model.set_weights(parameters)
    predictions, _ = net.call(test_data_tensor)

    loss = tf.reduce_mean(
        tf.keras.losses.categorical_crossentropy(
            test_labels_one_hot,
            predictions,
        )
    )

    y_pred = np.argmax(predictions, axis=1)

    macro_f1 = f1_score(
        test_labels_encoded,
        y_pred,
        average="macro",
        zero_division=1,
    )

    report = classification_report(
        test_labels_encoded,
        y_pred,
        target_names=[str(c) for c in class_names],
        zero_division=1,
    )

    cm = confusion_matrix(test_labels_encoded, y_pred)
    cm_df = pd.DataFrame(cm, columns=class_names, index=class_names)

    # Sauvegarde round
    round_model = constantes.MODELS_DIR / (
        f"server_model_round_{server_round:03d}_test_macro_f1_{macro_f1:.4f}.keras"
    )

    net.model.save(round_model)
    cm_df.to_csv(constantes.RESULTS_DIR / f"confusion_matrix_round_{server_round:03d}.csv")

    with open(
        constantes.RESULTS_DIR / f"classification_report_round_{server_round:03d}.txt",
        "w",
        encoding="utf-8",
    ) as f:
        f.write(report)

    print(f"Round {server_round} - Macro-F1: {macro_f1:.4f}")
    print(report)

    return float(loss.numpy()), {"test_macro_f1": float(macro_f1)}

strategy = ModifiedFedAvg(
    fraction_fit=constantes.FRACTION_FIT,
    fraction_evaluate=constantes.FRACTION_EVALUATE,
    min_fit_clients=constantes.MIN_FIT_CLIENTS,
    min_evaluate_clients=constantes.MIN_EVALUATE_CLIENTS,
    min_available_clients=constantes.MIN_AVAILABLE_CLIENTS,
    evaluate_fn=evaluate,
)

print("FD réel Raspberry")
print("Serveur train :", train_data_server.shape)
print("Serveur labels:", train_labels_server.shape)
print("Test global   :", test_data.shape)
print("Clients       :", constantes.NUM_CLIENTS)

fl.server.start_server(
    server_address=constantes.SERVER_ADDRESS,
    config=fl.server.ServerConfig(num_rounds=constantes.NUM_ROUNDS),
    strategy=strategy,
)

if getattr(strategy, "latest_parameters", None) is not None:
    net.model.set_weights(parameters_to_ndarrays(strategy.latest_parameters))

# Modèle final
final_model = constantes.MODELS_DIR / "server_model_final.keras"
net.model.save(final_model)

print(f"Final server model saved to: {final_model}")
