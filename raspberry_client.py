import argparse
import flwr as fl
import numpy as np
import constantes
import data_loader
import model

class AutoencoderClient(fl.client.NumPyClient):
    def __init__(self, cid, input_dim, train_data, fake_labels):
        self.cid = cid
        self.input_dim = input_dim
        self.train_data = np.asarray(train_data, dtype=np.float32)
        self.fake_labels = np.asarray(fake_labels, dtype=np.float32)

    def create_net(self):
        # Nouveau modèle
        return model.AutoencoderWithClassifier(
            self.input_dim,
            isServer=False,
            vae=constantes.VAE,
        )

    def get_parameters(self, config):
        # Paramètres initiaux
        return self.create_net().get_parameters()

    def fit(self, parameters, config):
        # Entraînement client
        net = self.create_net()
        net.set_parameters(parameters)

        history = net.train(
            self.train_data,
            self.fake_labels,
            epochs=constantes.EPOCHS_CLIENT,
        )

        loss = -1.0
        if history is not None:
            loss = float(history.history.get("loss", [-1.0])[-1])

        return net.get_parameters(), len(self.train_data), {
            "client_id": int(self.cid),
            "loss": loss,
            "num_examples": int(len(self.train_data)),
        }

    def evaluate(self, parameters, config):
        # Évaluation locale
        net = self.create_net()
        net.set_parameters(parameters)

        _, reconstructed = net.call(self.train_data)
        mse = float(np.mean(np.square(reconstructed - self.train_data)))

        return mse, len(self.train_data), {
            "client_id": int(self.cid),
            "mse": mse,
        }

def main():
    # Arguments client
    parser = argparse.ArgumentParser()
    parser.add_argument("--client_id", type=int, required=True)
    parser.add_argument("--server", type=str, required=True)
    args = parser.parse_args()

    train_data, fake_labels = data_loader.load_client_data(args.client_id)
    input_dim = train_data.shape[1]
    
    print("Client Raspberry")
    print("client_id      :", args.client_id)
    print("serveur        :", args.server)
    print("train_data     :", train_data.shape)
    print("fake_labels    :", fake_labels.shape)
    
    fl.client.start_client(
        server_address=args.server,
        client=AutoencoderClient(
            args.client_id,
            input_dim,
            train_data,
            fake_labels,
        ).to_client(),
    )

if __name__ == "__main__":
    main()
