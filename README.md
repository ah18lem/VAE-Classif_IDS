# VAE-Classif_IDS
## Overview 
This project implements the federated learning framework described in the article "A Two-Stage Federated Learning Method for Intrusion Detection in IoMT Using Variational Autoencoders and Server Classification." It provides implementations for both the classical Autoencoder and the Variational Autoencoder (VAE) architectures. The framework facilitates federated learning simulations via the Flower (flwr) platform, leveraging TensorFlow for client-side model training and server-side aggregation.

## Usage
1. Clone this repository to your local machine.

2. Install the required dependencies using the provided `requirements.txt` file:

```bash
pip install -r requirements.txt```


3. Open constantes.py to configure your simulation parameters. Here you can set:

-Number of clients (NUM_CLIENTS).
-Fraction of clients used for training and evaluation (FRACTION_FIT, FRACTION_EVALUATE).
-Minimum clients required for training and evaluation (MIN_FIT_CLIENTS, MIN_EVALUATE_CLIENTS, MIN_AVAILABLE_CLIENTS).
-Choice between Variational Autoencoder (VAE) or classic Autoencoder (VAE=True or False).
-Batch size, learning rate, and other hyperparameters.
-Select the dataset you want to use by uncommenting the corresponding block (BOT_IoT, Wustl-2020, or SCADA) and commenting out the others. These blocks specify dataset paths, class counts, epochs, architecture layers, and other relevant settings.

4. Run the server to start the simulation:

```bash
python server.py

The server manages and coordinates the entire federated learning simulation.






