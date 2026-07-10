# VAE-Classif_IDS
## Overview 
This project implements the federated learning framework described in the article "Federated Intrusion Detection in Medical IoT: Client-Side Feature Learning with Variational Autoencoders vs Autoencoders" It provides implementations for both the classical Autoencoder and the Variational Autoencoder (VAE) architectures. The framework facilitates federated learning simulations via the Flower (flwr) platform, leveraging TensorFlow for client-side model training and server-side aggregation.

## Usage
1. Clone this repository to your local machine.

2. Install the required dependencies using the provided `requirements.txt` file:

```bash
pip install -r requirements.txt

```


3. Open **constantes.py** to configure your simulation parameters. Here you can set:

### Configuration Parameters

- **Number of clients:** `NUM_CLIENTS`
- **Fraction of clients used for training and evaluation:** `FRACTION_FIT`, `FRACTION_EVALUATE`
- **Minimum clients required for training and evaluation:** `MIN_FIT_CLIENTS`, `MIN_EVALUATE_CLIENTS`, `MIN_AVAILABLE_CLIENTS`
- **Choice between Variational Autoencoder (VAE) or classic Autoencoder:** `VAE=True` or `False`
- **Batch size, learning rate, and other hyperparameters**
- **Dataset selection:**
  - Uncomment the corresponding dataset block to use:
    - `BOT_IoT`
    - `Wustl-2020`
  - Comment out the other dataset blocks.
  - Each block specifies dataset paths, class counts, epochs, architecture layers, and other relevant settings.

4. Run the server to start the simulation:

```bash
python server.py
```
The server manages and coordinates the entire federated learning simulation.



# Federated Learning on Raspberry Pi

This project runs a real federated learning setup using a PC as the Flower server and Raspberry Pi devices as federated clients.

## PC Server

### Expected data

Place the following files in the `data/` directory:

```text
server_train_data.csv
server_train_labels.csv
global_test_data.csv
global_test_labels.csv
class_names.csv
```

### Required files on the server

The PC server must contain:

```text
server.py
model.py
data_loader.py
constantes.py
requirements.txt
data/
```

### Run the server

From the project directory, run:

```bash
python server.py
```

The server will wait for the Raspberry Pi clients to connect.

---

## Raspberry Pi Clients

Each Raspberry Pi must contain the client code and its own local training data.

### Required files on each Raspberry Pi

Copy the following files to each Raspberry Pi:

```text
raspberry_client.py
model.py
data_loader.py
constantes.py
requirements.txt
data/client_X_train.csv
```

`X` is the client number: `0`, `1`, or `2`.

Example:

```text
Raspberry Pi 1 -> data/client_0_train.csv
Raspberry Pi 2 -> data/client_1_train.csv
Raspberry Pi 3 -> data/client_2_train.csv
```

### Run each client

On each Raspberry Pi, run:

```bash
python3 raspberry_client.py --client_id X --server IP_DU_PC:8080
```

Replace:

```text
X        -> client ID: 0, 1, or 2
IP_DU_PC -> IP address of the PC server
```







