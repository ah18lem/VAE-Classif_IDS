import flwr as fl
import numpy as np
import keras
import model
import constantes

# Define the Flower client class
class AutoencoderClient(fl.client.NumPyClient):
    def __init__(self, cid, net, train, labels, test):
        self.cid = cid
        self.net = net
        self.train = train
        self.test = test
        self.labels = labels

    def get_parameters(self, config):
        return self.net.get_parameters()

    def fit(self, parameters, config):
      self.net.set_parameters(parameters)
      self.net.train(self.train,self.labels, epochs=constantes.EPOCHS_CLIENT)
      return self.net.get_parameters(), len(self.train), {}


    def evaluate(self, parameters, config):
        self.net.set_parameters(parameters)
        test_features = self.test
        test_features = np.array(test_features)
        _, reconstructed_data = self.net.call(test_features)
        mse = float(
            np.mean(np.square(reconstructed_data - test_features))
        )  
        num_samples = len(self.test)  
        # Return the evaluation metrics as a tuple
        return mse, num_samples, {"mse": mse}


def get_client_fn(inputdim,dataset, fake_labels,test):

    def create_client(c):
        cid=int(c)
        net = model.AutoencoderWithClassifier(inputdim,isServer=False, vae=constantes.VAE)
        net.model.compile(optimizer=keras.optimizers.Adam(learning_rate=constantes.LEARNING_RATE),)
        tf_client = AutoencoderClient(cid,net,dataset[cid], fake_labels[cid], test)
        return tf_client

    return create_client

