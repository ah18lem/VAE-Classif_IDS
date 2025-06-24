import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import umap
import constantes
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import plotly.express as px 







def data_preprocessing(multiclass_target=constantes.MULTICLASS_TARGET_COL):
    train_data = pd.read_csv(constantes.TRAINING_DATA)  
    test_data = pd.read_csv(constantes.TESTING_DATA) 
    train_labels = train_data[multiclass_target]
    test_labels = test_data[multiclass_target]
    train_data = train_data.drop(columns=[multiclass_target])
    test_data = test_data.drop(columns=[multiclass_target])   
    # Combine the training and testing datasets for preprocessing
    combined_data = pd.concat([train_data, test_data], axis=0)
    
    # Drop columns only if DELETE_LIST exists in constantes and is not None
    if hasattr(constantes, 'DELETE_LIST') and constantes.DELETE_LIST:
        combined_data = combined_data.drop(columns=constantes.DELETE_LIST, errors='ignore')

    # Convert 'sport' column if it exists
    if 'sport' in combined_data.columns:
        combined_data['sport'] = combined_data['sport'].apply(lambda x: int(x, 16) if isinstance(x, str) and 'x' in x else int(x))

    # Convert 'dport' column if it exists
    if 'dport' in combined_data.columns:
        combined_data['dport'] = combined_data['dport'].apply(lambda x: int(x, 16) if isinstance(x, str) and 'x' in x else int(x))

    # Identify numerical columns
    numerical_columns = combined_data.select_dtypes(include=[np.number]).columns

    # Initialize the MinMaxScaler
    scaler = MinMaxScaler()

    # Scale the numerical columns 
    combined_data[numerical_columns] = scaler.fit_transform(
        combined_data[numerical_columns]
    )

    # Perform one-hot encoding for categorical columns
    if hasattr(constantes, 'ONE_HOT_ENCODING_LIST') and constantes.ONE_HOT_ENCODING_LIST:
        categorical_columns = constantes.ONE_HOT_ENCODING_LIST
        combined_data = pd.get_dummies(
        combined_data, columns=categorical_columns, dtype=int
        )

    # Split the combined dataset back into training and testing datasets
    train_data = combined_data[: len(train_data)]
    test_data = combined_data[len(train_data) :]

    # Convert the data to float64
    train_data = train_data.astype(np.float64)
    test_data = test_data.astype(np.float64)
    
    print("train:",train_data.shape)
    print("test:",test_data.shape)
    print("trainlabels:",train_labels.shape)
    print("testlabels:",test_labels.shape)
    return train_data,test_data ,train_labels, test_labels,combined_data


def one_hot_encode_column(column):
    column = column.astype(str)
    one_hot_encoded = pd.get_dummies(column, prefix='encoded')
    return one_hot_encoded


def split_balanced_datasets_clients(train_data, labels,nbrClients=10):
    
    balancedDatasets = []  # Create a list to store balanced datasets
    FakeLabels = []  # Create a list to store fake labels
    target=constantes.MULTICLASS_TARGET_COL
    train_data[target] = labels
    unique_classes = sorted(train_data[target].unique())
    class_partition_sizes = {cls: len(train_data[train_data[target] == cls]) // nbrClients for cls in unique_classes}
    
    for i in range(nbrClients):
        client_data = pd.DataFrame()
        fake_labels = []
        for cls in unique_classes:
            partition_size = class_partition_sizes[cls]
            start_idx = i * partition_size
            end_idx = start_idx + partition_size
            client_data_cls = train_data[train_data[target] == cls].iloc[start_idx:end_idx]
            client_data = pd.concat([client_data, client_data_cls], axis=0)
            fake_labels.extend([cls] * len(client_data_cls))
        balancedDatasets.append(client_data.drop(columns=[target]))  
        FakeLabels.append(pd.Series(fake_labels, name=target))
    train_data.drop(columns=[target], inplace=True)  
    return balancedDatasets, FakeLabels



def split_train_server_clients(ratioLabel=constantes.RATIO_LABEL):
    train_data, test_data, train_labels, test_labels ,combined_data= data_preprocessing()
    train_data_sampled, client_data, train_labels_sampled, client_labels = train_test_split(train_data, train_labels, train_size=ratioLabel, stratify=train_labels, random_state=42)
    train_labels_sampled = one_hot_encode_column(train_labels_sampled)
    combined_labels= pd.concat([train_labels, test_labels], axis=0)
    print("Sampled train data shape:", train_data_sampled.shape)
    print("Sampled train labels shape:", train_labels_sampled.shape)
    print("Client data shape:", client_data.shape)
    print("Client labels shape:", client_labels.shape)

    return train_data_sampled, train_labels_sampled, test_data, test_labels, client_data, client_labels,combined_data,combined_labels




def latentSpace_UMAP(mu, test_labels):
    umap_model = umap.UMAP(n_components=2)
    mu_umap = umap_model.fit_transform(mu)
    plt.figure(figsize=(10, 10))
    plt.scatter(mu_umap[:, 0], mu_umap[:, 1], c=test_labels, cmap="brg")
    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")
    plt.colorbar()
    plt.show()


def latentSpace_TSNE(mu, test_labels):   
    tsne_model = TSNE(n_components=2, random_state=42)
    mu_tsne = tsne_model.fit_transform(mu)
    plt.figure(figsize=(10, 10))
    plt.scatter(mu_tsne[:, 0], mu_tsne[:, 1], c=test_labels, cmap="brg")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.colorbar()
    plt.show()



def latentSpace_TSNE_3D_interactive(mu, test_labels, filename):
    tsne_model = TSNE(n_components=3, random_state=42)
    mu_tsne = tsne_model.fit_transform(mu)
    df = pd.DataFrame(mu_tsne, columns=["t-SNE Dimension 1", "t-SNE Dimension 2", "t-SNE Dimension 3"])
    df['Label'] = test_labels
    fig = px.scatter_3d(df, x="t-SNE Dimension 1", y="t-SNE Dimension 2", z="t-SNE Dimension 3", color='Label',
                        color_continuous_scale="rainbow",
                        title="3D t-SNE Visualization")
    fig.update_traces(marker=dict(size=5, opacity=0.8))  
    fig.write_html(filename)
    fig.show()
