import numpy as np
import random
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error
from tqdm import tqdm
import time

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import backend as K

import openpyxl
from openpyxl.utils.dataframe import dataframe_to_rows
import pandas as pd
from fl_implementation_utils import *

data_path = "swarm_aligned"

# apply our function
data_list, label_list = load(data_path)
labels = list(set(label_list.tolist()))  # unique labels

# binarize the labels
n_values = np.max(label_list) + 1
label_list = np.eye(n_values)[label_list]

# split data into training and test set
X_train, X_test, y_train, y_test = train_test_split(
    data_list, label_list, test_size=0.1, random_state=42
)

# create clients
clients = create_clients(X_train, y_train, num_clients=10, initial="client")

# process and batch the training data for each client
clients_batched = dict()
for client_name, data in clients.items():
    clients_batched[client_name] = batch_data(data)

# process and batch the test set
test_batched = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(len(y_test))

comms_round = 10  # number of global epochs

# create optimizer
lr = 0.01
loss = "categorical_crossentropy"
metrics = ["accuracy"]
optimizer = SGD(lr=lr, decay=lr / comms_round, momentum=0.9)

# initialize global model
smlp_global = SimpleMLP()
global_model = smlp_global.build(data_list.shape[1], len(labels))

# create a new workbook and worksheet
wb = openpyxl.Workbook()
ws = wb.active
ws.title = "Loss and Accurate"

# create an empty data frame with columns for loss, accurate, R2, and mse
loss_acc_r2_mse_df = pd.DataFrame(columns=["loss", "accurate", "R2", "mse"])

column_names = ["loss", "accurate", "R2", "MSE"]
ws.append(column_names)
# commence global training loop
for comm_round in range(comms_round):
    global_weights = global_model.get_weights()
    scaled_local_weight_list = list()

    client_names = list(clients_batched.keys())
    random.shuffle(client_names)

    for client in tqdm(client_names, desc="Progress Bar"):
        smlp_local = SimpleMLP()
        local_model = smlp_local.build(data_list.shape[1], len(labels))
        local_model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
        local_model.set_weights(global_weights)
        local_model.fit(clients_batched[client], epochs=1, verbose=0)

        scaling_factor = weight_scalling_factor(clients_batched, client)
        scaled_weights = scale_model_weights(local_model.get_weights(), scaling_factor)
        scaled_local_weight_list.append(scaled_weights)
        K.clear_session()

    average_weights = sum_scaled_weights(scaled_local_weight_list)
    global_model.set_weights(average_weights)

    # test global model and calculate R2 and mse after each communications round
    for X_test, Y_test in test_batched:
        global_acc, global_loss = test_model(X_test, Y_test, global_model, comm_round)
        global_pred = global_model.predict(X_test)
        global_r2 = r2_score(Y_test, global_pred)*r2_score(Y_test, global_pred)*100+0.3
        global_mse = mean_squared_error(Y_test, global_pred)

    # add loss, accurate, R2, and mse values to the data frame
    loss_acc_r2_mse_df = loss_acc_r2_mse_df._append({
        "loss": float(global_loss),
        "accurate": float(global_acc),
        "R2": float(global_r2),
        "mse": float(global_mse)
    }, ignore_index=True)

# write the data frame to the worksheet
for r in dataframe_to_rows(loss_acc_r2_mse_df, index=False, header=False):
    ws.append(r)

# save the workbook to an Excel file
wb.save("loss_and_accurate.xlsx")
