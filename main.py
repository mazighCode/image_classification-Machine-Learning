from architecture import *

import numpy as np
import pickle

algo_dico = {}

data = load_transform_label_train_data("Data/", "PX")

train_data = {}

train_data["X"] = np.asarray([data[filename][0] for filename in data.keys()])
train_data["y"] = np.asarray([data[filename][1] for filename in data.keys()])

print("training...")
model = learn_model_from_data(train_data, algo_dico)

pickle.dump(model, open('model.pkl', 'wb'))

print("done")

# Vous pouvez decommenter le code pour lancer le test de validation croisee
"""scores = estimate_model_score(train_data, model, 5)
score = sum(scores) / len(scores)

print(score)"""

