import pickle
from architecture import *

model = pickle.load(open("model.pkl", 'rb'))

test = load_transform_test_data("AllTest/", "PX")

print("writing predictions...")
write_predictions("result/", "the_data_wizards.txt", test, model)