#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
from PIL import Image, ImageOps
import numpy as np
from os import listdir
from os.path import isfile, join
from sklearn.model_selection import train_test_split
from skimage.transform import resize
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn import svm

NEW_SIZE = (256, 256)
CATEGORIES = [("Mer", "+1"), ("Ailleurs", "-1")]

"""
Created on Fri Jan 20 19:07:43 2023

@author: cecile capponi
"""

"""
Computes a representation of an image from the (gif, png, jpg...) file 
representation can be (to extend) 
'HC': color histogram
'PX': tensor of pixels
'GC ': matrix of gray pixels 
other to be defined
input = an image (jpg, png, gif)
output = a new representation of the image
"""    
def raw_image_to_representation(image, representation):
    if (representation == "HC"):
        if (image.mode != "RGB"): 
            image = image.convert("RGB")
        return np.array(image.histogram()).flatten()

    if (representation == "PX"):
        if (image.mode != "RGB"): 
            image = image.convert("RGB")
        return np.array(image).flatten()

    if (representation == "GC"):
        gc = image.convert(mode="L")
        return np.array(gc).flatten()


"""
Returns a data structure embedding train images described according to the 
specified representation and associate each image to its label.
-> Representation can be (to extend) 
'HC': color histogram
'PX': tensor of pixels 
'GC': matrix of gray pixels, g, b =
other to be defined
input = where are the data, which represenation of the data must be produced ? 
output = a structure (dictionnary ? Matrix ? File ?) where the images of the
directory have been transformed and labelled according to the directory they are
stored in.
-- uses function raw_image_to_representation
"""
def load_transform_label_train_data(directory, representation):
    dictionnary = {}
    for category in CATEGORIES:
        path = os.path.join(directory, category[0])
        files = [f for f in listdir(path)]
        for f in files:
            img = Image.open(os.path.join(path, f))
            img = img.resize(NEW_SIZE)
            img = img.convert("RGB")
            img_repr = raw_image_to_representation(img, representation)
            dictionnary[f] = (img_repr, category[1])
            img = ImageOps.flip(img)
            img_repr = raw_image_to_representation(img, representation)
            dictionnary[f + "_flipped"] = (img_repr, category[1])

    
    return dictionnary
    
"""
Returns a data structure embedding test images described according to the 
specified representation.r, g, b = image.split()
-> Representation can be (to extend) 
'HC': color histogram
'PX': tensor of pixels 
'GC': matrix of gray pixels 
other to be defined
input = where are the data, which represenation of the data must be produced ? 
output = a structure (dictionnary ? Matrix ? File ?) where the images of the
directory have been transformed (but not labelled)
-- uses function raw_image_to_representation
"""
def load_transform_test_data(directory, representation):
    dictionnary = {}
    files = [f for f in listdir(os.path.join(directory))]
    for f in files:
        img = Image.open(os.path.join(directory, f))
        img = img.resize(NEW_SIZE)
        img = img.convert("RGB")
        img_data = raw_image_to_representation(img, representation)
        dictionnary[f] = img_data.reshape(1, -1)
    return dictionnary

    
"""
Learn a model (function) from a representation of data, using the algorithm 
and its hyper-parameters described in algo_dico
Here data has been previously transformed to the representation used to learn
the model
input = transformed labelled data, the used learning algo and its hyper-parameters (a dico ?)
output =  a model fit with data
"""
def learn_model_from_data(train_data, algo_dico):
    X = train_data["X"]
    y = train_data["y"]
    clf = svm.SVC()
    clf.set_params(**algo_dico)
    clf.fit(X, y)
    
    return clf



"""
Given one example (representation of an image as used to compute the model),
computes its class according to a previously learned model.
Here data has been previously transformed to the representation used to learn
the model
input = representation of one data, the learned model
output = the label of that one data (+1 or -1)
-- uses the model learned by function learn_model_from_data
"""
def predict_example_label(example, model):
    return model.predict(example)


"""
Computes an array (or list or dico or whatever) that associates a prediction 
to each example (image) of the data, using a previously learned model. 

-- uses the model learned by function learn_model_from_dataload_transform_label_train_data
Here data has been previously transformed to the representation used to learn
the model
input = a structure (dico, matrix, ...) embedding all transformed data to a representation, and a model
output =  a structure that associates a label to each data (image) of the input sample
"""
def predict_sample_label(data, model):
    predictions = {}
    for filename, representation in data.items():
        label = predict_example_label(representation, model)
        predictions[filename] = label
    return predictions


"""
Save the predictions on data to a text file with syntax:
filename <space> label (either -1 or 1)  
NO ACCENT  
Here data has been previously transformed to the representation used to learn
the model
input = where to save the predictions, structure embedding the data, the model used
for predictions
output =  OK if the file has been saved, not OK if not
"""
def write_predictions(directory, filename, data, model):
    if (not os.path.exists(directory)):
        print(directory + "/ does not exsit")
        return

    path = os.path.join(directory, filename)
    with open(path, "w") as f:
        predictions = predict_sample_label(data, model)
        f.write("the_data_wizards\n")
        for filename, label in predictions.items():
            f.write(filename + " " + str(label[0]) + "\n")
        print("OK")


"""
Estimates the accuracy of a previously learned model using train data, 
either through CV or mean hold-out, with k folds.
Here data has been previously transformed to the representation used to learn
the model
input = the train labelled data as previously structured, the learned model, and
the number of split to be used either in a hold-out or by cross-validation  
output =  The score of success (betwwen 0 and 1, the higher the better, scores under 0.5
are worst than random
"""
def estimate_model_score(train_data, model, k):
    return cross_val_score(model, train_data["X"], train_data["y"], cv=k, verbose=2)

    
