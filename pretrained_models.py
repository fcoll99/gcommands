import math
import torch.nn as nn
import torch.nn.functional as F

from model import VGG

PATH = "trained_models/model_1.pt"

def initialize_model(model_name, num_classes, feature_extract, use_pretained = True):

# Each of these variables are model specific.
    model_ft = None
    input_size = 0 # -------> depen del model utilitzat

# Model per donar sentit a la funció d'inicialització
    if model_name == "resnet":
        model_ft = models.resnet18(pretrained = use_pretained)
        set_parameter_requieres_grad(model_ft, feature_extract)

        #Reshape del output del model
        num_fltrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_fltrs, num_classes)
        input_size = 224

    elif model_name == "gcommands":
        model_ft = VGG(*args, **kwargs)
        model.load_state_dict(torch.load(PATH))
        model.eval() #Necessari?
        set_parameter_requieres_grad(model_ft, feature_extract)

        #Reshape del output del model
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        #en aquest cas num_ftrs = 4096
        input_size = 224 #--> sha d'utilitzaar per fer un reshape de la base de dades

    return model_ft, input_size

'''
This helper function sets the .requires_grad attribute of the parameters in the
 model to False when we are feature extracting. By default, when we load a pretrained
model all of the parameters have .requires_grad=True, which is fine if we are training
from scratch or finetuning. However, if we are feature extracting and only want to compute
gradients for the newly initialized layer then we want all of the other parameters to not
require gradients.
'''

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False
