# Configurations
import torch
import os
import numpy as np
# Base Configuration class
# Don't use this class directly. Instead, sub-class it and override

class Config():

    def display(self):
        """Display Configuration values"""
        print("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n")


class Option(Config):
    """Configuration for training on Kaggle Data Science Bowl 2018
    Derived from the base Config class and overrides specific values
    """

    file_name = os.path.basename(__file__).split('.')[0]

    batch_size = 12

    workers = 12

    stage_epochs = [20, 10, 10]

    lr = 1e-4

    lr_decay = 5
 
    weight_decay = 1e-4

    kfolds = 5
    
    stage = 0
    
    start_epoch = 0
    
    total_epochs = sum(stage_epochs)

    print_freq = 1

    val_ratio = 0.12

    evaluate = False

    resume = False