import torch
import matplotlib.pyplot as plt

# Load the saved .pt file
checkpoint = torch.load("./results/ratio_seed42_l0.01_epoch200_ratio0.8_datasetCIFAR-10_modelresnet20_samplings1/resnet20.pt")

print(checkpoint.keys())  # Check available keys in the checkpoint