# This code file is used to set the network running device (CPU or GPU).
import torch

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
