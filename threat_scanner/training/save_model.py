import torch

def save_model(model, path):
    torch.save(model.state_dict(), path)
    