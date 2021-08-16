import torch
import torch.nn as nn

mse_loss = nn.MSELoss()

def regularized_MSE(pred_values, true_values, parameters):
    l2_lambda = 0.01

    loss = mse_loss(pred_values, true_values)

    l2_reg = torch.tensor(0.).to(loss.device)
    for param in parameters:
        l2_reg += torch.norm(param)
    loss += l2_lambda * l2_reg

    return loss

def MSE(pred_values, true_values):
    return mse_loss(pred_values, true_values)
