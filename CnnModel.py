import os
import torch
import torch.utils.data as data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score
import numpy as np
import csv_parser
import subprocess
import evaluate
import math
import time

OUTPUT_INDICES = [0,1,2,3,9]

OUTPUT_RANGES = [[0.1, 1.5], [-1, 1], [1, 4], [1, 5], [0.1, 2.0]]

AIMSUM_PROGRAM_NAME = 'calibration_data_gen.py'

VALIDATION_PERCENTAGE = 0.2

DETECTOR_COUNT = 8

use_cuda = True

# Limit outputs as a set of sigmoid activation functions at the output layer.
def output_limits(x, ranges, use_cuda):
    assert x.shape[0] == len(OUTPUT_INDICES), F"Input shape into the limit activation is {np.shape(x)}, \
                                                however the expected shape is ({np.shape(x)[0]}, {len(OUTPUT_INDICES)})"
    if use_cuda and torch.cuda.is_available():
        ranges = ranges.cuda()

    return torch.add(torch.mul(torch.sigmoid(x), (ranges[:, 1] - ranges[:, 0])), ranges[:, 0])

def get_data_indices(data_size, batch_size=1):
    assert data_size % batch_size == 0, F"data size is not a multiple of {batch_size}."

    total_batch_num = data_size / batch_size
    val_batch_num = math.floor(total_batch_num * VALIDATION_PERCENTAGE)
    train_batch_num = total_batch_num - val_batch_num

    train_data_end_index = train_batch_num * batch_size - 1

    return int(train_data_end_index)

def get_data_loaders(dataset, batch_size=1): 
    # Load training and validation data.  
    data_size = len(dataset)

    end_training_index = get_data_indices(data_size, batch_size)

    train_loader = data.DataLoader(dataset, batch_size=batch_size, sampler=data.SequentialSampler(range(end_training_index)))
    val_loader   = data.DataLoader(dataset, batch_size=batch_size, sampler=data.SequentialSampler(range(end_training_index+1, data_size)))

    return train_loader, val_loader

class CNNModel(nn.Module):
    def __init__(self, use_cuda=True):
        super(CNNModel, self).__init__()
        self.name = "CNNModel"
        self.use_cuda = use_cuda
        self.cnn = nn.Sequential(
          nn.Conv1d(in_channels=2, out_channels=2, kernel_size=3, stride=1),
          nn.Conv1d(in_channels=2, out_channels=2, kernel_size=5, stride=1),
        )
        self.detector_fc = nn.Linear(28, 10)
        self.dropout = nn.Dropout(p=0.2)
        self.overall_fc = nn.Linear(10 * DETECTOR_COUNT, len(OUTPUT_INDICES))

    def forward(self, values, locations):
        values = values.squeeze(0)
        locations = locations.squeeze(0)

        values = values.view(-1, 18, 2) # M detectors x N time intervals x 2 values
        values = values.permute(0, 2, 1)
        values = self.cnn(values)
        values = values.view(-1, 2*12)  # 2 channels (speed & flow/lane) * 12 cnn output

        locations = locations.view(-1, 4)

        combined = torch.cat((values, locations), 1)
        output =self.detector_fc(combined)

        output = output.view([1, 10 * DETECTOR_COUNT])
        output = self.overall_fc(output)
        output = output.squeeze(0)

        return output_limits(output, ranges=torch.FloatTensor(OUTPUT_RANGES), use_cuda=self.use_cuda)

def structure3():    
    def evaluate(model, loader, criterion):
        model.eval()
        # Evaluate the model by returning the loss
        total_loss = 0.0
        for i, (detector_ids, values, locations, label) in enumerate(loader, 0):
            if use_cuda and torch.cuda.is_available():
                values = values.cuda()
                locations = locations.cuda()
                label = label.cuda()
            output = model(values, locations)
            feature_count = len(output)
            loss = criterion(output.reshape(1, feature_count), label.reshape(1, feature_count))
            total_loss += loss.item()
        loss = float(total_loss) / (i + 1)
        return loss

    def train(model, dataset, num_epochs=500, batch_size=1, learning_rate=0.0005):
        torch.manual_seed(1)

        if use_cuda and torch.cuda.is_available():
            model.cuda()
            print('CUDA is available!  Training on GPU ...')
        else:
            print('CUDA is not available.  Training on CPU ...')

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
        criterion = nn.MSELoss()

        train_loader, val_loader = get_data_loaders(dataset, batch_size=batch_size)

        train_loss = np.zeros(num_epochs)
        val_loss = np.zeros(num_epochs)

        model.train()
        for epoch in range(num_epochs):
            start_time = time.time()
            total_train_loss = 0.0
            for i, (detector_ids, values, locations, train_label) in enumerate(train_loader, 0):
                if use_cuda and torch.cuda.is_available():
                    values = values.cuda()
                    locations = locations.cuda()
                    train_label = train_label.cuda()
                output = model(values, locations)
                feature_count = len(output)
                #loss = aimsun_loss(output, detector_input, interval_input, value_input)
                loss = criterion(output.reshape(1, feature_count), train_label.reshape(1, feature_count))    #all train_labels are the same, becuase they came from the same set of y values
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                total_train_loss += loss.item()

            train_loss[epoch] = total_train_loss / (i + 1)
            step_val_loss = evaluate(model, val_loader, criterion)
            val_loss[epoch] = step_val_loss

            print(("Epoch {}: Train loss: {} | "+
            "Validation loss: {}").format(
                epoch + 1,
                train_loss[epoch],
                step_val_loss))
            
            if epoch == 0 or step_val_loss < min_val_loss:
                min_val_loss = step_val_loss
                min_val_loss_index = epoch
                best_model = model.state_dict()

            end_time = time.time()
            print(F"Total time for epoch {epoch + 1}: {end_time - start_time}s\n")

        print(F"\nMinimum validation mse: {min_val_loss:.5f}\nEpoch: {min_val_loss_index + 1}")

        torch.save(best_model, cwd + F'/../model/{model.name}-checkpoint')

        plt.plot(train_loss)
        plt.plot(val_loss)
        plt.title(F'mean squared loss: Indices {OUTPUT_INDICES} \
                    \nMinimum validation mse: {min_val_loss:.5f}(Epoch: {min_val_loss_index + 1})')
        plt.ylabel('msl')
        plt.xlabel('epoch')
        plt.legend([F'train', \
                    F'test'], loc='upper right')
        plt.show()

    cwd = os.getcwd()
    x1, y1 = csv_parser.csv_read_structure2(cwd + '/../data2/dataset1')
    x2, y2 = csv_parser.csv_read_structure2(cwd + '/../data3/dataset2')
    x = x1 + x2
    y = y1 + y2
    
    detectors = torch.tensor([row[0] for row in x])
    values = torch.tensor([row[1] for row in x])
    locations = torch.tensor([row[2] for row in x])

    y = torch.tensor(y)[:, torch.tensor(OUTPUT_INDICES)]  # only take the needed columns from y

    dataset = data.TensorDataset(detectors, values, locations, y)

    model = CNNModel()

    if os.path.exists(cwd + F'/../model/{model.name}-checkpoint'):
        try:
            state = torch.load(cwd + F'/../model/{model.name}-checkpoint')
            model.load_state_dict(state)
        except:
            print("Model doesn't match checkpoint")

    train(model, dataset, num_epochs=200, batch_size=1, learning_rate=0.0005)

if __name__ == '__main__':
    structure3()