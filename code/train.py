import os
import torch
import torch.utils.data as data
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
import numpy as np
import math
import time
BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)))

# import from local files
import csv_parser
from models import GRUModel6_att, output_limits, GRUModel5, GRUModel6

OUTPUT_INDICES = [0,1,2,3,9]
use_cuda = True

class IOStream():
    def __init__(self, path):
        self.f = open(path, 'a')

    def cprint(self, text):
        print(text)
        self.f.write(text+'\n')
        self.f.flush()

    def close(self):
        self.f.close()

def get_data_indices(data_size, batch_size=32, test_percentage=0.2):
    if test_percentage == 0:
        return data_size
    elif test_percentage == 1:
        return 0

    total_batch_num = math.ceil(data_size / batch_size)
    val_batch_num = math.floor(total_batch_num * test_percentage)
    train_batch_num = total_batch_num - val_batch_num

    train_data_end_index = train_batch_num * batch_size - 1

    return int(train_data_end_index)

def get_data_loaders(dataset, batch_size=32, test_percentage=0.2):
    # Load training and validation data.  
    data_size = len(dataset)

    end_training_index = get_data_indices(data_size, batch_size, test_percentage=test_percentage)

    train_loader = data.DataLoader(dataset, batch_size=batch_size, sampler=data.SequentialSampler(range(end_training_index)))
    val_loader   = data.DataLoader(dataset, batch_size=batch_size, sampler=data.SequentialSampler(range(end_training_index+1, data_size)))

    return train_loader, val_loader
  
def evaluate(model, eval_loaders, criterion=nn.MSELoss()):
    model.eval()
    # Evaluate the model by returning the loss
    total_loss = 0.0
    loss_iter = 0
    for val_loader in eval_loaders:
        for detector_ids, values, locations, label in val_loader:
            if use_cuda and torch.cuda.is_available():
                values = values.cuda()
                locations = locations.cuda()
                label = label.cuda()
            output = model(values, locations)
            loss = criterion(output, label)
            total_loss += loss.item()
            loss_iter += 1

    loss = float(total_loss) / loss_iter
    return loss

def train(io, model, train_loaders, eval_loaders, num_epochs=500, learning_rate=0.0005):
    if use_cuda and torch.cuda.is_available():
        model.cuda()
        print('CUDA is available!  Training on GPU ...')
    else:
        print('CUDA is not available.  Training on CPU ...')
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    criterion = nn.MSELoss()

    train_loss = np.zeros(num_epochs)
    val_loss = np.zeros(num_epochs)

    for epoch in range(num_epochs):
        start_time = time.time()
        total_train_loss = 0.0
        iter_per_epoch = 0
        model.train()
        for train_loader in train_loaders:
            for detector_ids, values, locations, train_label in train_loader:
                if use_cuda and torch.cuda.is_available():
                    values = values.cuda()
                    locations = locations.cuda()
                    train_label = train_label.cuda()
                output = model(values, locations)
                loss = criterion(output, train_label)    #all train_labels are the same, becuase they came from the same set of y values
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                total_train_loss += loss.item()
                iter_per_epoch += 1

        train_loss[epoch] = total_train_loss / iter_per_epoch
        step_val_loss = evaluate(model, eval_loaders, criterion)
        val_loss[epoch] = step_val_loss

        io.cprint(("Epoch {}: Train loss: {} | "+
        "Validation loss: {}").format(
            epoch + 1,
            train_loss[epoch],
            step_val_loss))
        
        if epoch == 0 or step_val_loss < min_val_loss:
            min_val_loss = step_val_loss
            min_val_loss_index = epoch
            best_model = model.state_dict()

        end_time = time.time()
        io.cprint(F"Total time for epoch {epoch + 1}: {end_time - start_time}s\n")

    io.cprint(F"\nMinimum validation mse: {min_val_loss:.5f}\nEpoch: {min_val_loss_index + 1}")

    torch.save(best_model, BASE_DIR + F'/../model/{model.name}-hidden_size{model.hidden_size}-version{model.version}-checkpoint', _use_new_zipfile_serialization=False)

    plt.plot(train_loss)
    plt.plot(val_loss)
    plt.title(F'mean squared loss: Indices {OUTPUT_INDICES} \
                \nMinimum validation mse: {min_val_loss:.5f}(Epoch: {min_val_loss_index + 1})')
    plt.ylabel('msl')
    plt.xlabel('epoch')
    plt.legend([F'train', \
                F'test'], loc='upper right')
    plt.savefig(BASE_DIR + F'/../figures/{model.name}-hidden_size{model.hidden_size}-version{model.version}.png')
    
def train_all_QEW_test_all_QEW(io):
    batch_size = 32
    learning_rate = 0.001
    num_epochs = 500
    hidden_size = 100
    use_double_GRU = False
    GRUModel = GRUModel6_att

    QEW_dataset_indicies = [2, 3, 4, 5]

    train_loaders = []
    eval_loaders = []
    for i in QEW_dataset_indicies:
        x, y = csv_parser.csv_read_structure(BASE_DIR + F'/../QEW-dataset/dataset_QEW{i}')

        detectors = torch.tensor([row[1] for row in x]).float()
        values = torch.tensor([row[2] for row in x]).float()
        locations = torch.tensor([row[3] for row in x]).float()

        y = torch.tensor(y)[:, torch.tensor(OUTPUT_INDICES)]  # only take the needed columns from y

        dataset = data.TensorDataset(detectors, values, locations, y)

        train_loader, val_loader = get_data_loaders(dataset, batch_size=batch_size, test_percentage=0.2)

        train_loaders.append(train_loader)
        eval_loaders.append(val_loader)

    model = GRUModel(hidden_size=hidden_size, detector_is_in_order=use_double_GRU) # Assuming all detectors are aligned and in order, else set to False

    checkpoint_path = BASE_DIR + F'/../model/{model.name}-hidden_size{model.hidden_size}-version{model.version}-2345-checkpoint'
    if os.path.exists(checkpoint_path):
        try:
            state = torch.load(checkpoint_path)
            model.load_state_dict(state)
            io.cprint(F"Taken model checkpoint from: {checkpoint_path}")
        except:
            io.cprint("Model doesn't match checkpoint, starting from random model weights...")
    else:
        io.cprint("Starting from random model weights...")

    train(io, model, train_loaders, eval_loaders, num_epochs=num_epochs, learning_rate=learning_rate)

def train_partial_QEW_test_leftout_QEW_network(io, leftout_network_index=4):
    transfer_learning_percentage = 0.7
    batch_size = 32
    learning_rate = 0.0005
    num_epochs = 100
    hidden_size = 100
    transfer_learning = True
    use_double_GRU = False
    GRUModel = GRUModel6_att

    if not transfer_learning:
        QEW_dataset_indicies = [2,3,4,5]
    else:
        QEW_dataset_indicies = [leftout_network_index]

    train_loaders = []
    eval_loaders = []
    for i in QEW_dataset_indicies:
        x, y = csv_parser.csv_read_structure(BASE_DIR + F'/../QEW-dataset/dataset_QEW{i}')

        detectors = torch.tensor([row[1] for row in x]).float()
        values = torch.tensor([row[2] for row in x]).float()
        locations = torch.tensor([row[3] for row in x]).float()

        y = torch.tensor(y)[:, torch.tensor(OUTPUT_INDICES)]  # only take the needed columns from y

        dataset = data.TensorDataset(detectors, values, locations, y)

        if not transfer_learning:
            data_loader, _ = get_data_loaders(dataset, batch_size=batch_size, test_percentage=0)

            if i != leftout_network_index:
                train_loaders.append(data_loader)
            else:
                eval_loaders.append(data_loader)
        else:
            train_loader, val_loader = get_data_loaders(dataset, batch_size=batch_size, test_percentage=transfer_learning_percentage)
            train_loaders.append(train_loader)
            eval_loaders.append(val_loader)

    model = GRUModel(hidden_size=hidden_size, detector_is_in_order=use_double_GRU).float() # Assuming all detectors are aligned and in order, else set to False
    checkpoint_path = BASE_DIR + F'/../model/{model.name}-hidden_size{model.hidden_size}-version{model.version}-checkpoint'

    if os.path.exists(checkpoint_path):
        try:
            state = torch.load(checkpoint_path)
            model.load_state_dict(state)
            io.cprint(F"Taken model checkpoint from: {checkpoint_path}")
        except:
            io.cprint("Model doesn't match checkpoint, starting from random model weights...")
    else:
        io.cprint("Starting from random model weights...")

    train(io, model, train_loaders, eval_loaders, num_epochs=num_epochs, learning_rate=learning_rate)

def train_QEW_test_on_small_network(io):
    batch_size = 32
    learning_rate = 0.0005
    num_epochs = 500
    hidden_size = 100
    use_double_GRU = True
    GRUModel = GRUModel6

    train_loaders = []
    eval_loaders = []

    # Get validation data from data2, data3
    x1, y1 = csv_parser.csv_read_structure(BASE_DIR + '/../data2/dataset1')
    x2, y2 = csv_parser.csv_read_structure(BASE_DIR + '/../data3/dataset2')
    x = x1 + x2
    y = y1 + y2
    
    detectors = torch.tensor([row[1] for row in x]).float()
    values = torch.tensor([row[2] for row in x]).float()
    locations = torch.tensor([row[3] for row in x]).float()

    y = torch.tensor(y)[:, torch.tensor(OUTPUT_INDICES)]  # only take the needed columns from y

    dataset = data.TensorDataset(detectors, values, locations, y)
    train_loader, val_loader = get_data_loaders(dataset, batch_size=batch_size, test_percentage=0.2)

    train_loaders.append(train_loader)
    eval_loaders.append(val_loader)

    model = GRUModel(hidden_size=hidden_size, detector_is_in_order=use_double_GRU) # Assuming all detectors are aligned and in order, else set to False

    if os.path.exists(BASE_DIR + F'/../model/{model.name}-hidden_size{model.hidden_size}-version{model.version}-checkpoint'):
        try:
            state = torch.load(BASE_DIR + F'/../model/{model.name}-hidden_size{model.hidden_size}-version{model.version}-checkpoint')
            model.load_state_dict(state)
            io.cprint(F"Taken model checkpoint from: {model.name}-hidden_size{model.hidden_size}-version{model.version}-checkpoint")
        except:
            io.cprint("Model doesn't match checkpoint, starting from random model weights...")
    else:
        io.cprint("Starting from random model weights...")

    train(io, model, train_loaders, eval_loaders, num_epochs=num_epochs, learning_rate=learning_rate)

if __name__ == '__main__':
    fig_dir = os.path.join(BASE_DIR, "../figures")
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
    log_dir = os.path.join(BASE_DIR, "../log")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    io = IOStream(os.path.join(log_dir, 'run-transfer.log'))

    #train_all_QEW_test_all_QEW(io)
    train_QEW_test_on_small_network(io)
    #train_partial_QEW_test_leftout_QEW_network(io, leftout_network_index=5)