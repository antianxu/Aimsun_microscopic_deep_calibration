import os
import csv
import torch
import torch.utils.data as data
import numpy as np
from configparser import ConfigParser, ExtendedInterpolation
import subprocess

import csv_parser

from RnnModel import GRUModel, OUTPUT_INDICES, get_data_indices
from CnnModel import CNNModel, OUTPUT_INDICES, get_data_indices

y_list = ['simStep', 'CFAggressivenessMean', 'maxAccelMean', 'normalDecelMean', 'clearance']
OUTPUT_HEADER = [''] + y_list + ['Avg flow/lane GEH', '# of flow GEH < 5', '# of flow GEH < 10', 'Avg speed rms', '# of speed GEH < 5', '# of speed GEH < 10']

aimsun_speed_index = 3
aimsun_flow_per_lane_index = 6

def aimsun_simulate(y_predict, y_list):
    assert len(y_predict) == len(y_list), "predicted outputs must match the labels"

    cwd = os.getcwd()
    iniFile = os.path.join(cwd, 'generate_calibration_data_8.4.ini')
    id = 890
    index = 0
    datasetDir = os.path.join(cwd, 'dataset')
    objectsFile = os.path.join(cwd, 'list_detectors.csv')
    scriptFile = os.path.join(cwd, 'calibration_data_gen.py')

    parser = ConfigParser(interpolation=ExtendedInterpolation())
    parser.read(iniFile)
    aimsunExe = os.path.join(parser['Paths']['AIMSUN_DIR'],'aconsole.exe')
	
    # construct and run aconsole command
    cmd = [aimsunExe, '-log_file', 'aimsun.log', '-script']
    cmd.append(scriptFile)
    cmd.append(iniFile)
    cmd.append(str(id))
    cmd.append(datasetDir)
    cmd.append(objectsFile)
    cmd.append('--index')
    cmd.append(str(index))
    cmd.append('-l')
    cmd.append('info')

    for i in range(len(y_list)):
        cmd.append(F'--{y_list[i]}')
        cmd.append(str(y_predict[i].item()))
    ps = subprocess.Popen(cmd,shell=True,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    stdout, stderr = ps.communicate()
    output_str = stdout.decode("utf-8")
    aimsun_output = output_str.split('\n')[1:]
    x_simulated = []
    for row in aimsun_output:
        new_row = [float(n) for n in list(filter(lambda x: x != '', row.split(' ')))]
        if new_row != []:
            x_simulated.append(new_row)

    return x_simulated

def calculate_GEH(sim_value, true_value):
    total_GEH = torch.sqrt(2 * torch.pow(sim_value - true_value, 2) / (sim_value + true_value))

    avg_GEH = (total_GEH.sum() / total_GEH.shape[0]).item()
    count_less_than_5 = (total_GEH < 5).sum().item()
    count_less_than_10 = (total_GEH < 10).sum().item()

    return avg_GEH, count_less_than_5, count_less_than_10

def calculate_speed_rms(sim_speed, true_speed):
    assert len(sim_speed) == len(true_speed), F"length of simulated speed ({len(sim_speed)}) should match that of the true value ({len(true_speed)})"

    return torch.sqrt(torch.pow(sim_speed-true_speed, 2).sum() / sim_speed.shape[0]).item()

def Predict(model, test_loader, path_to_file):
    model.eval()
    with open(path_to_file, 'w') as csvfile:
        writer = csv.writer(csvfile)    
        writer.writerow(OUTPUT_HEADER)

        sim_x = None
        sim_num = 500
        for index, (detector_ids, values, locations, labels) in enumerate(test_loader, 0):
            output = model(values, locations)

            print(F"Running aimsun for predicted y values:")
            print(F"Predicted y values: {[y_list[i] + ': ' + str(y.item()) for i, y in enumerate(output)]}")
            print()
            print(F"True y values: {[y_list[i] + ': ' + str(label.item()) for i, label in enumerate(labels[0])]}")
            print()

            sim_x = aimsun_simulate(output, y_list=y_list)

            print(F"Running comparison...")

            sim_flow_per_lane = torch.tensor([data[aimsun_flow_per_lane_index] for data in sim_x if data != []])
            sim_speed = torch.tensor([data[aimsun_speed_index] for data in sim_x if data != []])

            detector_count = detector_ids.shape[1]
            interval_count = values.shape[2]

            true_speed = values.squeeze(0)[:, :, 0].view(detector_count * interval_count,)
            true_flow_per_lane = values.squeeze(0)[:, :, 1].view(detector_count * interval_count,)

            avg_flow_GEH, flow_GEH_less_5_count, flow_GEH_less_10_count = calculate_GEH(sim_flow_per_lane, true_flow_per_lane)
            avg_speed_GEH, speed_GEH_less_5_count, speed_GEH_less_10_count = calculate_GEH(sim_speed, true_speed)

            print("Average flow/lane GEH: ", avg_flow_GEH)
            print()            
            #avg_rms = calculate_speed_rms(sim_speed, true_speed)

            writer.writerow([F'y{index}-Prediction'] + [str(value.item()) for value in output] + [avg_flow_GEH, flow_GEH_less_5_count, flow_GEH_less_10_count, avg_speed_GEH, speed_GEH_less_5_count, speed_GEH_less_10_count])
            writer.writerow([F'y{index}-True'] + [str(label.item()) for label in labels[0]])
            writer.writerow([])

            sim_num -= 1
            if sim_num <= 0:
                break

if __name__ == '__main__':
    model_type = 'GRU'  # choose which model to use

    cwd = os.getcwd()
    save_dir = os.path.join(cwd, '..', 'output', 'model_prediction')
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

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
    
    data_size = len(dataset)
    train_test_split_index =  get_data_indices(data_size)

    test_loader = data.DataLoader(dataset, batch_size=1, sampler=data.SequentialSampler(range(train_test_split_index+1, data_size)))
    
    if model_type == 'GRU':
        model = GRUModel(hidden_size=20, use_cuda=False)

        path_to_file = os.path.join(save_dir, F'{model.name}-hidden_size{model.hidden_size}-version{model.version}.csv')

        if torch.cuda.is_available():
            state = torch.load(cwd + F'/../model/{model.name}-hidden_size{model.hidden_size}-version{model.version}-checkpoint')
            model.load_state_dict(state)
        else:
            state = torch.load(cwd + F'/../model/{model.name}-hidden_size{model.hidden_size}-version{model.version}-checkpoint', map_location=torch.device('cpu'))
            model.load_state_dict(state)
    elif model_type == 'CNN':
        model = CNNModel(use_cuda=False)

        path_to_file = os.path.join(save_dir, F'{model.name}.csv')

        if torch.cuda.is_available():
            state = torch.load(cwd + F'/../model/{model.name}-checkpoint')
            model.load_state_dict(state)
        else:
            state = torch.load(cwd + F'/../model/{model.name}-checkpoint', map_location=torch.device('cpu'))
            model.load_state_dict(state)

    Predict(model, test_loader, path_to_file=path_to_file)








    