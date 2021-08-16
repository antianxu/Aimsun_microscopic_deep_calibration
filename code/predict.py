import os
import sys
for SITEPACKAGES in ["C:\\Users\\antianxu\\AppData\\Roaming\\Python\\Python37\\Lib\\site-packages",
					 "C:\\Users\\antianxu\\AppData\\Roaming\\Python\\Python37\\Lib",
					 "C:\\Users\\antianxu\\AppData\\Roaming\\Python\\Python37\\DLLs"]:
	if SITEPACKAGES not in sys.path:
		sys.path.append(SITEPACKAGES)

import csv
import torch
import torch.utils.data as data
import numpy as np
from configparser import ConfigParser, ExtendedInterpolation
import subprocess
import argparse
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# import from local files
import csv_parser
from train import OUTPUT_INDICES, get_data_loaders
from models import GRUModel6, GRUModel6_att
from CnnModel import CNNModel

y_list = ['simStep', 'CFAggressivenessMean', 'maxAccelMean', 'normalDecelMean', 'clearance']
OUTPUT_HEADER = [''] + y_list + ['Avg flow/lane GEH', '% flow GEH < 5', ' % flow GEH < 10', 'Avg speed GEH', '% speed GEH < 5', '% speed GEH < 10']

aimsun_speed_index = 3
aimsun_flow_per_lane_index = 6

AIMSUM_PROGRAM_NAME = 'calibration_data_gen.py'

REP_IDS = [10299791, 10299788, 10299785, 10299782]

def aimsun_simulate_QEW(y_predict, y_list, loader_id):
    assert len(y_predict) == len(y_list), "predicted outputs must match the labels"

    QEW_aimsun_dir = os.path.join(BASE_DIR, 'QEW_aimsun')
    #iniFile = os.path.join(BASE_DIR, 'generate_calibration_data_8.4.ini')
    iniFile = os.path.join(QEW_aimsun_dir, F'QEW_20_{loader_id}.ini')
    id = REP_IDS[loader_id-2]  # Aimsun replication id for QEW, -2 to bring min id from 2 to 0
    index = 0
    datasetDir = os.path.join(QEW_aimsun_dir, 'dataset')
    objectsFile = os.path.join(QEW_aimsun_dir, F'QEW_20_detectors{loader_id}.csv')
    scriptFile = os.path.join(BASE_DIR, 'calibration_data_gen.py')

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

    #cmd.append('-w')    # write to csv

    print(cmd)
    ps = subprocess.Popen(cmd,shell=True,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    stdout, stderr = ps.communicate()
    output_str = stdout.decode("utf-8")
    #print("output:\n")
    #print(stdout)
    #print("output ends\n")
    #exit()
    aimsun_output = output_str.split('\n')
    x_simulated = []
    for row in aimsun_output:
        new_row = [float(n) for n in list(filter(lambda x: x != '', row.split(' ')))]
        if new_row != []:
            x_simulated.append(new_row)

    return x_simulated

def calculate_GEH(sim_value, true_value):
    total_GEH = torch.sqrt(2 * torch.pow(sim_value - true_value, 2) / (sim_value + true_value))

    avg_GEH = (total_GEH.sum() / total_GEH.shape[0]).item()
    percentage_less_than_5 = (total_GEH < 5).sum().item() / total_GEH.shape[0]
    percentage_less_than_10 = (total_GEH < 10).sum().item() / total_GEH.shape[0]

    return avg_GEH, percentage_less_than_5, percentage_less_than_10

def calculate_speed_rms(sim_speed, true_speed):
    assert len(sim_speed) == len(true_speed), F"length of simulated speed ({len(sim_speed)}) should match that of the true value ({len(true_speed)})"

    return torch.sqrt(torch.pow(sim_speed-true_speed, 2).sum() / sim_speed.shape[0]).item()

def Predict_QEW(model, loaders_list, load_ids, path_to_file):
    model.eval()
    with open(path_to_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)    
        writer.writerow(OUTPUT_HEADER)

    sim_x = None
    per_loader_sim_num = 50
    for loader_index, (_, test_loader) in enumerate(loaders_list):
        sim_num = per_loader_sim_num
        if sim_num <= 0:
            break

        loader_id = load_ids[loader_index] # Since loader id starts from 2 in QEW dataset
        for file_id, detector_ids, values, locations, labels in test_loader:
            file_id = file_id.item()

            output = model(values, locations)
            output = output.squeeze(0)
            print(F"Running aimsun with predicted y values in QEW{loader_id} file {file_id}:")
            print(F"Predicted y values: {[y_list[i] + ': ' + str(y.item()) for i, y in enumerate(output)]}")
            print()
            print(F"True y values: {[y_list[i] + ': ' + str(label.item()) for i, label in enumerate(labels[0])]}")
            print()

            detector_count = detector_ids.shape[1]
            interval_count = values.shape[2]

            true_speed = values.squeeze(0)[:, :, 0].view(detector_count * interval_count,)
            true_flow_per_lane = values.squeeze(0)[:, :, 1].view(detector_count * interval_count,)

            sim_x = aimsun_simulate_QEW(output, y_list, loader_id)

            # QEW_aimsun_dir = os.path.join(BASE_DIR, 'QEW_aimsun')
            # datasetDir = os.path.join(QEW_aimsun_dir, 'dataset')
            # file_name = os.path.join(datasetDir, 'x'+str(666)+'.txt')

            # with open(file_name, 'w') as f:
            #     for row in sim_x:
            #         for word in row:
            #             f.write(str(word) + ' ')
            #         f.write('\n')

            print(F"Running comparison...")

            sim_flow_per_lane = torch.tensor([data[aimsun_flow_per_lane_index] for data in sim_x if data != []])
            sim_speed = torch.tensor([data[aimsun_speed_index] for data in sim_x if data != []])

            avg_flow_GEH, flow_GEH_less_5_percentage, flow_GEH_less_10_percentage = calculate_GEH(sim_flow_per_lane, true_flow_per_lane)
            avg_speed_GEH, speed_GEH_less_5_percentage, speed_GEH_less_10_percentage = calculate_GEH(sim_speed, true_speed)

            print("Average flow/lane GEH: ", round(avg_flow_GEH, 5))
            print()            
            #avg_rms = calculate_speed_rms(sim_speed, true_speed)

            with open(path_to_file, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)    
                writer.writerow([F'QEW{loader_id} y{file_id} Prediction'] + [str(round(value.item(), 5)) for value in output] + [round(avg_flow_GEH, 5), round(flow_GEH_less_5_percentage, 3), round(flow_GEH_less_10_percentage, 3), round(avg_speed_GEH, 5), round(speed_GEH_less_5_percentage, 3), round(speed_GEH_less_10_percentage, 3)])
                writer.writerow([F'QEW{loader_id} y{file_id} True'] + [str(round(label.item(), 5)) for label in labels[0]])
                writer.writerow('\n')

            sim_num -= 1
            if sim_num <= 0:
                break
        
        with open(path_to_file, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow('\n')

def aimsun_simulate_simple(y_predict, y_list):
    assert len(y_predict) == len(y_list), "predicted outputs must match the labels"

    QEW_aimsun_dir = os.path.join(BASE_DIR, '..', '..')
    #iniFile = os.path.join(BASE_DIR, 'generate_calibration_data_8.4.ini')
    iniFile = os.path.join(QEW_aimsun_dir, F'generate_calibration_data_8.4.ini')
    id = 890  # Aimsun replication id for simple network
    index = 0
    datasetDir = os.path.join(QEW_aimsun_dir, 'dataset')
    objectsFile = os.path.join(QEW_aimsun_dir, F'list_detectors.csv')
    scriptFile = os.path.join(BASE_DIR, 'calibration_data_gen.py')

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

    #cmd.append('-w')    # write to csv

    print(cmd)
    ps = subprocess.Popen(cmd,shell=True,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    stdout, stderr = ps.communicate()
    output_str = stdout.decode("utf-8")
    aimsun_output = output_str.split('\n')

    # print("output:\n")
    # print(aimsun_output[:3])
    # print("output ends\n")
    # exit()

    x_simulated = []
    for row in aimsun_output[1:]:   # Get rid of the first row in the return message
        new_row = [float(n) for n in list(filter(lambda x: x != '', row.split(' ')))]
        if new_row != []:
            x_simulated.append(new_row)

    return x_simulated

def Predict_simple(model, test_loader, path_to_file):
    model.eval()
    with open(path_to_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)    
        writer.writerow(OUTPUT_HEADER)

    sim_x = None
    sim_num = 50
    for file_id, detector_ids, values, locations, labels in test_loader:
        file_id = file_id.item()

        output = model(values, locations)
        output = output.squeeze(0)
        print(F"Running aimsun with predicted y values in simple network file {file_id}:")
        print(F"Predicted y values: {[y_list[i] + ': ' + str(y.item()) for i, y in enumerate(output)]}")
        print()
        print(F"True y values: {[y_list[i] + ': ' + str(label.item()) for i, label in enumerate(labels[0])]}")
        print()

        detector_count = detector_ids.shape[1]
        interval_count = values.shape[2]

        true_speed = values.squeeze(0)[:, :, 0].view(detector_count * interval_count,)
        true_flow_per_lane = values.squeeze(0)[:, :, 1].view(detector_count * interval_count,)

        sim_x = aimsun_simulate_simple(output, y_list)

        print(F"Running comparison...")

        sim_flow_per_lane = torch.tensor([data[aimsun_flow_per_lane_index] for data in sim_x if data != []])
        sim_speed = torch.tensor([data[aimsun_speed_index] for data in sim_x if data != []])

        avg_flow_GEH, flow_GEH_less_5_percentage, flow_GEH_less_10_percentage = calculate_GEH(sim_flow_per_lane, true_flow_per_lane)
        avg_speed_GEH, speed_GEH_less_5_percentage, speed_GEH_less_10_percentage = calculate_GEH(sim_speed, true_speed)

        print("Average flow/lane GEH: ", round(avg_flow_GEH, 5))
        print()            
        #avg_rms = calculate_speed_rms(sim_speed, true_speed)

        with open(path_to_file, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)    
            writer.writerow([F'y{file_id} Prediction'] + [str(round(value.item(), 5)) for value in output] + [round(avg_flow_GEH, 5), round(flow_GEH_less_5_percentage, 3), round(flow_GEH_less_10_percentage, 3), round(avg_speed_GEH, 5), round(speed_GEH_less_5_percentage, 3), round(speed_GEH_less_10_percentage, 3)])
            writer.writerow([F'y{file_id} True'] + [str(round(label.item(), 5)) for label in labels[0]])
            writer.writerow('\n')

        sim_num -= 1
        if sim_num <= 0:
            break
        
        with open(path_to_file, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow('\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    parser.add_argument('--network_type', type=str, default='simple',
                        help='use simple or QWE network')
    parser.add_argument('--model_type', type=str, default='GRU',
                        help='use GRU or CNN model')
    parser.add_argument('--hidden_size', type=int, default=100,
                        help='hidden size of the model')
    parser.add_argument('--version', type=float, default=1.6,
                        help='model version')
    parser.add_argument('--use_cuda', type=bool, default=False,
                        help='use gpu or not')
    parser.add_argument('--use_LSTM', type=bool, default=False,
                        help='use LSTM or CDAN model')
    parser.add_argument('--exp_description', type=str, default='2345',
                        help='more detailed description of the exp')   
    args = parser.parse_args()

    model_type = args.model_type  # choose which model to use

    # print(args.use_LSTM)
    # print(args.hidden_size)
    # print(args.exp_description)
    # exit()

    save_dir = os.path.join(BASE_DIR, '..', 'output', 'model_prediction')
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    if args.network_type == 'QEW':
        loaders_list = []
        QEW_dataset_indicies = [5]  #[2, 3, 4, 5]

        for i in QEW_dataset_indicies:
            x, y = csv_parser.csv_read_structure(BASE_DIR + F'/../QEW-dataset/dataset_QEW{i}')

            file_ids = torch.tensor([row[0] for row in x])
            detectors = torch.tensor([row[1] for row in x]).float()
            values = torch.tensor([row[2] for row in x]).float()
            locations = torch.tensor([row[3] for row in x]).float()

            y = torch.tensor(y)[:, torch.tensor(OUTPUT_INDICES)]  # only take the needed columns from y

            dataset = data.TensorDataset(file_ids, detectors, values, locations, y)

            train_loader, val_loader = get_data_loaders(dataset, batch_size=1, test_percentage=0.7)

            loaders_list.append([train_loader, val_loader])
    elif args.network_type == 'simple':
        # Get validation data from data2, data3
        x1, y1 = csv_parser.csv_read_structure(BASE_DIR + '/../data2/dataset1')
        x2, y2 = csv_parser.csv_read_structure(BASE_DIR + '/../data3/dataset2')
        x = x1 + x2
        y = y1 + y2
        
        file_ids = torch.tensor([row[0] for row in x])
        detectors = torch.tensor([row[1] for row in x]).float()
        values = torch.tensor([row[2] for row in x]).float()
        locations = torch.tensor([row[3] for row in x]).float()

        y = torch.tensor(y)[:, torch.tensor(OUTPUT_INDICES)]  # only take the needed columns from y

        dataset = data.TensorDataset(file_ids, detectors, values, locations, y)
        _, val_loader = get_data_loaders(dataset, batch_size=1, test_percentage=0.2)

        loader = val_loader
    
    if model_type == 'GRU':
        if args.version == 1.6:
            model = GRUModel6(hidden_size=args.hidden_size, version=args.version, use_cuda=args.use_cuda, detector_is_in_order=args.use_LSTM)
        if args.version == 1.61:
            model = GRUModel6_att(hidden_size=args.hidden_size, version=args.version, use_cuda=args.use_cuda, detector_is_in_order=args.use_LSTM)

        path_to_file = os.path.join(save_dir, F'{model.name}-hidden_size{model.hidden_size}-version{model.version}-{args.exp_description}.csv')
        if os.path.exists(path_to_file):
            print(F'{path_to_file} exist, do not overwrite!')
            exit()

        model_path = BASE_DIR + F'/../model/{model.name}-hidden_size{model.hidden_size}-version{model.version}-{args.exp_description}-checkpoint'

        if torch.cuda.is_available():
            state = torch.load(model_path)
            model.load_state_dict(state)
        else:
            state = torch.load(model_path, map_location=torch.device('cpu'))
            model.load_state_dict(state)
    elif model_type == 'CNN':
        model = CNNModel(use_cuda=False)

        path_to_file = os.path.join(save_dir, F'{model.name}.csv')
        model_path = BASE_DIR + F'/../model/{model.name}-checkpoint'

        if torch.cuda.is_available():
            state = torch.load(model_path)
            model.load_state_dict(state)
        else:
            state = torch.load(model_path, map_location=torch.device('cpu'))
            model.load_state_dict(state)

    if args.network_type == 'QEW':
        Predict_QEW(model, loaders_list, QEW_dataset_indicies, path_to_file=path_to_file)
    elif args.network_type == 'simple':
        Predict_simple(model, loader, path_to_file=path_to_file)








    