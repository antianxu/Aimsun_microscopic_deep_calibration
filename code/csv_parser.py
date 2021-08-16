import os
import csv
import fnmatch
import numpy as np

#Define input size and output size:
#INPUT_SIZE = 180
#OUTPUT_SIZE = 9

#INTERVAL_COUNT = 18
#DATA_COUNT = 2

def csv_write(data, indices, path_to_file):
    with open(path_to_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)    
        writer.writerow(indices)
        for row in data:
            writer.writerow(row)

def csv_read_to_dictionary(directory):
    all_files = os.listdir(directory)
    input_file_names = fnmatch.filter(all_files, 'x*.csv')
    output_file_names = fnmatch.filter(all_files, 'y*.csv')

    assert len(input_file_names) == len(output_file_names), "Sizes of input & output don't match"
    
    output_dict = dict()

    for input_file_name in input_file_names:
        output_file_name = get_output_file_name(input_file_name)

        input_file_path  = directory + '/' + input_file_name
        output_file_path = directory + '/' + output_file_name

        #Read output data: append output for every row in x table
        output = []
        with open(output_file_path, newline='') as outputcsvfiles:      
            outputcsvreader = csv.reader(outputcsvfiles)
            for outputrow in outputcsvreader:
                if outputrow[0] == '':
                    output.append(0.0)
                else:
                    output.append(float(outputrow[0]))
        
        output = tuple(output)

        if output in output_dict:   # same y will result in same x values due to static seed set in Aimsun
            continue
        else:
            output_dict[output] = []

        with open(input_file_path, newline='') as inputcsvfiles:
            inputcsvreader = csv.reader(inputcsvfiles)
            next(inputcsvreader)
            for inputrow in inputcsvreader:
                id, interval, _, speed, _, _, flow_per_lane, density_per_lane, \
                    before_onramp, after_onramp, before_offramp, after_offramp = np.array(inputrow, dtype=float)

                output_dict[output].append([id, interval, speed, flow_per_lane, before_onramp, after_onramp, before_offramp, after_offramp])

    return output_dict

def csv_read_structure(directory):
    all_files = os.listdir(directory)
    input_file_names = fnmatch.filter(all_files, 'x*.csv')
    output_file_names = fnmatch.filter(all_files, 'y*.csv')

    assert len(input_file_names) == len(output_file_names), "Sizes of input & output don't match"
    
    x = []
    y = []

    for input_file_name in input_file_names:
        file_id = file_name_to_id(input_file_name)
        output_file_name = id_to_output_file_name(file_id)

        input_file_path  = directory + '/' + input_file_name
        output_file_path = directory + '/' + output_file_name

        #Read output data: append output for every row in x table
        output_list = []
        with open(output_file_path, newline='') as outputcsvfiles:      
            outputcsvreader = csv.reader(outputcsvfiles)
            for outputrow in outputcsvreader:
                if outputrow[0] == '':
                    output_list.append(0.0)
                else:
                    output_list.append(float(outputrow[0]))
        y.append(output_list)

        detectors = []
        values = []
        locations = []

        last_detector_id = None
        with open(input_file_path, newline='') as inputcsvfiles:
            inputcsvreader = csv.reader(inputcsvfiles)
            next(inputcsvreader)
            for inputrow in inputcsvreader:
                detector_id, interval, lane_num, speed, _, _, flow_per_lane, density_per_lane, \
                    before_onramp, after_onramp, before_offramp, after_offramp = np.array(inputrow, dtype=float)
                
                if last_detector_id is None or detector_id != last_detector_id:
                    if detector_id == 961:  # temporary fix: don't include id 961
                        continue
                    # append a list for a detector with [detector id, detector values, location info]
                    detectors.append(detector_id)
                    values.append([])
                    locations.append([lane_num, before_onramp, after_onramp, before_offramp, after_offramp])

                    last_detector_id = detector_id
                values[-1].append([speed, flow_per_lane])
        
        x.append([file_id, detectors, values, locations])

    return x, y

def get_output_file_name(input_file_name):
    id = file_name_to_id(input_file_name)
    return id_to_output_file_name(id)

def file_name_to_id(file_name):
    try:
        return int(file_name.partition(".")[0][1:])
    except:
        assert False, F"Can't get id number from {file_name}."

def id_to_output_file_name(id):
    return 'y' + str(id) + '.csv'