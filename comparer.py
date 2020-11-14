import os
import csv_parser
import statistics

def output_comparison_result(x_std, y_std, ids, directory):
    newFile = open(os.path.join(directory, "comparison_result.txt"), 'w')

    newFile.write("Comparison result on y:\n")
    newFile.write('    '.join([str(entry) for entry in y_std]) + "\n")

    newFile.write("\n")
    newFile.write("Comparison result on x (speed, flow/lane):\n")
    for id in ids:
        newFile.write(F"id: {id}    ")
        for i in range(csv_parser.INTERVAL_COUNT):
            newFile.write(F"{x_std[(id, i)]}    ")
        newFile.write("\n\n")

    newFile.close()

def compare_output(directory):
    output_dict = csv_parser.csv_read_to_dictionary(directory)

    y_collections = []
    x_collections = {}
    ids = []

    for key in output_dict:
        y = key
        x = output_dict[key]

        for i, entry in enumerate(y, 0):
            if len(y_collections) > i:
                y_collections[i].append(entry)
            else:
                y_collections.append([entry])

        for entry in x:
            id = int(entry[0])
            interval = int(entry[1])
            speed = entry[2]
            flow = entry[3]

            if not id in ids:
                ids.append(id)

            if (id, interval) in x_collections:
                x_collections[(id, interval)][0].append(speed)
                x_collections[(id, interval)][1].append(flow)
            else:
                x_collections[(id, interval)] = [[speed], [flow]]
    
    y_std = [0] * len(y_collections)       
    x_std = {}

    for i in range(len(y_collections)):
        y_std[i] = statistics.pstdev(y_collections[i])
    for key in x_collections:
        x_std[key] = statistics.pstdev(x_collections[key][0]), statistics.pstdev(x_collections[key][1])

    output_comparison_result(x_std, y_std, ids, directory) 


compare_output("/home/tianxu/Desktop/ITSOS/AimSun-Calibration/shared_folder/dataset/original/vary_clearance")