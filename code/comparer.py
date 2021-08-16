import os
import csv_parser
import statistics
import numpy as np
import matplotlib.pyplot as plt

#OUTPUT_INDICES = [0,1,2,3,9]
OUTPUT_INDICES = [0,1,2,3]
OUTPUT_PARAM_NAMES = ['simStep', 'CFAggressivenessMean', 'maxAccelMean', 'normalDecelMean', 'clearance']

def output_comparison_result(x_std, y_std, ids, directory):
    newFile = open(os.path.join(directory, "comparison_result.txt"), 'w')

    newFile.write("Comparison result on y:\n")
    newFile.write('    '.join([str(entry) for entry in y_std]) + "\n")

    newFile.write("\n")
    newFile.write("Comparison result on x (speed, flow/lane):\n")

    interval_len = list(x_std.keys())[-1][1] + 1
    for id in ids:
        newFile.write(F"id: {id}    ")
        for i in range(interval_len):
            newFile.write(F"{x_std[(id, i)]}    ")
        newFile.write("\n\n")

    newFile.close()

def plot_comparison(x_std, x_avg, y_std, y_avg, ids, directory):
    plot_file_name = os.path.join(directory, "std-plot.png")

    detector_len = len(ids)
    interval_len = list(x_std.keys())[-1][1] + 1

    t = np.arange(interval_len)
    speed_std = np.zeros([detector_len, interval_len])
    speed_avg = np.zeros([detector_len, interval_len])
    flow_std = np.zeros([detector_len, interval_len])
    flow_avg = np.zeros([detector_len, interval_len])

    fig = plt.figure(figsize=(17, 10))

    speed_std_ax = fig.add_subplot(2, 2, 1)
    flow_std_ax = fig.add_subplot(2, 2, 2)
    speed_avg_ax = fig.add_subplot(2, 2, 3)
    flow_avg_ax = fig.add_subplot(2, 2, 4)

    for i in range(detector_len):
        for j in range(interval_len):
            speed_std[i, j] = x_std[ids[i], j][0]
            speed_avg[i, j] = x_avg[ids[i], j][0]
            flow_std[i, j] = x_std[ids[i], j][1]
            flow_avg[i, j] = x_avg[ids[i], j][1]
        speed_std_ax.plot(t, speed_std[i], '-o', label=F'detector {ids[i]}')
        speed_avg_ax.plot(t, speed_avg[i], '-o', label=F'detector {ids[i]}')
        flow_std_ax.plot(t, flow_std[i], '-o', label=F'detector {ids[i]}')
        flow_avg_ax.plot(t, flow_avg[i], '-o', label=F'detector {ids[i]}')

    speed_std_ax.set_title("speed std")
    flow_std_ax.set_title("flow std")
    speed_avg_ax.set_title("speed average")
    flow_avg_ax.set_title("flow average")

    speed_std_ax.legend()
    speed_avg_ax.legend()
    flow_std_ax.legend()
    flow_avg_ax.legend()

    fig.suptitle(F'parameters:{OUTPUT_PARAM_NAMES}\nstd:{y_std}\naverage:{y_avg}', x=0.2, fontsize=12, ha='left')
    plt.show()

    fig.savefig(plot_file_name)
    
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
    
    y_std = [0] * len(OUTPUT_INDICES)
    y_avg = [0] * len(OUTPUT_INDICES)

    x_std = {}
    x_avg = {}

    for i, y_index in enumerate(OUTPUT_INDICES):
        y_std[i] = statistics.pstdev(y_collections[y_index])
        y_avg[i] = statistics.mean(y_collections[y_index])
    for key in x_collections:
        x_std[key] = statistics.pstdev(x_collections[key][0]), statistics.pstdev(x_collections[key][1])
        x_avg[key] = statistics.mean(x_collections[key][0]), statistics.mean(x_collections[key][1])

    #output_comparison_result(x_std, y_std, ids, directory) 
    all_speed_std = [std[0] for std in x_std.values()]
    all_flow_std = [std[1] for std in x_std.values()]
    all_speed_mean = [mean[0] for mean in x_avg.values()]
    all_flow_mean = [mean[1] for mean in x_avg.values()]

    speed_std_mean = statistics.mean(all_speed_std)
    flow_std_mean = statistics.mean(all_flow_std)
    speed_mean_mean = statistics.mean(all_speed_mean)
    flow_mean_mean = statistics.mean(all_flow_mean)

    print(F'y std: {y_std}')
    print(F'y mean: {y_avg}')
    print()
    print(F'Mean speed std: {speed_std_mean}')
    print(F'Mean flow std: {flow_std_mean}')
    print(F'Mean speed mean: {speed_mean_mean}')
    print(F'Mean flow mean: {flow_mean_mean}')

    plot_comparison(x_std, x_avg, y_std, y_avg, ids, directory)


compare_output("/home/tianxu/Desktop/ITSOS/AimSun-Calibration/shared_folder/dataset/another4/vary_give_way_time")