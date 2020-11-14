import math

def evaluate(y_pred, y_list):
	'''
	x_true = obs_flow, obs_speed, # obs_flow = {(det, interval):}
	y_pred = [simStep = None,
	CFAggressivenessMean = None,
	maxAccelMean = None,
	normalDecelMean = None,
	aggressiveness = None,
	cooperation = None,
	onRampMergingDistance = None,
	distanceZone1 = None,
	distanceZone2 = None,
	clearance = None]
	'''
	'''
	Run simulation within AIMSUN running using original python.exe
	example:
	python generate_calibration_data.py 
	'C:/Aimsun Projects/Calibration Using Neural Networks/generate_calibration_data.ini' 
	890 10000 
	'C:/Aimsun Projects/Calibration Using Neural Networks/dataset/' 
	'C:/Aimsun Projects/Calibration Using Neural Networks/list_detectors.csv' 
	-l info
	'''
	
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
		cmd.append(str(y_predict[i]))

	ps = subprocess.Popen(cmd,shell=True,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
	stdout, stderr = ps.communicate()

	aimsun_output = stdout.split('\n')[1:]
	
	print(aimsun_output[:4])

	x_simulated = []
	for row in aimsun_output:
		x_simulated.append([float(n) for n in list(filter(lambda x: x != '', row.split(' ')))])
	
	#Islam
	###########################################################
	# Tianxu
	sim_flow, sim_speed = read_csv()
	
	GEH = calculate_GEH(sim_flow, obs_flow) # flow/lane
	
	speed_rms = calculate_speed_rms(sim_speed, obs_speed) # speed
	
	#w1 = 0
	#w2 = 1 - w1
	return w_flow * GEH + w_speed * speed_rms

def calculate_GEH(sim_flow_dict, obs_flow_dict):
    def GEH(sim_flow, obs_flow):
        return math.sqrt(2 * math.pow(sim_flow - obs_flow, 2) / (sim_flow + obs_flow))

    total_GEH = 0
    count = 0

    for key in sim_flow_dict:
        if key in obs_flow_dict:
            total_GEH += GEH(sim_flow_dict[key], obs_flow_dict[key])
            count += 1

    if count == 0:
        assert False, "No matching key between sim_flow and obs_flow dictionaries"

    return total_GEH / count

def calculate_speed_rms(sim_speed_dict, obs_speed_dict):
    squared_sum = 0
    count = 0

    for key in sim_speed_dict:
        if key in obs_speed_dict:
            squared_sum += math.pow((sim_speed_dict[key] - obs_speed_dict[key]), 2)
            count += 1

    if count == 0:
        assert False, "No matching key between sim_speed and obs_speed dictionaries"

    return math.sqrt(squared_sum / count)
	