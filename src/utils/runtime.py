import random
from scipy.stats import truncnorm

RUN_TIME_APP_LIST = ['idle', '1080p', '4k', 'inference', 'detection', 'web']
RUN_TIME_DEGRADE_FACTOR_DIC = {'idle': 1, '1080p': 1.36, '4k': 2.18, 'inference': 1.91, 'detection': 9.9, 'web': 4.32}

NETWORK_LIST = ['DSL', 'cable', '4G', '5G', 'wifi', 'fiber', 'satelite']
# upload speed: Mbps latency: ms
NETWORK_DIC = {'DSL': {'upload speed': (1,20), 'latency': (11,40)},
               'cable': {'upload speed': (1,50), 'latency': (13,27)},
               '4G': {'upload speed': (1,30), 'latency': (30,50)},
               '5G': {'upload speed': (24,58), 'latency': (4,10)},
               'wifi': {'upload speed': (1,50), 'latency': (10,50)},
               'fiber': {'upload speed': (250,1000), 'latency': (10,12)},
               'satelite': {'upload speed': [3,5], 'latency': (594,624)}}

# Ptx: mW
DEVICE_Ptx = (100,500)
SERVER_Ptx = (10000,20000)

# Model size: K
MODEL_SIZE_LeNet5 = 62
MODEL_SIZE_VGG11 = 9774

# LATENCY_SCALING_FACTOR = 40000
# ENERGY_SCALING_FACTOR = 100


def sample_devices(device_random_seed, num_clients, flsys_profile, neural_network, dataset):
    # generate the specific device for each client
    # the device for each client is fixed - fixed random seed per epoch
    training_latency_on_each_device_per_bs_10_list, training_energy_on_each_device_per_bs_10_list \
        = read_standard_device_info(flsys_profile, neural_network, dataset)
    
    num_unique_device = len(training_latency_on_each_device_per_bs_10_list)
    # print("num of unique devices: ", num_unique_device)

    training_latency_on_each_client = []
    training_energy_on_each_client = []

    random.seed(device_random_seed)
    device_id_list = [random.randint(0,num_unique_device-1) for x in range(num_clients)]
    # print("device id list:")
    # print(device_id_list)

    for id in device_id_list:
        training_latency_on_each_client.append(training_latency_on_each_device_per_bs_10_list[id])
        training_energy_on_each_client.append(training_energy_on_each_device_per_bs_10_list[id])
    # print(training_latency_on_each_client)
    # print(training_energy_on_each_client)
    return training_latency_on_each_client, training_energy_on_each_client, device_id_list

def sample_runtime_app(num_clients, rand_seed):
    # generate the specific runtime applications for each client
    # runtime application for each client is dynamic - different random seed per epoch
    num_runtime_status = len(RUN_TIME_APP_LIST)
    # print("num of runtime status: ", num_runtime_status)

    runtime_app_for_each_client_list = []
    degrade_factor_for_each_client_list = []

    random.seed(rand_seed)
    runtime_app_id_list = [random.randint(0,num_runtime_status-1) for x in range(num_clients)]
    # print(runtime_app_id_list)
    for id in runtime_app_id_list:
        runtime_app_for_each_client_list.append(RUN_TIME_APP_LIST[id])
    # print(runtime_app_for_each_client_list)

    for key in runtime_app_for_each_client_list:
        degrade_factor_for_each_client_list.append(RUN_TIME_DEGRADE_FACTOR_DIC[key])
    # print(degrade_factor_for_each_client_list)

    return degrade_factor_for_each_client_list, runtime_app_id_list

def get_truncated_normal(mean=0, sd=1, low=0, upp=10):
    return truncnorm(
        (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)

def sample_runtime_network(num_clients, rand_seed):
    # generate network speed/latency for each client
    # type of network and runtime speed/latency of network for each client is dynamic - different rand seed per epoch
    num_network = len(NETWORK_LIST)
    network_for_each_client_list = []
    network_speed_for_each_client_list = []
    network_latency_for_each_client_list = []
    Ptx_for_each_client_list = []

    random.seed(rand_seed)
    network_id_list = [random.randint(0,num_network-1) for x in range(num_clients)]
    # print(network_id_list)
    for id in network_id_list:
        network_for_each_client_list.append(NETWORK_LIST[id])
    # print(network_for_each_client_list)
    for key in network_for_each_client_list:
        lower_speed = NETWORK_DIC[key]['upload speed'][0]
        upper_speed = NETWORK_DIC[key]['upload speed'][1]
        runtime_speed_gen = get_truncated_normal(mean=(lower_speed+upper_speed)/2, sd=upper_speed/5, low=lower_speed, upp=upper_speed)
        runtime_speed = runtime_speed_gen.rvs(random_state=rand_seed+1)
        network_speed_for_each_client_list.append(runtime_speed)

        lower_latency = NETWORK_DIC[key]['latency'][0]
        upper_latency = NETWORK_DIC[key]['latency'][1]
        runtime_latency_gen = get_truncated_normal(mean=(lower_latency+upper_latency)/2, sd=upper_latency/5, low=lower_latency, upp=upper_latency)
        runtime_latency = runtime_latency_gen.rvs(random_state=rand_seed+2)
        network_latency_for_each_client_list.append(runtime_latency) 

        Ptx_gen = get_truncated_normal(mean=(DEVICE_Ptx[0]+DEVICE_Ptx[1])/2, sd = DEVICE_Ptx[1]/5, low=DEVICE_Ptx[0], upp=DEVICE_Ptx[1])
        Ptx = Ptx_gen.rvs(random_state=rand_seed+3)
        Ptx_for_each_client_list.append(Ptx)
    # print(network_speed_for_each_client_list)
    # print(network_latency_for_each_client_list)
    # print(Ptx_for_each_client_list)
    
    return network_speed_for_each_client_list, network_latency_for_each_client_list, Ptx_for_each_client_list, network_id_list

def read_standard_device_info(flsys_profile_info, neural_network_name, dataset_name):

    # read the client information from the device pool
    device_name_list = []
    training_latency_on_each_device_per_bs_10_list = [] #s
    training_energy_on_each_device_per_bs_10_list = [] #mWh

    file = open(flsys_profile_info, 'r')
    while True:

        line = file.readline()

        if not line: break

        line = line.replace('\n', ' ').replace('\t', ' ')
        splitline = line.split(" ")
        splitline = splitline[:-1]

        if splitline[0] == '' or splitline[0] == 'Client': continue

        device_name_list.append(splitline[0])
        if dataset_name == 'mnist' and neural_network_name == 'LeNet5':
            training_latency_on_each_device_per_bs_10_list.append(float(splitline[3]))
            training_energy_on_each_device_per_bs_10_list.append(float(splitline[7]))
        elif dataset_name == 'mnist' and neural_network_name == 'vgg11':
            training_latency_on_each_device_per_bs_10_list.append(float(splitline[4]))
            training_energy_on_each_device_per_bs_10_list.append(float(splitline[8]))
        elif dataset_name == 'cifar' and neural_network_name == 'LeNet5':
            training_latency_on_each_device_per_bs_10_list.append(float(splitline[5]))
            training_energy_on_each_device_per_bs_10_list.append(float(splitline[9]))
        elif dataset_name == 'cifar' and neural_network_name == 'vgg11':
            training_latency_on_each_device_per_bs_10_list.append(float(splitline[6]))
            training_energy_on_each_device_per_bs_10_list.append(float(splitline[10]))
        else:
            raise NotImplementedError("Not a supported benchmark: {}".format(neural_network_name + ' on ' + dataset_name))
        
        # print(splitline)

    file.close()
    # print(device_name_list)
    # print(training_latency_on_each_device_per_bs_10_list)
    # print(training_energy_on_each_device_per_bs_10_list)

    return training_latency_on_each_device_per_bs_10_list, training_energy_on_each_device_per_bs_10_list

def cost_func(mode, T, Etr, Ptx, R, B, L, M, gamma, theta, LATENCY_SCALING_FACTOR, ENERGY_SCALING_FACTOR):
    # cost function definition
    # T: training latency (s)
    # Etr: training energy consumption (mWh)
    # Ptx: power consumption of data transmit on device (mW)
    # R: degrade factor
    # B: network bandwidth/speed (Mbps)
    # L: network latency (ms)
    # M: model size (K, number of parameters)
    # gamma, theta: parameters to indicate performance-bias or energy-bias: (gamma, theta) = (1,0) or (0,1) 

    # latency - ms
    
    latency = (R*T*1000 + M*1000*32/1000000/B*1000 + L)
    scaled_latency = latency / LATENCY_SCALING_FACTOR

    # energy - mWh
    energy = (Etr*R + Ptx*(M*1000*32/1000000/B+L/1000)/3600)
    scaled_energy = energy / ENERGY_SCALING_FACTOR
    


    if mode == 'bias':
        cost = gamma * scaled_latency + theta * scaled_energy
    elif mode == 'energy-efficiency':
        cost = scaled_latency * scaled_energy
    else:
        raise NotImplementedError("Not a supported mode: {}".format(mode))
    
    # print(cost)
    return cost

def sys_efficiency_U(mode, gamma, theta, flsys_info, nn, data_set, client_num, dev_rand_seed, runtime_rand_seed):
    training_latency_per_client, training_energy_per_client, device_id_list = sample_devices(dev_rand_seed, client_num, flsys_info, nn, data_set)
    degrade_factor_per_client_list, runtime_app_id_list= sample_runtime_app(client_num, runtime_rand_seed)
    network_bw_per_client_list, network_latency_per_client_list, Ptx_per_client_list, network_id_list= sample_runtime_network(client_num, runtime_rand_seed+1)
    true_degrade_factor_per_client_list = []

    for degrade_factor in degrade_factor_per_client_list:
        true_degrade_factor_gen = get_truncated_normal(mean = degrade_factor, sd=degrade_factor/2, low=degrade_factor/5, upp=degrade_factor*1.8)
        true_degrade_factor = true_degrade_factor_gen.rvs(random_state=runtime_rand_seed-1)
        true_degrade_factor_per_client_list.append(true_degrade_factor)

    cost_per_client_list = []
    cost_true_per_client_list = []

    if nn == 'LeNet5':
        M = MODEL_SIZE_LeNet5
        LATENCY_SCALING_FACTOR = 60000
        ENERGY_SCALING_FACTOR = 100
    elif nn == 'vgg11':
        M = MODEL_SIZE_VGG11
        LATENCY_SCALING_FACTOR = 30000000
        ENERGY_SCALING_FACTOR = 100000

    else:
        raise NotImplementedError("Not a supported neural network: {}".format(nn))
    
    # print(training_latency_per_client)
    # print(training_energy_per_client)
    # print(device_id_list)
    # print(degrade_factor_per_client_list)
    # print(runtime_app_id_list)
    # print(network_bw_per_client_list)
    # print(network_latency_per_client_list)
    # print(Ptx_per_client_list)
    # print(network_id_list)


    for i in range(client_num):
        T = training_latency_per_client[i]
        Etr = training_energy_per_client[i]
        Ptx = Ptx_per_client_list[i]
        R = degrade_factor_per_client_list[i]
        R_true = true_degrade_factor_per_client_list[i]
        B = network_bw_per_client_list[i]
        L = network_latency_per_client_list[i]

        cost = cost_func(mode, T, Etr, Ptx, R, B, L, M, gamma, theta, LATENCY_SCALING_FACTOR, ENERGY_SCALING_FACTOR)
        # u = 1 / (rho*cost)

        true_cost = cost_func(mode, T, Etr, Ptx, R_true, B, L, M, gamma, theta, LATENCY_SCALING_FACTOR, ENERGY_SCALING_FACTOR)
        # u_true = 1 / (rho*true_cost)

        cost_per_client_list.append(cost)
        cost_true_per_client_list.append(true_cost)

    return cost_per_client_list, cost_true_per_client_list, training_latency_per_client, training_energy_per_client, device_id_list, degrade_factor_per_client_list, \
            true_degrade_factor_per_client_list, runtime_app_id_list, network_bw_per_client_list, network_latency_per_client_list, Ptx_per_client_list, network_id_list







