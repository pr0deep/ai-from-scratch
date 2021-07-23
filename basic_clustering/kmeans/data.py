import numpy as np,pandas as pd

def generate_data(POINTS_PER_CLUSTER,CLUSTERS):
    data_x = []
    data_y = []
    for i in range(CLUSTERS):
        noise_y = 2*np.random.rand(POINTS_PER_CLUSTER)
        noise_x = 2*np.random.rand(POINTS_PER_CLUSTER)
        temp_x = 4*i
        temp_y = np.random.randint(-50,50)
        for j in range(POINTS_PER_CLUSTER):
            data_x.append(noise_x[j]+temp_x)
            data_y.append(noise_y[j]+temp_y)
    return data_x,data_y


