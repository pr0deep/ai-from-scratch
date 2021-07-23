import numpy as np

def kmeans(data,k=3):
    #choosing k points
    models = []
    for i in range(4*k):
        cluster_points = []

        for i in range(k):
            x = np.random.randint(0,len(data))
            while x in cluster_points:
                x = np.random.randint(0, len(data))
            cluster_points.append(x)

        #creating result
        category = [0 for i in range(len(data))]
        for j in range(k):
            category[cluster_points[j]] = j

        for i in range(len(data)):
            if i in cluster_points:
                continue
            distances = []
            for j in range(k):
                temp_distance = (np.power(data[i]-data[cluster_points[j]],2))
                distances.append(temp_distance)
            category[i] = category[cluster_points[np.argmin(distances)]]
            models.append(category)

    variances = []
    for i in models:
        variances.append(variance(i,k))
    variances = np.abs(variances)
    choosen = np.argmin(variances)

    return models[choosen]

def variance(a,k):
    counts = [0 for i in range(k)]
    for i in a:
        counts[i] += 1
    return np.std(counts)

