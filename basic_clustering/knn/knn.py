import numpy as np

def knn(clustered,new_point,features,groups,k):
    distances = [[0,0] for i in range(len(clustered))]

    for i in range(len(clustered)):
        for j in range(len(features)):
            distances[i][0] += np.power(clustered[i][features[j]]-new_point[features[j]],2)
        distances[i][1]=groups[(clustered[i][-1])]

    distances.sort()

    votes = [0 for i in range(len(groups))]
    for i in range(k):
       votes[distances[i][1]] += 1

    return softmax(votes)


def softmax(a):
    b = np.exp(a)
    b /= np.sum(b)
    return b