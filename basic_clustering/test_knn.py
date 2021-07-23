import numpy as np
from knn import data,knn,groups,features

result = knn.knn(data[:100],data[110],features,groups,20)
print("Original :",groups[data[110][-1]]," KNN :",np.argmax(result)," with ",np.max(result)*100,"% probability")