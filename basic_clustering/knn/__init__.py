import pandas as pd,numpy as np

df = pd.read_csv('knn/iris.csv')
df.dropna()
#features = ["SepalLengthCm","SepalWidthCm","PetalLengthCm","PetalWidthCm"]
#df.groupby('Species')

groups = {"Iris-virginica":0,"Iris-versicolor":1,"Iris-setosa":2}
features = [1,2,3,4]
data = np.array(df)
np.random.shuffle(data)