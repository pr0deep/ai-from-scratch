from kmeans.data import generate_data
from kmeans.kmeans import kmeans
import matplotlib.pyplot as plt


NO_OF_CLUSTERS = 5
SIZE_OF_CLUSTER = 100

x,y = generate_data(SIZE_OF_CLUSTER,NO_OF_CLUSTERS)

plt.style.use("seaborn")
fig,ax = plt.subplots()

category = kmeans(y,NO_OF_CLUSTERS)

ax.scatter(x=x,y=y,c = category,cmap='hsv',alpha=0.5)

plt.show()


