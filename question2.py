import seaborn as sns
from matplotlib import pyplot as plt

data= sns.load_dataset('iris')
sns.lineplot(x='sepal_length', y='sepal_width', data=data)
plt.title("seaborn graph")
plt.show()


