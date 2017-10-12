import numpy as np
import matplotlib.pyplot as plt

x1 = []
x2 = []
y = []
with open('./hw1data.dat') as f:
	line = f.readline()
	lst = line.split(" ")
	data_size = int(lst[0])
	input_num = int(lst[1])
	output_num = int(lst[2])
	for x in range(data_size):
		lst = f.readline().split("\t")
		x1.append(float(lst[0]))
		x2.append(float(lst[1]))
		y.append(float(lst[2]))

color= ['red' if l == 1 else 'green' for l in y]
plt.scatter(x1, x2, color=color)
plt.show()