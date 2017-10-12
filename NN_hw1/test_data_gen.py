import random

f = open("test_data", "w")
f.write("1000 2 1\n")
random.seed(10)
for i in range(1000):
	rd1 = random.randint(-10, 10)
	rd2 = random.randint(-10, 10)
	f.write(str(rd1) + "\t" + str(rd2) + "\t" + str(1 if rd1 + rd2 > 0 else -1) + "\n")
f.close()
