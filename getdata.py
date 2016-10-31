import numpy 

f = open('dataset.txt')
data = []
for line in f:
	line = line.strip().split()
	for e in line:
		data.append(e)

# print data	
a = numpy.array(data)	
b = numpy.reshape(a, (-1, 4))
print b