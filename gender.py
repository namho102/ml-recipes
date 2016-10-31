import csv
from sklearn import tree
import numpy 

features = []
labels = []

# with open('Howell.csv', 'r') as csvfile:
# 	for row in csv.reader(csvfile):
# 		# height = float(row["height"])
# 		# weight = float(row["weight"])
# 		# age = float(row["age"])
# 		# male = row["male"]
# 		height = float(row[0])
# 		weight = float(row[1])
# 		age = float(row[2])
# 		male = row[3]
# 		features.append([height, weight, age])
# 		labels.append(male)

f = open('dataset.txt')
data = []
for line in f:
	line = line.strip().split()
	for e in line:
		data.append(e)

# print data	
a = numpy.array(data)	
a = numpy.reshape(a, (-1, 4))
# print b

for row in a:
	height = float(row[2])*2.54
	weight = float(row[3])*0.45359237
	age = int(float(row[1])/12)
	features.append([height, weight, age])
	labels.append(row[0])

# print features
clf = tree.DecisionTreeClassifier()
clf = clf.fit(features, labels)

height = raw_input("Height: ")
weight = raw_input("Weight: ")
age = raw_input("Age: ")

results = clf.predict([[height, weight, age]])
print results
# if results[0] == '1': 
# 	print 'Male' 
# else: print 'Female'



	