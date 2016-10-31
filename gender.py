import csv
from sklearn import tree
import numpy 

features = []
labels = []

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
	height = float("{0:.2f}".format(float(row[2])*2.54)) 
	weight = float("{0:.2f}".format(float(row[3])*0.45359237))
	age = int(float(row[1])/12)
	features.append([height, weight, age])
	labels.append(row[0])

# print features

clf = tree.DecisionTreeClassifier()
clf = clf.fit(features, labels)

name = raw_input("Name: ")
height = raw_input("Height: ")
weight = raw_input("Weight: ")
age = raw_input("Age: ")

results = clf.predict([[height, weight, age]])
# print results
if results[0] == 'm': 
	print name + ' may be a guy!' 
else: print name + ' may be a girl!'


# viz code
from sklearn.externals.six import StringIO
import pydotplus

# dot_data = tree.export_graphviz(clf, out_file=None) 

dot_data = StringIO()
tree.export_graphviz(clf,
        out_file=dot_data,
        feature_names=['height', 'weight', 'age'],
        class_names=['m', 'f'],
        filled=True, rounded=True,
        impurity=False)

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())

graph.write_pdf("gender.pdf") 	