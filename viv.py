import numpy as np
from sklearn.datasets import load_iris
from sklearn import tree

iris = load_iris()
print iris.feature_names
print iris.target_names
# print iris.data[0]
# print iris.target[0]

test_idx = [0, 50, 100]

# training data
train_target = np.delete(iris.target, test_idx)
train_data = np.delete(iris.data, test_idx, axis = 0)

# testing data 
test_target = iris.target[test_idx]
test_data = iris.data[test_idx]

print train_target
print train_data

clf = tree.DecisionTreeClassifier()
clf = clf.fit(train_data, train_target)

# print test_target
# print test_data
# clf.predict(test_data)

# viv code
from sklearn.externals.six import StringIO
# import pydot
from IPython.display import Image  
import pydotplus
dot_data = StringIO()
tree.export_graphviz(clf,
        out_file=dot_data,
        feature_names=iris.feature_names,
        class_names=iris.target_names,
        filled=True, rounded=True,
        impurity=False)

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
# graph.write_pdf("iris.pdf")

# print graph
# print dir(graph)
graph.write_pdf("iris.pdf") 
# Image(graph.create_png())  

