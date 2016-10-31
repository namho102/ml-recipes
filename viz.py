from sklearn import tree
X = [[0, 0], [1, 1]]
Y = [0, 1]
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, Y)
res = clf.predict([[2., 2.]])
print res

from sklearn.externals.six import StringIO
import pydotplus

dot_data = StringIO()
tree.export_graphviz(clf,
        out_file=dot_data,
        feature_names=['x1', 'x2'],
        class_names=['0', '1'],
        filled=True, rounded=True,
        impurity=False)

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())

graph.write_pdf("gender.pdf") 	