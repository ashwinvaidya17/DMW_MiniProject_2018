from sklearn.tree import DecisionTreeClassifier

class DecisionTreeModel:

    def __init__(self, X_train, X_test, y_train, y_test):
        self.model = DecisionTreeClassifier()
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        print("Training decision tree")
        self.model.fit(X_train, y_train)
        print("Finished training decision tree")


    def test(self):
        # print accuracy and predictions too
        print(self.model.score(self.X_test, self.y_test))


    def visualize(self):
        return