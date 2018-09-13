from sklearn.svm import LinearSVC

class SVC:

    def __init__(self, X_train, X_test, y_train, y_test):
        self.model = LinearSVC()
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        print("Training LinearSCV")
        self.model.fit(self.X_train, self.y_train)
        print("Finished training LinearSVC")

    def test(self):
        print(self.model.score(self.X_test, self.y_test))

    def visualize(self):
        return
