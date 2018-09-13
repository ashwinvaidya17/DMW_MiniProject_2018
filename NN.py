from keras.models import Sequential
from keras.layers import Dense


class NN:

    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.model = Sequential()
        self.model.add(Dense(60, input_shape=(10,), activation='relu'))
        self.model.add(Dense(60, activation='relu'))
        self.model.add(Dense(60, activation='relu'))
        self.model.add(Dense(1, activation='sigmoid'))
        self.model.compile(optimizer='adam', loss='binary_crossentropy',
                           metrics=['accuracy'])
        print("Training neural network")
        self.model.fit(self.X_train, self.y_train,batch_size=10, epochs=100)

    def test(self):
        score = self.model.evaluate(self.X_test, self.y_test, batch_size=10)
        print("Accuracy: {}, Loss: {}".format(score[1], score[0]))