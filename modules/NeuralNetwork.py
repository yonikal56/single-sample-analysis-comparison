from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from modules.GLV import GLV


class NeuralNetwork:
    def __init__(self, data):
        self.__model = None
        self.__data = data
        self.train()

    def train(self):
        X = []
        y = []
        # create training data arrays
        cohorts = len(self.__data['models'])
        for n in range(1, cohorts + 1):
            m = len(self.__data['models'][n - 1]["cohort"])
            X.extend(self.__data['models'][n - 1]["cohort"])
            y += [[0] * (n - 1) + [1] + [0] * (cohorts - n)] * m
        # define the keras model
        model = Sequential()
        # add layers inside network
        model.add(Dense(50, input_shape=(GLV.numOfPopulations,), activation='relu'))
        model.add(Dense(100, input_shape=(GLV.numOfPopulations,), activation='relu'))
        model.add(Dense(cohorts, activation='sigmoid'))
        # compile the keras model
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        # fit the keras model on the dataset
        model.fit(X, y, epochs=len(y), batch_size=10, verbose=0)
        # evaluate the keras model and print accuracy
        _, accuracy = model.evaluate(X, y)
        print('Accuracy: %.2f' % (accuracy * 100))
        # save trained model for predictions
        self.__model = model

    def predict(self, samples):
        # prediction for all samples
        all_predicts = self.__model.predict(samples).tolist()
        # return the class with the maximum prediction value
        return [predict.index(max(predict)) for predict in all_predicts]

    def predict_real(self, cohort, samples):
        # prediction for all samples
        all_predicts = self.__model.predict(samples).tolist()
        # return prediction for class 0
        return [predict[0] for predict in all_predicts]

    def __str__(self):
        return "Neural Network"
