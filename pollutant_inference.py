import csv # collect data from csv files
import os # dealing with system
import argparse # parsing tool
import matplotlib.pyplot as plt # plot resuts
import numpy as np # scientific computations
import pandas as pd # treat databases
import pickle # save and load data

from keras.preprocessing.sequence import TimeseriesGenerator # create data from time series usable by machine learning models

from keras.models import Sequential # Base of a ML model
from keras.layers import Dense # Dense (or fully connected) layer
from keras.layers import LSTM # Long Short Term Memory recurent network (very useful for time series)
from keras.models import load_model # load model from existing file

from keras.callbacks import ModelCheckpoint # save best model
from keras.callbacks import ReduceLROnPlateau # learning parameters

from sklearn.metrics import mean_squared_error # mean squared error to check the accuracy of the model

from statsmodels.tsa.arima.model import ARIMA # ARIMA model (classic time series model)

class Solution():

    def __init__(self):

        self._parse()
        self._choose_file()
        self._load_data()

    def _parse(self): # Specify the path to the met and stn files,
                    # the city to analyse and the particle to predict(all are strings).

        parser = argparse.ArgumentParser(description='')
        parser.add_argument('--path', type=str, help='path to met and stn files')
        parser.add_argument('--city', type=str, help='name of the city to analyse')
        parser.add_argument('--particle', type=str, help='name of the particle to analyse')

        args = parser.parse_args()
        self.path, self.city, self.particle = args.path, args.city, args.particle

    def _choose_file(self): # load the data paths according to the city

        if self.city == "niigata":
            self.name_met = "met_data/data_met_cams_jpn_niigata_15201220_met.csv"
            self.name_stn = "stn_data/data_stations_jpn_niigata_15201220_pm25_o3.csv"
        elif self.city == "osaka":
            self.name_met = "met_data/data_met_cams_jpn_osaka_27227040_met.csv"
            self.name_stn = "stn_data/data_stations_jpn_osaka_27227040_pm25_o3.csv"
        elif self.city == "tokyo":
            self.name_met = "met_data/data_met_cams_jpn_tokyo_13219010_met.csv"
            self.name_stn = "stn_data/data_stations_jpn_tokyo_13219010_pm25_o3.csv"
        elif self.city == "boston":
            self.name_met = "met_data/data_met_cams_usa_boston_250250042_met.csv"
            self.name_stn = "stn_data/data_stations_usa_boston_250250042_pm25_o3.csv"
        elif self.city == "los_angeles":
            self.name_met = "met_data/data_met_cams_usa_los_angeles_060371103_met.csv"
            self.name_stn = "stn_data/data_stations_usa_los_angeles_060371103_pm25_o3.csv"
        elif self.city == "pheonix":
            self.name_met = "met_data/data_met_cams_usa_pheonix_040139997_met.csv"
            self.name_stn = "stn_data/data_stations_usa_pheonix_040139997_pm25_o3.csv"
        else:
            raise os.error

    def _load_data(self): # load the data into the variables

        self.met = {} # met is a dictionnary that contains all the meteorological data
        self.panda = pd.read_csv(os.path.join(self.path, self.name_met), delimiter=',', index_col=[0])
        with open(os.path.join(self.path, self.name_met)) as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',')
            next(spamreader)
            for row in spamreader:
                row = row[1:] # first column is an irrelevant index
                row[1:] = list(map(float, row[1:]))
                self.met[row[0]] = row[1:] # the key is the date and the value is a list of the meteorological conditions
                                            # [temperature, humidity, precipitation, radiation, wind speed, wind direction]


        o3 = [] # list of doubles : date + concentration of o3
        pm25 = [] # list of doubles : date + concentration of pm25
        with open(os.path.join(self.path, self.name_stn)) as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',')
            next(spamreader) # skip first row that is only a header
            for row in spamreader:
                newrow = []
                row = row[1:5] # skip first column that is only an index
                newrow.append(row[0].split("+00:00")[0]) # split function is used to match the date format
                                                            # with the dates of meteorological data
                newrow.append(float(row[1]))
                if row[3] == "True": # boolean that says if the data is valid
                    if row[2] == "o3":
                        o3.append(newrow)
                    if row[2] == "pm25":
                        pm25.append(newrow)
                else: # if the data is not valid
                    if row[2] == "o3":
                        newrow[1] = o3[-1][1] # we duplicate the last value (best estimation we have)
                        o3.append(newrow)
                    if row[2] == "pm25":
                        newrow[1] = pm25[-1][1]
                        pm25.append(newrow)


        self.pollutants = {"o3": pd.DataFrame(o3, columns=["date", "concentration"]), # Dictionary that contains both o3 and pm25 data
                            "pm25": pd.DataFrame(pm25, columns=["date", "concentration"])}

    def prepare_features(self, train_prop=0.8, output_len=5, n_hours=24, use_features=True): # Prepare the features for the machine learning model

        self.use_features = use_features # boolean asking if the model takes the meteorological data in count or only the particle concentrations
        self.n_hours = n_hours # number of hours to take into account for the prediction (default is 24)
        self.output_len = output_len # number of hours to predict in the future (default is 5)

        self.data = np.array(self.pollutants[self.particle]["concentration"]) # data of the concentration of the particle
        times = np.array(self.pollutants[self.particle]["date"]) # times of the measures
        n = len(self.data)
        self.last_train = int(train_prop*n) # last index of the train set
                                                # train_prop is the proportion of training data

        self.normalizing_ratio = np.max(self.data)
        self.min_data, self.max_data = np.min(self.data), np.max(self.data)
        data = self.data/self.normalizing_ratio # normalizing the data
        # data = (self.data - self.min_data) / (self.max_data - self.min_data)

        outputs = [] # regroups the outputs : the 5 next concentrations at each time
        for i in range(n - output_len):
            output = [data[j] for j in range(i+1, i+output_len+1)]
            outputs.append(output)

        self.output_train = outputs[:self.last_train-output_len] # remove the last 5 predictions that are in the test set
        self.output_test = outputs[self.last_train:]

        if not use_features:
            self.train_features = data[:self.last_train - output_len] # since we predict 5 steps ahead, we can't predict the 5 last steps
            self.test_features = data[self.last_train:n-output_len] # same here
            self.n_features = 1 # only predict with air pollution

        else:
            features = []
            for i in range(n):
                feature = self.met[times[i]] # all the meteorological conditions
                feature.append(data[i]) # append the air pollution
                features.append(feature)

            features = features / np.max(features, axis=0) # normalizing the features

            features = (features - np.min(features, axis=0)) / (np.max(features, axis=0) - np.min(features, axis=0))

            self.train_features = features[:self.last_train-output_len] # split train and tes features
            self.test_features = features[self.last_train:len(features)-output_len]

            self.n_features = len(feature)



    def train(self, load=False, batch_size=16, units=100, epochs=100): # train the ML model
                                                                        # batch size is the number of inputs fed at the same time
                                                                        # units is the number of neurons in the LSTM network

        self.load = load # load is a boolean that specify if we want to load a previous model

        if not load:

            self.generator_train = TimeseriesGenerator(self.train_features, self.output_train, length=self.n_hours, batch_size=batch_size) # generates the couples (input, output) for the time serie
            self.generator_test = TimeseriesGenerator(self.test_features, self.output_test, length=self.n_hours, batch_size=1)              # split between train and test


            model = Sequential() # initialize a neural network
            model.add(LSTM(units, activation='relu', input_shape=(self.n_hours, self.n_features))) # long short term memory layer : efficient for time series
            model.add(Dense(self.output_len)) # converge to the number of expected outputs

            model.compile(optimizer='Adam', loss="mse") # classic optimizer and loss

            model.summary()

            mcp_save = ModelCheckpoint(os.path.join(self.path, "model"), save_best_only=True, monitor='val_loss', mode='min') # save the best model according to validation loss
            reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=15, verbose=1, min_delta=1e-4, mode='min') # reduce learning rate if the loss hasn't decreased for too long

            model.fit(self.generator_train, validation_data = self.generator_test, epochs=epochs, callbacks=[mcp_save, reduce_lr_loss]) # train model

            with open(os.path.join(self.path, "model",'loss_history'), 'wb') as file_pi:
                pickle.dump(model.history.history, file_pi) # save loss history

        self.model = load_model(os.path.join(self.path, "model")) # load best model
        loss_history = pickle.load(open(os.path.join(self.path, "model",'loss_history'), "rb")) # load loss history
        self.train_loss = loss_history['loss']
        self.val_loss = loss_history['val_loss']

    def test(self): # predict the test outputs

        test_predictions = [] # collect predictions

        for i in range(0, len(self.test_features)-self.n_hours, self.output_len): # predict every 5 values

            current_batch = self.test_features[i:self.n_hours+i].reshape((1, self.n_hours, self.n_features)) #reshape to adapt to model

            current_pred = self.model.predict(current_batch)[0]

            test_predictions.extend(current_pred)

        self.test_predictions = test_predictions[:len(self.test_features)-self.n_hours] # can't match the final predictions that go outside the timing range


    def plot_data(self): # plot all the graphs
        plt.plot(self.train_loss, label='train') # train loss
        plt.plot(self.val_loss, label='test') # val loss
        plt.legend()
        plt.show()

        if self.use_features:
            to_plot = np.multiply(self.test_features[self.n_hours:,self.n_features-1], self.max_data-self.min_data) + self.min_data
            # plt.plot(np.multiply(self.test_features[self.n_hours:,self.n_features-1], self.normalizing_ratio), label="test data") # re multiply to have the real order of magnitude
            plt.plot(to_plot, label="test data")
            error = mean_squared_error(self.test_features[self.n_hours:,self.n_features-1],self.test_predictions) # mean squared error between predictions and outputs
        else:
            to_plot = np.multiply(self.test_features[self.n_hours:], self.max_data-self.min_data) + self.min_data
            # plt.plot(np.multiply(self.test_features[self.n_hours:], self.normalizing_ratio), label="test data")
            plt.plot(to_plot, label="test data")
            error = mean_squared_error(self.test_features[self.n_hours:],self.test_predictions)
        predictions_normal_size = np.multiply(self.test_predictions, self.max_data-self.min_data) + self.min_data
        plt.plot(predictions_normal_size, label="predictions")
        plt.legend()
        plt.show()

        print("the mean squared error between the test data set and the predictions is :", error)

    def arima(self): # classic arima model

        self.train_arima = self.data[:self.last_train] # training set does not exclude the 5 last values
        self.test_arima = self.data[self.last_train:]

        self.history = [x for x in self.train_arima] # history used to fit the model
        self.arima_predictions = [] # collects predictions

        for i in range(0, len(self.test_arima), self.output_len):
            model = ARIMA(self.history, order=(5,0,5)) # create the arima model
            model_fit = model.fit() # train the arima model
            output = model_fit.forecast(self.output_len) # predict the next 5 outputs
            self.arima_predictions.extend(output)
            obs = self.test_arima[i:i+self.output_len]
            self.history.extend(obs) # history extends

    def plot_arima(self):

        plt.plot(self.test_arima, label="test data")
        plt.plot(self.arima_predictions, label="arima predictions")
        plt.legend()
        plt.show()

        error = mean_squared_error(self.test_arima/self.normalizing_ratio, self.arima_predictions[:len(self.test_arima)]/self.normalizing_ratio)

        print("For the ARIMA model, the mean squared error between the test data set and the predictions is :", error)

S = Solution()
S.prepare_features(use_features=True)
S.train(load=True)
S.test()
S.plot_data()
# S.arima()
# S.plot_arima()

