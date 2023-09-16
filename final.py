import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import yfinance as yf
from sklearn.metrics import confusion_matrix
from keras import regularizers
import os
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import PolynomialFeatures

if not os.path.exists('result'):
    os.makedirs('result')
import warnings
warnings.filterwarnings('ignore')

class data():
    def __init__(self, stock, datasize):
        self.stock = stock
        self.datasize = datasize
        # obtain the data
        self.raw_data = yf.download(stock)
        # cut off the data to make sure dataset is constant
        cutoff_date = pd.to_datetime('2023-09-01')
        self.raw_data = self.raw_data.loc[self.raw_data.index <= cutoff_date]
        self.plot_hist()
        self.get_Direction()
        self.add_normal_features()
    
    def get_data(self):
        return self.raw_data[-self.datasize:]
    
    # visualize the history
    def plot_hist(self):
        if not os.path.exists(f'result/{self.stock}_result.png'):
            plt.figure()
            plt.plot(self.raw_data[['Close']], color = 'black')
            plt.title(f'{self.stock} history in past {datasize} days')
            plt.xlabel('date')
            plt.ylabel('closing price')
            plt.savefig(f"result/{self.stock}_history.png")
            plt.close()
    
    def get_Direction(self):
        # set up as 1, down as 0
        self.raw_data["Direction"] = (self.raw_data['Close']>=self.raw_data['Close'].shift()).astype('int')

    # more new features here
    def add_normal_features(self):
        # CCI
        TP = (self.raw_data['High'] + self.raw_data['Low'] + self.raw_data['Close']) / 3
        MD = TP.rolling(window=14).apply(lambda x: abs(x - x.mean()).mean())
        self.raw_data['CCI'] = (TP - TP.rolling(window=14).mean()) / (0.015 * MD)
        
        # RA_5 rolling average: stdev in 5 day windows
        self.raw_data['RA_5'] = self.raw_data['Close'].rolling(window=5).std()
        # RA_10 rolling average: stdev in 5 day windows
        self.raw_data['RA_10'] = self.raw_data['Close'].rolling(window=10).std()
        # MACD: Moving Average Convergence Divergence
        short_window = 12
        long_window = 26
        # short window
        short_ema = self.raw_data['Close'].ewm(span=short_window, adjust=False).mean()
        # long window
        long_ema = self.raw_data['Close'].ewm(span=long_window, adjust=False).mean()
        self.raw_data['MACD'] = short_ema - long_ema

        #
        self.raw_data['TR'] = self.raw_data[['High', 'Low', 'Close']].apply(lambda x: max(x) - min(x), axis=1)
        self.raw_data['ATR'] = self.raw_data['TR'].rolling(14).mean()
        # Calculate the Bollinger Bands
        self.raw_data['Middle Band'] = self.raw_data['Close'].rolling(window=20).mean()
        self.raw_data['Upper Band'] = self.raw_data['Middle Band'] + 2 * self.raw_data['Close'].rolling(window=20).std()
        self.raw_data['Lower Band'] = self.raw_data['Middle Band'] - 2 * self.raw_data['Close'].rolling(window=20).std()
        # Calculate the Simple Moving Average (SMA), default window = 10
        self.raw_data['SMA'] = self.raw_data['Close'].rolling(window=10).mean()
        
        #
        self.raw_data["Momentum_1"] = 100*self.raw_data['Close']/self.raw_data['Close'].shift()
        self.raw_data["Momentum_3"] = 100*self.raw_data['Close']/self.raw_data['Close'].shift(3)
        self.raw_data["Momentum_7"] = 100*self.raw_data['Close']/self.raw_data['Close'].shift(7)
        self.raw_data["RateOfChange"] = 100 * (self.raw_data['Close'] - self.raw_data['Close'].shift()) / self.raw_data['Close'].shift()



class model():
    def __init__(self, stock, datasize, hyperparameter):
        self.stock = stock
        self.hyperparameter = hyperparameter
        self.data = data(stock, datasize).get_data()
        self.batchsize, self.epochs, self.unit, self.layer= hyperparameter
        self.timestep = 10
        self.guassian_choice = 1
        self.poly_choice = 0

    def set_guassian(self, choice):
      #1 no guassian; 2 only guassian; 3 both
      self.guassian_choice = choice
    
    def guassian_transform(self, X):
        def gaussian_features(X):
            return np.exp(-0.5 * ((X - X.mean()) / X.std())**2)
        transformer = FunctionTransformer(func=gaussian_features, validate=False)
        self.features_gaussian = transformer.transform(X)

        if self.guassian_choice == 2:
          X = self.features_gaussian
        elif self.guassian_choice == 3:
          X = np.hstack((X, self.features_gaussian))
        return X

    def set_polynomial(self, choice):
      #0 no transform; 2 => 2 degree polynomial
      self.poly_choice = 2

    def polynomial_transform(self, X):
        if self.poly_choice != 0:
            poly = PolynomialFeatures(degree=self.poly_choice)
            X = poly.fit_transform(X)
        return X
    
    def process_data(self):
        self.X = self.data.drop(columns = ['Direction']).values
        self.y = self.data["Direction"].values

        # normalization
        scaler = MinMaxScaler(feature_range = (0,1))
        self.X_normalized = scaler.fit_transform(self.X)

        # guassian transform
        self.X_normalized = self.guassian_transform(self.X_normalized)

        # polynomial transform
        self.X_normalized = self.polynomial_transform(self.X_normalized)

        self.X_normalized = np.hstack((self.X_normalized, self.y.reshape(-1, 1)))
        # time sequence transform, check time_seq_demo.py for simple demo
        self.X_timeseq, self.y_timeseq = [],[] 
        for i in range(len(self.X_normalized)-self.timestep): 
            self.X_timeseq.append(self.X_normalized[i:(i+self.timestep)])
            self.y_timeseq.append(self.y[i+self.timestep])
        
        # print(self.X_normalized[-7:-2])
        # print(self.data_x[-2])
        # print(self.data_y[-2])
        
        # split data into train, vali, test set randomly
        self.X_train, X_temp, self.y_train, y_temp = train_test_split(self.X_timeseq, self.y_timeseq, test_size=0.2)
        self.X_val, self.X_test, self.y_val, self.y_test = train_test_split(X_temp, y_temp, test_size=0.5)
        self.X_train, self.y_train = np.array(self.X_train), np.array(self.y_train)
        self.X_val, self.y_val = np.array(self.X_val), np.array(self.y_val)
        self.X_test, self.y_test = np.array(self.X_test), np.array(self.y_test)

    def train_model(self):
        self.model = Sequential()

        if self.layer == 1:
            self.model.add(LSTM(units=self.unit, input_shape=(self.X_train.shape[1], self.X_train.shape[2]), 
                                bias_regularizer=regularizers.l2(1e-6)))
        else:
            #multiple lstm layer
            self.model.add(LSTM(units=self.unit, 
                                return_sequences=True,
                                input_shape=(self.X_train.shape[1], self.X_train.shape[2]), 
                                bias_regularizer=regularizers.l2(1e-6)))
            for i in range(self.layer-2):
                self.model.add(LSTM(units=self.unit,
                                    return_sequences=True,
                                    bias_regularizer=regularizers.l2(1e-6)))
            self.model.add(LSTM(units=self.unit,
                                bias_regularizer=regularizers.l2(1e-6)))
            
        # Add a dense layer for binary classification
        self.model.add(Dense(units=1, activation='sigmoid'))

        # Compile the model
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        # train the model
        history = self.model.fit(self.X_train, self.y_train, 
                                epochs=self.epochs, 
                                batch_size=self.batchsize, 
                                validation_data=(self.X_val, self.y_val),
                                verbose=0) 
        # store the training history of loss 
        self.history = history.history
        #return history.history
    
    def evaluate_model(self):
        loss, accuracy = self.model.evaluate(self.X_test, self.y_test)
        print(f"Test Loss: {loss:.3f}, Test Accuracy: {accuracy:.3f}")

        # Get the predicted probabilities for each class
        predicted_probabilities = self.model.predict(self.X_test)

        # Convert the probabilities to binary predictions (0 or 1)
        predicted_labels = (predicted_probabilities > 0.5).astype(int)
        
        # avoid further analysis for useless result(predict all 1s or 0s)
        bad_threshold = 0.2
        if np.mean(predicted_labels==0) <= bad_threshold or np.mean(predicted_labels==0) >= (1- bad_threshold):
            print("useless result")
            accuracy = 0

        # visualization by heatmep
        cm = confusion_matrix(self.y_test, predicted_labels)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title("Distribution of results")
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.savefig(f"result/{self.hyperparameter}_guassian{self.guassian_choice}_result.png")
        plt.close()

        # visualization of loss history
        plt.plot(range(len(self.history['loss'])), self.history['loss'], 
                label="training loss")
        plt.plot(range(len(self.history['val_loss'])), self.history['val_loss'], 
                label="validation loss")
        plt.title(f'{self.stock} loss during training in epochs')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.tight_layout()
        plt.legend()
        plt.savefig(f"result/{self.hyperparameter}_guassian{self.guassian_choice}_loss.png")
        plt.close()

        # visualization of accuracies by epochs
        plt.plot(range(len(self.history['accuracy'])), self.history['accuracy'],
                label="training")
        plt.plot(range(len(self.history['val_accuracy'])), self.history['val_accuracy'],
                label="validation")
        plt.title(f'{self.stock} accuracies during training in epochs')
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        plt.tight_layout()
        plt.legend()
        plt.savefig(f"result/{self.hyperparameter}_guassian{self.guassian_choice}_accuracies.png")
        plt.close()
        return accuracy





# stock1 = "^SP500-20"
# stock2 = "aapl"
stock = "^SP500-20"
batchsize, epoch, unit, layer = 32, 1, 50, 1
temp_hyperparameter = [batchsize, epoch, unit, layer]
datasize = 1000

model1 = model(stock, datasize, temp_hyperparameter)
model1.set_guassian(1)
model1.process_data()
model1.train_model()
accuracy = model1.evaluate_model()


# result_dict = {"origin":[],
#            "only_guassian":[],
#            "origin_guassian":[]}
# num_left = 3*50
# for i in range(50):
#   model1 = model(stock, datasize, temp_hyperparameter)
#   count = 1
#   for key in result_dict.keys():
#     model1.set_guassian(count)
#     model1.process_data()
#     model1.train_model()
#     result_dict[key].append(model1.evaluate_model())
#     count += 1
#     num_left -=1
#     print(f"{num_left}record left")
# df= pd.DataFrame(result_dict)
# plt.hist(df, label=df.columns, bins=15)
# plt.legend()
# plt.title(f"accuracy of {stock} in 50 trails using guassian or not")
# plt.ylabel("count")
# plt.xlabel("accuracy")
# plt.savefig("guassian_compare1.png")