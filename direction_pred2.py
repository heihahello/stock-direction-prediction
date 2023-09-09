import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import math
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import yfinance as yf
from sklearn.metrics import confusion_matrix
from keras import regularizers



class model():
    def __init__(self, stock, datasize):
        self.stock = stock
        self.get_data(stock, datasize)
        self.timestep = 30
        self.layer = 1
        self.unit = 200


    def get_data(self, stock, datasize):
        # obtain the data
        raw_data = yf.download(stock)

        # make sure we use constant data
        cutoff_date = pd.to_datetime('2023-09-01')
        raw_data = raw_data.loc[raw_data.index <= cutoff_date]

        # preprocess the data
        # set up as 1, down as 0
        raw_data["Direction"] = (raw_data['Close']>=raw_data['Close'].shift()).astype('int')
        
        self.data = raw_data[-datasize:]
        print(self.data.tail())

        plt.figure()
        plt.plot(self.data[['Close']], color = 'black')
        plt.title(f'{self.stock} history in past {datasize} days')
        plt.xlabel('date')
        plt.ylabel('closing price')
        plt.savefig("history2.png")

    def process_data(self):
        # Construct X and y
        self.X = self.data.drop(columns = ['Direction']).values
        self.y = self.data["Direction"].values

        # normalize input features
        scaler = MinMaxScaler(feature_range = (0,1))
        self.X_normalized = scaler.fit_transform(self.X)

        # time sequence transform
        self.X_timeseq, self.y_timeseq = [],[] 
        for i in range(len(self.X_normalized)-self.timestep): 
            self.X_timeseq.append(self.X_normalized[i:(i+self.timestep)])
            self.y_timeseq.append(self.y[i+self.timestep])
        
        # print(self.X_normalized[-7:-2])
        # print(self.data_x[-2])
        # print(self.data_y[-2])
        
        # split the data into train, val, test set
        self.X_train, X_temp, self.y_train, y_temp = train_test_split(self.X_timeseq, self.y_timeseq, test_size=0.2)
        self.X_val, self.X_test, self.y_val, self.y_test = train_test_split(X_temp, y_temp, test_size=0.5)
        # transform the input to format needed by LSTM
        self.X_train, self.y_train = np.array(self.X_train), np.array(self.y_train)
        self.X_val, self.y_val = np.array(self.X_val), np.array(self.y_val)
        self.X_test, self.y_test = np.array(self.X_test), np.array(self.y_test)

    def train_model(self):
        self.model = Sequential()

        if self.layer == 1:
            self.model.add(LSTM(units=50, input_shape=(self.X_train.shape[1], self.X_train.shape[2]), 
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
            
        self.model.add(Dense(units=1, activation='sigmoid'))

        # Compile the model
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        # train the model
        history = self.model.fit(self.X_train, self.y_train, 
                                epochs=70, 
                                batch_size=10, 
                                validation_data=(self.X_val, self.y_val)) 
        self.history = history.history
        #return history.history
    
    def evaluate_model(self):
        loss, accuracy = self.model.evaluate(self.X_test, self.y_test)
        print(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")

        # Get the predicted probabilities 
        predicted_probabilities = self.model.predict(self.X_test)

        # Convert the probabilities to binary predictions (0 or 1)
        predicted_labels = (predicted_probabilities > 0.5).astype(int)
        # print(predicted_probabilities)
        # print(predicted_labels)
        cm = confusion_matrix(self.y_test, predicted_labels)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.savefig("result2.png")

        plt.figure(figsize=(20, 15))
        plt.plot(range(len(self.history['loss'])), self.history['loss'], 
                label="loss of epochs")
        plt.title(f'{self.stock} loss during training in epochs')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.tight_layout()
        plt.savefig("loss2.png")
        plt.close()





stock1 = "^SP500-20"
stock2 = "aapl"
model1 = model(stock2, 1000)
model1.process_data()
model1.train_model()
model1.evaluate_model()

