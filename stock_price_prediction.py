
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import matplotlib.pyplot as plt

# Load the dataset (example uses a CSV file)
# data = pd.read_csv('AAPL.csv')
# For simplicity, let's create some dummy data
data = pd.DataFrame({
    'Date': pd.date_range(start='1/1/2010', periods=2500),
    'Close': np.sin(np.linspace(0, 100, 2500)) * 50 + np.linspace(50, 150, 2500)
})

# Preprocessing
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

# Creating the training and testing datasets
train_data, test_data = train_test_split(scaled_data, test_size=0.2, shuffle=False)

# Function to create datasets for LSTM
def create_dataset(dataset, time_step=60):
    X, y = [], []
    for i in range(len(dataset) - time_step - 1):
        X.append(dataset[i:(i + time_step), 0])
        y.append(dataset[i + time_step, 0])
    return np.array(X), np.array(y)

time_step = 60
X_train, y_train = create_dataset(train_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)

# Reshape input to be [samples, time steps, features] for LSTM
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Building the LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, batch_size=64, epochs=10, verbose=1)

# Predictions
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# Inverse transform to get actual values
train_predict = scaler.inverse_transform(train_predict)
y_train = scaler.inverse_transform(y_train.reshape(-1, 1))
test_predict = scaler.inverse_transform(test_predict)
y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

# Plotting
plt.figure(figsize=(10,6))
plt.plot(data['Close'], label='True Data')
plt.plot(range(time_step, len(train_predict) + time_step), train_predict, label='Train')
plt.plot(range(len(data) - len(test_predict), len(data)), test_predict, label='Test')
plt.legend()
plt.title('Apple Stock Close Price')
plt.xlabel('Date')
plt.ylabel('Close')
plt.show()
