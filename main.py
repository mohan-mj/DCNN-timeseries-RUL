"""main.py: Turbo Jet Engine RUL Prediction"""

__author__ = "Jidhu Mohan M"
__license__ = "GNU GPL"
__version__ = "0.1.0"
__maintainer__ = "Jidhu Mohan M"
__email__ = "Jidhu.Mohan@gmail.com"
__status__ = "Development"

# %% import libraries for data handling
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from snapml import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit
import pickle as pk

# import deep learning libraries
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Conv1D, Flatten, Activation
from tensorflow.keras.layers import Dropout, MaxPooling1D, Input
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers

# %%

class TURBO:
    """class contains functions to train the DCNN model and predict the remaining useful life (RUL) of turbo jet engine
    """

    def __init__(self) -> None:
        """set column names, read data
        """
        # define column names
        self.index_names = ['unit_no', 'time_cycles']
        self.setting_names = ['setting_1', 'setting_2', 'setting_3']
        self.sensor_names = [f's_{i}' for i in range(1, 22)]
        self.col_names = self.index_names + self.setting_names + self.sensor_names

        # read data from files
        self.read_data()

    def read_data(self,
                    filename='dataset/train_FD001.txt',
                  filename_X="dataset/test_FD001.txt",
                  filename_y="dataset/RUL_FD001.txt"):
        """read data from file and create train and test data

        Args:
            filename (str, optional): train filename. Defaults to 'dataset/train_FD001.txt'.
            filename_X (str, optional): test filename. Defaults to "dataset/test_FD001.txt".
            filename_y (str, optional): test filename. Defaults to "dataset/RUL_FD001.txt".
        """
        # read train data and copute RUL and add for training
        self.train = self._add_RUL(pd.read_csv(filename,
                                                sep='\s+', 
                                                header=None, 
                                                names=self.col_names))
        self.test = pd.read_csv(filename_X, sep='\s+',
                                header=None, 
                                names=self.col_names)
        self.y_test = pd.read_csv(filename_y,
                                  sep='\s+', 
                                  header=None, 
                                  names=['RUL'])

    def _add_RUL(self, df):
        """add RUL column to dataframe

        Args:
            df (_type_): dataframe input

        Returns:
            _type_: resulting dataframe with RUL column
        """
        # Get the total number of cycles for each unit
        grouped_by_unit = df.groupby(by="unit_no")
        max_cycle = grouped_by_unit["time_cycles"].max()

        # Merge the max cycle back into the original frame
        result_frame = df.merge(max_cycle.to_frame(name='max_cycle'), 
                                left_on='unit_no', 
                                right_index=True)

        # Calculate remaining useful life for each row
        remaining_useful_life = result_frame["max_cycle"] - result_frame["time_cycles"]
        result_frame["RUL"] = remaining_useful_life

        # drop max_cycle as it's no longer needed
        result_frame = result_frame.drop("max_cycle", axis=1)
        return result_frame

    def plot_sensors(self, sensor_names):
        """Plot sensors

        Args:
            sensor_names (list): sensor names
        """
        if sensor_names is None:
            sensor_names = self.sensor_names
        for sensor_name in sensor_names:
            plt.figure(figsize=(10, 4))
            for i in self.train['unit_no'].unique():
                if (i % 10 == 0):  # only plot every 10th unit_no
                    plt.plot('RUL', sensor_name,
                             data=self.train[self.train['unit_no'] == i])
            plt.xlim(250, 0)  # reverse the x-axis so RUL counts down to zero
            plt.xticks(np.arange(0, 275, 25))
            plt.ylabel(sensor_name)
            plt.xlabel('Remaining Useful Life(RUL)')
            plt.show()

    def plot_max_RUL(self, df_rul, title=" "):
        """distribution of RUL for train and test

        Args:
            df_max_rul (_type_): _description_
            title (str, optional): _description_. Defaults to " ".
        """
        df_rul.hist(bins=15, figsize=(10, 4))
        plt.title(title)
        plt.xlabel('RUL')
        plt.ylabel('frequency')
        plt.show()

    def plot(self):
        """plot sensor data and RUL distribution for train and test"""
        # plot raw sensor data
        self.plot_sensors(self.sensor_names)

        # plot RUL histogram
        self.plot_max_RUL(self.train[['unit_no', 'RUL']].groupby(
            'unit_no').max().reset_index()['RUL'])
        self.plot_max_RUL(self.y_test, "y_test")

    def evaluate(self, y_true, y_hat, label='test'):
        """evaluate model"""
        mse = mean_squared_error(y_true, y_hat)
        rmse = np.sqrt(mse)
        variance = r2_score(y_true, y_hat)
        print(f'{label} set RMSE:{rmse}, R2:{variance}')

    def model_GLM(self):
        """create generalized linear model (GLM) using snapml
        """
        # get X_train from train
        X_train = self.train[self.setting_names +
                             self.sensor_names].copy()

        # get last row of each engine for testing
        X_test = self.test.drop('time_cycles', axis=1).groupby(
            'unit_no').last().copy()

        # clip RUL by 125
        y_train = self.train['RUL'].copy()
        y_train = y_train.clip(upper=125)

        # create and fit model
        reg = LinearRegression()
        reg.fit(X_train.values, y_train.values)

        # predict and evaluate
        y_pred_train = reg.predict(X_train.values)
        self.evaluate(y_train.values, y_pred_train, 'train')

        y_pred_test = reg.predict(X_test.values)
        self.evaluate(self.y_test.values, y_pred_test)

    def add_operating_condition(self, df):
        """condition specific scaling

        Args:
            df (_type_): _description_

        Returns:
            _type_: _description_
        """
        df_op_cond = df.copy()

        df_op_cond['setting_1'] = df_op_cond['setting_1'].round()
        df_op_cond['setting_2'] = df_op_cond['setting_2'].round(decimals=2)

        # converting settings to string and concatinating makes the operating condition into a categorical variable
        df_op_cond['op_cond'] = df_op_cond['setting_1'].astype(str) + '_' + \
            df_op_cond['setting_2'].astype(str) + '_' + \
            df_op_cond['setting_3'].astype(str)
        return df_op_cond

    def condition_scaler(self, df_train, df_test, sensor_names):
        """apply operating condition specific scaling

        Args:
            df_train (_type_): _description_
            df_test (_type_): _description_
            sensor_names (_type_): _description_

        Returns:
            _type_: _description_
        """
        scaler = StandardScaler()
        for condition in df_train['op_cond'].unique():
            scaler.fit(df_train.loc[df_train['op_cond'] == condition, sensor_names])
            df_train.loc[df_train['op_cond'] == condition, sensor_names] = scaler.transform(
                df_train.loc[df_train['op_cond'] == condition, sensor_names])
            df_test.loc[df_test['op_cond'] == condition, sensor_names] = scaler.transform(
                df_test.loc[df_test['op_cond'] == condition, sensor_names])
        return df_train, df_test

    def exponential_smoothing(self, df, sensors, n_samples, alpha=0.4):
        """exponential smoothing"""
        df = df.copy()
        # first, take the exponential weighted mean
        df[sensors] = df.groupby('unit_no')[sensors].apply(
            lambda x: x.ewm(alpha=alpha).mean())

        # second, drop first n_samples of each unit_no to reduce filter delay
        def create_mask(data, samples):
            result = np.ones_like(data)
            result[0:samples] = 0
            return result

        mask = df.groupby('unit_no')['unit_no'].transform(
            create_mask, samples=n_samples).astype(bool)
        df = df[mask]
        return df

    def gen_train_data(self, df, sequence_length, columns):
        """generate training dataframe

        Args:
            df (_type_): _description_
            sequence_length (_type_): _description_
            columns (_type_): _description_

        Yields:
            _type_: _description_
        """
        data = df[columns].values
        num_elements = data.shape[0]

        # -1 and +1 because of Python indexing
        for start, stop in zip(range(0, num_elements-(sequence_length-1)), range(sequence_length, num_elements+1)):
            yield data[start:stop, :]

    def gen_data_wrapper(self, df, sequence_length, columns, unit_nos=np.array([])):
        """wrapper for gen_train_data for data array creation

        Args:
            df (_type_): _description_
            sequence_length (_type_): _description_
            columns (_type_): _description_
            unit_nos (_type_, optional): _description_. Defaults to np.array([]).

        Returns:
            _type_: _description_
        """
        if unit_nos.size <= 0:
            unit_nos = df['unit_no'].unique()

        data_gen = (list(self.gen_train_data(df[df['unit_no'] == unit_no], sequence_length, columns))
                    for unit_no in unit_nos)
        data_array = np.concatenate(list(data_gen)).astype(np.float32)
        return data_array

    def gen_labels(self, df, sequence_length, label):
        """generate label dataframe

        Args:
            df (_type_): _description_
            sequence_length (_type_): _description_
            label (_type_): _description_

        Returns:
            _type_: _description_
        """
        data_matrix = df[label].values
        num_elements = data_matrix.shape[0]

        # -1 because I want to predict the rul of that last row in the sequence, not the next row
        return data_matrix[sequence_length-1:num_elements, :]

    def gen_label_wrapper(self, df, sequence_length, label, unit_nos=np.array([])):
        """wrapper on gen_labels for generating label dataset

        Args:
            df (_type_): _description_
            sequence_length (_type_): _description_
            label (_type_): _description_
            unit_nos (_type_, optional): _description_. Defaults to np.array([]).

        Returns:
            _type_: _description_
        """
        if unit_nos.size <= 0:
            unit_nos = df['unit_no'].unique()

        label_gen = [self.gen_labels(df[df['unit_no'] == unit_no], sequence_length, label)
                     for unit_no in unit_nos]
        label_array = np.concatenate(label_gen).astype(np.float32)
        return label_array

    def gen_test_data(self, df, sequence_length, columns, mask_value):
        """generate test dataset for prediction

        Args:
            df (_type_): _description_
            sequence_length (_type_): _description_
            columns (_type_): _description_
            mask_value (_type_): _description_

        Yields:
            _type_: _description_
        """
        if df.shape[0] < sequence_length:
            data_matrix = np.full(shape=(sequence_length, len(
                columns)), fill_value=mask_value)  # pad
            idx = data_matrix.shape[0] - df.shape[0]
            # fill with available data
            data_matrix[idx:, :] = df[columns].values
        else:
            data_matrix = df[columns].values

        # specifically yield the last possible sequence
        stop = num_elements = data_matrix.shape[0]
        start = stop - sequence_length
        for i in list(range(1)):
            yield data_matrix[start:stop, :]

    def build_train_model(self, sequence_length=20, plot=False, epochs=20):
        """build and train model"""

        input_shape=(sequence_length, self.train_array.shape[2])
        block_size, nblocks = 2, 2
        kernel_size=3
        dropout=0.5
        fc1, fc2 =256, 128
        conv_activation='relu' #f.keras.layers.LeakyReLU(alpha=0.1)
        batch_normalization=1

        input_tensor = Input(input_shape)
        x = input_tensor
        # DCNN
        for i, n_cnn in enumerate([block_size] * nblocks):
            for j in range(n_cnn):
                x = Conv1D(32*2**min(i, 2), kernel_size=kernel_size, padding='same', 
                        kernel_regularizer=regularizers.l1_l2(),
                        kernel_initializer='he_uniform',
                        name='Conv%d_Block%d' % (j, i) )(x)
                if batch_normalization:
                    x = BatchNormalization(name='BN%d_Block%d' % (j, i))(x)
                x = Activation(conv_activation, 
                                name='A%d_Block%d' % (j, i))(x)
            x = MaxPooling1D(2, name='MP%d_Block%d' % (j, i))(x)
            if dropout > 0:
                x = Dropout(dropout, name='DO%d_Block%d' % (j, i))(x)

        x = Flatten()(x)

        # FNN
        x = Dense(fc1, name='FC1', 
                kernel_regularizer=regularizers.l1_l2())(x)
        x = Activation(conv_activation, name='Act_FC1' )(x)
        if dropout > 0:
            x = Dropout(dropout, name='DO_FC1')(x)

        x = Dense(fc2, name='FC2', 
                kernel_regularizer=regularizers.l1_l2())(x)
        x = Activation(conv_activation, name='Act_FC2')(x)
        if dropout > 0:
            x = Dropout(dropout, name='DO_FC2')(x)

        x = Dense(1, activation='relu', name='predictions')(x) 

        self.model = Model(inputs=input_tensor, outputs=x)

        self.model.compile(loss='mean_squared_error', 
                        optimizer=Adam(), #lr=0.001, decay=1e-6
                        metrics=[tf.keras.metrics.MeanAbsoluteError(name="MAE")])

        # model training
        history = self.model.fit(self.train_split_array, self.train_split_label,
                                 validation_data=(
                                     self.val_split_array, self.val_split_label),
                                 epochs=epochs,
                                 batch_size=32)
        self.model.save('model/turbo_RUL_model.h5')
        pk.dump(history.history, open("model/history.pk", 'wb'))
        if plot:
            self.plot_loss(history)

    def read_model(self):
        """read model from file"""
        self.model = tf.keras.models.load_model(
            'model/turbo_RUL_model.h5')

    def model_eval(self):
        """evaluate model
        """
        # predict and evaluate
        y_hat_train = self.model.predict(self.train_array)
        self.evaluate(self.label_array, y_hat_train, 'train')

        y_hat_test = self.model.predict(self.test_array)
        self.evaluate(self.y_test, y_hat_test)

    def plot_loss(self, fit_history):
        """plot training history

        Args:
            fit_history (_type_): _description_
        """
        plt.figure(figsize=(13, 5))
        plt.plot(range(1, len(
            fit_history.history['loss'])+1), fit_history.history['loss'], label='train')
        plt.plot(range(1, len(fit_history.history['val_loss'])+1),
                 fit_history.history['val_loss'], label='validate')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

    def prepare_data_DCNN(self):

        # select sensors, selected by checking the sensor data
        remaining_sensors = ['s_2', 's_3', 's_4', 's_7', 's_8', 's_9',
                             's_11', 's_12', 's_13', 's_14', 's_15', 's_17', 's_20', 's_21']
        drop_sensors = [
            element for element in self.sensor_names if element not in remaining_sensors]

        sequence_length = 20  # default setting in the training function too
        self.train['RUL'].clip(upper=125, inplace=True)

        self.X_train = self.add_operating_condition(
            self.train.drop(drop_sensors, axis=1))
        self.X_test = self.add_operating_condition(
            self.test.drop(drop_sensors, axis=1))

        self.X_train, self.X_test = self.condition_scaler(
            self.X_train, self.X_test, remaining_sensors)

        self.X_train = self.exponential_smoothing(
            self.X_train, remaining_sensors, 0, 0.4)
        self.X_test = self.exponential_smoothing(
            self.X_test, remaining_sensors, 0, 0.4)

        # train-val split
        gss = GroupShuffleSplit(n_splits=1, train_size=0.80, random_state=42)
        for train_unit, val_unit in gss.split(self.X_train['unit_no'].unique(), groups=self.X_train['unit_no'].unique()):
            # gss returns indexes and index starts at 1
            train_unit = self.X_train['unit_no'].unique()[train_unit]
            val_unit = self.X_train['unit_no'].unique()[val_unit]

            self.train_split_array = self.gen_data_wrapper(
                self.X_train, sequence_length, remaining_sensors, train_unit)
            self.train_split_label = self.gen_label_wrapper(
                self.X_train, sequence_length, ['RUL'], train_unit)

            self.val_split_array = self.gen_data_wrapper(
                self.X_train, sequence_length, remaining_sensors, val_unit)
            self.val_split_label = self.gen_label_wrapper(
                self.X_train, sequence_length, ['RUL'], val_unit)

        # create sequences train, test
        self.train_array = self.gen_data_wrapper(
            self.X_train, sequence_length, remaining_sensors)

        self.label_array = self.gen_label_wrapper(
            self.X_train, sequence_length, ['RUL'])

        self.test_gen = (list(self.gen_test_data(self.X_test[self.X_test['unit_no'] == unit_no], 
                                                sequence_length, 
                                                remaining_sensors, -99.))
                         for unit_no in self.X_test['unit_no'].unique())
        self.test_array = np.concatenate(
            list(self.test_gen)).astype(np.float32)



if __name__ == "__main__":

    Turbo = TURBO()
    # Turbo.plot()

    # build and train regression model using snapML
    Turbo.model_GLM()
    # prepare data for training
    Turbo.prepare_data_DCNN()
    # train and evaluate model
    Turbo.build_train_model(plot=True, epochs=20)
    Turbo.model_eval()
