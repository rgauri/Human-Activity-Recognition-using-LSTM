from google.colab import drive

drive.mount('/content/drive')
!ls "/content/drive/My Drive"

!pip3 install tensorflow-gpu==2.0.0

import tensorflow  as tf
import tensorflow.keras as keras
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
from glob import glob

Sampling_Freq = 50
noisy_rows = 250
N_features = 6
N_WindowSize = 250
N_WindowShift = 125
N_prefetch = 8
batch_size = 16
N_shuffleBuffer = 200
N_classes = 12

df_train = []
df_test = []
df_validate = []
labels_train = []
labels_test = []
labels_validate = []
raw_acc_data_frame = []
rows_per_exp = []

Acitivity_Names = AN = {
    1: 'WALKING', 2: 'WALKING_UPSTAIRS', 3: 'WALKING_DOWNSTAIRS',  # 3 dynamic activities
    4: 'SITTING', 5: 'STANDING', 6: 'LIYING',  # 3 static activities

    7: 'STAND_TO_SIT', 8: 'SIT_TO_STAND', 9: 'SIT_TO_LIE', 10: 'LIE_TO_SIT',
    11: 'STAND_TO_LIE', 12: 'LIE_TO_STAND',  # 6 postural Transitions
}

#Function to read csv files with appropriate column headers
#Input: file path and column headers
#Output: Pandas dataframe
def read_data(file_path, column_names):
    df = pd.read_csv(file_path, delimiter=' ',
                     header=None,
                     names=column_names)

    x = column_names[0]
    y = column_names[1]
    z = column_names[2]
    df[x] = df[x].apply(convert_to_float)
    df[y] = df[y].apply(convert_to_float)
    df[z] = df[z].apply(convert_to_float)
    df.dropna(axis=0, how='any', inplace=True)
    return df

#Function to convert all accelerometer and gyroscope values to float
def convert_to_float(x):
    try:
        return np.float(x)
    except:
        return np.nan

#Read files from Google drive and sort in ascending order
Raw_data_paths = sorted(glob("/content/drive/My Drive/RawData/*"))

#Seperate out gyro and accelerometer files
Raw_acc_paths = Raw_data_paths[0:61]
Raw_gyro_paths = Raw_data_paths[61:122]

#Read the labels file which is the last file in raw dataset
labels_dataframe = pd.read_csv(Raw_data_paths[122], delimiter=' ',
                               header=None,
                               names=['ExpID', 'UserID', 'ActivityID', 'StartTime', 'EndTime'])

#Function to check the number of samples available per user, experiment and activity type 
#This is just for data visualization and understanding the imbalance in dataset
#Input: User ID, exp ID, activity type
#Output: Number of corresponding rows in labels file
def data_distribution(user_ID, exp_ID, activity_ID):
    # To select rows in labels file of a fixed activity in a fixed experiment
    return labels_dataframe[
        (labels_dataframe["ExpID"] == exp_ID) &
        (labels_dataframe["UserID"] == user_ID) &
        (labels_dataframe["ActivityID"] == activity_ID)
        ]
#Sample Plot of this data distribution for a particular user for all activity types
for activity_Id in range(1, 13):  # iterating throw activity ids from 1 to 12
    print('Activity number ' + str(activity_Id))
    display(data_distribution(1, 1, activity_Id))

#Function to plot signals **********************************************************************
#Input: data, user id, exp id, activity type, signal type whether accelerometer or gyroscope and height, weight for plotting figures
#Output: Plot obtained
def visualize_signals(data_frame, user_id, exp_id, act, signal_type, width, height):
    start_point, end_point = labels_dataframe[
        (labels_dataframe["ExpID"] == int(exp_id)) &
        (labels_dataframe["UserID"] == int(user_id)) &
        (labels_dataframe["ActivityID"] == act)
        ][['StartTime', 'EndTime']].iloc[0]
    
    #Remove noisy rows  
    data_df = data_frame[start_point - noisy_rows:end_point - noisy_rows]
    columns = data_df.columns

    if signal_type == 'acc':  # if the columns to be visualized are acceleration columns

        # acceleration columns are the first 3 columns acc_X, acc_Y and acc_Z
        X_component = data_df[columns[0]]  # acc_X
        Y_component = data_df[columns[1]]  # acc_Y
        Z_component = data_df[columns[2]]  # acc_Z

        # accelerations legends
        legend_X = 'acc_X'
        legend_Y = 'acc_Y'
        legend_Z = 'acc_Z'

        # y axis label
        figure_Ylabel = 'Acceleration'
        title = "acceleration signals of experiment " + str(exp_id) + " when user " + str(
            user_id) + ' was performing activity: ' + str(act) + '(' + AN[act] + ')'

    elif signal_type == 'gyro':  # if the columns to be visualized are gyro columns

        # gyro columns are the last 3 columns gyro_X, gyro_Y and gyro_Z
        X_component = data_df[columns[3]]  # gyro_X
        Y_component = data_df[columns[4]]  # gyro_Y
        Z_component = data_df[columns[5]]  # gyro_Z

        # gyro legends
        legend_X = 'gyro_X'
        legend_Y = 'gyro_Y'
        legend_Z = 'gyro_Z'

        # y axis label
        figure_Ylabel = 'Angular Velocity [rad/s]'

        # title for gyroscope
        title = "gyroscope signals of experiment " + str(exp_id) + " when user " + str(
            user_id) + ' was performing activity: ' + str(act) + '(' + AN[act] + ')'

    # chosing colors : yellow for X component cyan for Y component and magenta for Z component
    colors = ['y', 'c', 'm']
    len_df = len(data_df)  # number of rows in this dataframe

    # converting row numbers into time duration
    time = [1 / float(Sampling_Freq) * j for j in range(len_df)]

    # Define the figure and setting dimensions width and height
    fig = plt.figure(figsize=(width, height))

    # plot each signal component
    _ = plt.plot(time, X_component, color='y', label=legend_X)
    _ = plt.plot(time, Y_component, color='c', label=legend_Y)
    _ = plt.plot(time, Z_component, color='m', label=legend_Z)

    # Set the figure info defined earlier
    _ = plt.ylabel(figure_Ylabel)
    _ = plt.xlabel('Time in seconds (s)')
    _ = plt.title(title)

    # location of the figure's legends
    _ = plt.legend(loc="upper left")  # upper left corner

    path = '/content/drive/My Drive/HAR_Images/' + title + '.png'
    plt.savefig(path, bbox_inches='tight')
    plt.show()
#End of Function to plot signals **********************************************************************

#Sample plot for user id=1, exp id=1 ***************************************************************
user_id = 1
exp_id = 1
raw_acc_columns = ['acc_X', 'acc_Y', 'acc_Z']
raw_gyro_columns = ['gyro_X', 'gyro_Y', 'gyro_Z']
#Create appropriate dataframe for the plot for user id=1, exp id=1
raw_acc_data_frame = read_data(Raw_data_paths[user_id], raw_acc_columns)
raw_acc_data_frame = raw_acc_data_frame.iloc[noisy_rows:]
raw_acc_data_frame = raw_acc_data_frame.iloc[:-noisy_rows]
raw_gyro_data_frame = read_data(Raw_data_paths[user_id + 61], raw_gyro_columns)
raw_gyro_data_frame = raw_gyro_data_frame.iloc[noisy_rows:]
raw_gyro_data_frame = raw_gyro_data_frame.iloc[:-noisy_rows]
raw_signals_data_frame = pd.concat([raw_acc_data_frame, raw_gyro_data_frame], axis=1)
for act in range(1, 13):  # Iterating throw each activity Id from 1 to 12
    visualize_signals(raw_signals_data_frame, user_id, exp_id, act, 'acc', 18,
                      3)  # visualize acc signals related to this activity
    visualize_signals(raw_signals_data_frame, user_id, exp_id, act, 'gyro', 18,
                      3)  # visualize gyro signals related to this activity
#End of sample plot for user 1, exp id=1 ************************************************************

#Seperation of data into train/test/validation datasets begins here************************************************************
#For all path indices: (gyro path index = accelerometer path index + 61)
for path_index in range(0, 61):
    #Extract user id, exp id from the file name
    user = int(Raw_data_paths[path_index][-6:-4])
    exp_id = int(Raw_data_paths[path_index][-13:-11])
    
    #Create dataframes for accelerometer data
    raw_acc_data_frame = read_data(Raw_data_paths[path_index], raw_acc_columns)
    #Remove noisy rows
    raw_acc_data_frame = raw_acc_data_frame.iloc[noisy_rows:]
    raw_acc_data_frame = raw_acc_data_frame.iloc[:-noisy_rows]
    #Normalize data using Z-Score normalization
    raw_acc_data_frame['acc_X'] = (raw_acc_data_frame['acc_X'] - raw_acc_data_frame['acc_X'].mean()) / \
                                  raw_acc_data_frame['acc_X'].std(ddof=0)
    raw_acc_data_frame['acc_Y'] = (raw_acc_data_frame['acc_Y'] - raw_acc_data_frame['acc_Y'].mean()) / \
                                  raw_acc_data_frame['acc_Y'].std(ddof=0)
    raw_acc_data_frame['acc_Z'] = (raw_acc_data_frame['acc_Z'] - raw_acc_data_frame['acc_Z'].mean()) / \
                                  raw_acc_data_frame['acc_Z'].std(ddof=0)
    # Round numbers
    raw_acc_data_frame = raw_acc_data_frame.round({'acc_X': 4, 'acc_Y': 4, 'acc_Z': 4})
    
    #Create dataframes for gyro data
    raw_gyro_data_frame = read_data(Raw_data_paths[path_index + 61], raw_gyro_columns)
    raw_gyro_data_frame = raw_gyro_data_frame.iloc[noisy_rows:]
    raw_gyro_data_frame = raw_gyro_data_frame.iloc[:-noisy_rows]
    #Normalize data using Z-Score normalization
    raw_gyro_data_frame['gyro_X'] = (raw_gyro_data_frame['gyro_X'] - raw_gyro_data_frame['gyro_X'].mean()) / \
                                    raw_gyro_data_frame['gyro_X'].std(ddof=0)
    raw_gyro_data_frame['gyro_Y'] = (raw_gyro_data_frame['gyro_Y'] - raw_gyro_data_frame['gyro_Y'].mean()) / \
                                    raw_gyro_data_frame['gyro_Y'].std(ddof=0)
    raw_gyro_data_frame['gyro_Z'] = (raw_gyro_data_frame['gyro_Z'] - raw_gyro_data_frame['gyro_Z'].mean()) / \
                                    raw_gyro_data_frame['gyro_Z'].std(ddof=0)
    raw_gyro_data_frame = raw_gyro_data_frame.round({'gyro_X': 4, 'gyro_Y': 4, 'gyro_Z': 4}) #round the data values
    #Concat accelerometer and gyro data
    raw_signals_data_frame = pd.concat([raw_acc_data_frame, raw_gyro_data_frame], axis=1) 
    raw_signals_data_frame_numpy = raw_signals_data_frame.to_numpy()
    #Create labels array with all 0s (unlabelled data has to be labelled as 0)
    labels = np.zeros(len(raw_signals_data_frame_numpy))

    #Assign value to labels array created above using the values from the labels dataframe
    #This is to assign values to labels array which are not unlabelled and have labels from 0 to 12. 
    for index, rows in labels_dataframe.iterrows():
        if rows['ExpID'] == exp_id:
            start = rows['StartTime']
            end = rows['EndTime']
            label_value = int(rows['ActivityID'])
            labels[start - noisy_rows:end - noisy_rows] = label_value

    if 1 <= user <= 21: #training samples
        labels_train.append(labels)
        for row in raw_signals_data_frame_numpy:
            row = row.reshape(1, 6).flatten()
            df_train.append(row)

    elif 22 <= user <= 27: #test samples
        labels_test.append(labels)
        for row in raw_signals_data_frame_numpy:
            row = row.reshape(1, 6).flatten()
            df_test.append(row)

    elif 28 <= user <= 30: #validation samples
        labels_validate.append(labels)
        for row in raw_signals_data_frame_numpy:
            row = row.reshape(1, 6).flatten()
            df_validate.append(row)        

labels_train = np.concatenate(labels_train).astype('int32')
labels_test = np.concatenate(labels_test).astype('int32')
labels_validate = np.concatenate(labels_validate).astype('int32')
#Seperation of data into train/test/validation datasets ends here************************************************************

# Plotting data for first few samples of train data set to check if data is correctly assigned to the numpy array 
df_train_subset = df_train[0:20000]
len_df = len(df_train_subset)
time = [1 / float(Sampling_Freq) * j for j in range(len_df)]
legend_X = 'acc_X'
legend_Y = 'acc_Y'
legend_Z = 'acc_Z'
fig = plt.figure(figsize=(18, 5))

# acc columns are the first 3 columns acc_X, acc_Y and acc_Z
X_component = [df_train_subset[i][0] for i in range(len_df)]
Y_component = [df_train_subset[i][1] for i in range(len_df)]
Z_component = [df_train_subset[i][2] for i in range(len_df)]

plt.plot(time, X_component, color='y', label=legend_X)
plt.plot(time, Y_component, color='c', label=legend_Y)
plt.plot(time, Z_component, color='m', label=legend_Z)
plt.legend(loc="upper left")
title = 'Acc_TrainData'
path = '/content/drive/My Drive/HAR_Images/' + title + '.png'
plt.savefig(path, bbox_inches='tight')
plt.show()

# gyro columns are the last 3 columns gyro_X, gyro_Y and gyro_Z
legend_X = 'gyro_X'
legend_Y = 'gyro_Y'
legend_Z = 'gyro_Z'

X_component = [df_train_subset[i][3] for i in range(len_df)]
Y_component = [df_train_subset[i][4] for i in range(len_df)]
Z_component = [df_train_subset[i][5] for i in range(len_df)]

fig = plt.figure(figsize=(18, 5))
plt.plot(time, X_component, color='y', label=legend_X)
plt.plot(time, Y_component, color='c', label=legend_Y)
plt.plot(time, Z_component, color='m', label=legend_Z)
plt.legend(loc="upper left")
title = 'Gyro_TrainData'
path = '/content/drive/My Drive/HAR_Images/' + title + '.png'
plt.savefig(path, bbox_inches='tight')
plt.show()
#Plotting train data for few samples ends here

# Plotting number of training examples for each activity ID
unique, counts = np.unique(labels_train, return_counts=True)
print(dict(zip(unique, counts)))

df = pd.DataFrame({'data': labels_train})
df['data'].value_counts().plot(kind='bar',
                               title='Training Examples by Activity')
plt.xlabel('Activity ID')
plt.ylabel('Number of Training Examples')
title = 'Training Examples by Activity'
path = '/content/drive/My Drive/HAR_Images/' + title + '.png'
plt.savefig(path, bbox_inches='tight')
plt.show()

'''
exp_ids=[i for i in range(1,62)]
plt.bar(exp_ids,rows_per_exp) # plot the bar plot
plt.xlabel('experiment number') 
plt.ylabel('number of rows') 
plt.title('number of rows per experiment') 
plt.grid()
plt.show()
'''
#Function to one hot code labels which now have values 0 to 12
#0 label is encoded to all 0s
#Input: label  
#Output: one hot encoded label
def one_hot_labels(n_labels):
    labels_one_hot = []
    x = np.zeros(N_classes)
    for i in range(0, n_labels.size):
        x = np.zeros(N_classes)

        if n_labels[i] == 1:
            x[0] = 1
            labels_one_hot.append(x)

        elif n_labels[i] == 2:
            x[1] = 1
            labels_one_hot.append(x)

        elif n_labels[i] == 3:
            x[2] = 1
            labels_one_hot.append(x)

        elif n_labels[i] == 4:
            x[3] = 1
            labels_one_hot.append(x)

        elif n_labels[i] == 5:
            x[4] = 1
            labels_one_hot.append(x)

        elif n_labels[i] == 6:
            x[5] = 1
            labels_one_hot.append(x)

        elif n_labels[i] == 7:
            x[6] = 1
            labels_one_hot.append(x)

        elif n_labels[i] == 8:
            x[7] = 1
            labels_one_hot.append(x)

        elif n_labels[i] == 9:
            x[8] = 1
            labels_one_hot.append(x)

        elif n_labels[i] == 10:
            x[9] = 1
            labels_one_hot.append(x)

        elif n_labels[i] == 11:
            x[10] = 1
            labels_one_hot.append(x)

        elif n_labels[i] == 12:
            x[11] = 1
            labels_one_hot.append(x)

        elif n_labels[i] == 0:
            labels_one_hot.append(x)

    return labels_one_hot

#One hot encode labels of train/test/validate dataset
train_labels_one_hot = one_hot_labels(labels_train)
test_labels_one_hot = one_hot_labels(labels_test)
validate_labels_one_hot = one_hot_labels(labels_validate)

#Function to create windows and window overlap (Manually created this function instead of using tf.data.Dataset.window function)
#Input: acc. and gyro concatenated data, labels data, window size, window overlap
#Output: files and labels with window function appplied
def create_window(files, labels, window_size, window_overlap):
    list_files = []
    list_labels = []
    for i in range(0, int(len(files) / window_overlap) - 1):
        list_files.append(files[i * window_overlap:i * window_overlap + window_size])
        list_labels.append(labels[i * window_overlap:i * window_overlap + window_size])
    return list_files, list_labels

#Apply window function to test/train/validate dataset
train_files, train_labels = create_window(df_train, train_labels_one_hot, N_WindowSize, N_WindowShift)
test_files, test_labels = create_window(df_test, test_labels_one_hot, N_WindowSize, N_WindowShift)
validate_files, validate_labels = create_window(df_validate, validate_labels_one_hot, N_WindowSize, N_WindowShift)

#Build dataset and create batches of data and shuffle the batches
def build_dataset(files, labels):
    ds = tf.data.Dataset.from_tensor_slices((files, labels))
    ds = ds.batch(batch_size)
    ds = ds.shuffle(N_shuffleBuffer)
    ds = ds.prefetch(N_prefetch)
    return ds

#Datasets created here
train_ds = build_dataset(train_files, train_labels)
test_ds = build_dataset(test_files, test_labels)
validate_ds = build_dataset(validate_files, validate_labels)

#Debug : to check if value-labels pairs still are correctly matched
for files, labels in train_ds.take(1):
    print(files)
    print(labels)
    print('\n')

#Input Pipleline successfully created ****************************************************************************************
