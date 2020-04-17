#Functional API of model

inputs = keras.Input(shape=(N_WindowSize, N_features))
x = layers.LSTM(256,return_sequences=True)(inputs)
x = layers.Dropout(0.1)(x)
x = layers.LSTM(128,return_sequences=True)(x)
x = layers.Dense(64,activation='relu')(x)
x = layers.Dropout(0.1)(x)
x = layers.Dense(32,activation='relu')(x)
outputs = layers.Dense(12,activation='softmax')(x)

mdl = keras.Model(inputs=inputs, outputs=outputs, name='DR_model')

#Define loss and optimizer
mdl.compile(loss=tf.keras.losses.categorical_crossentropy,
            optimizer='RMSProp',
            metrics=['accuracy'])

N_epochs = 20

#Train model
history = mdl.fit(train_ds, epochs=N_epochs,
                  validation_data=validate_ds)

#Plot curves for training and validation accuracy and loss
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(N_epochs)

plt.figure(figsize=(24, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.savefig('/content/drive/My Drive/HAR_Images/Metrics_HAR.png', bbox_inches='tight')
plt.savefig('/content/drive/My Drive/HAR_Images/Metrics_HAR.pdf', bbox_inches='tight')
plt.show()
#Plotting curves for acuracy and loss end here*********************************************************************

#Evaluate model on Test data
results = mdl.evaluate(test_ds)
print('test loss, test acc:', results)

####Test Data Visualization begins here******************************************************************************** 
df_test_subset=[]
test_sample= test_ds.take(1)
test_sample_unbatch = test_sample.unbatch() #unbatch the first sample of test data
y_pred = []
y_true = []

predict= mdl.predict(test_sample) # predicted output
result_predict = np.argmax(predict, axis = 2)
y_pred=np.array(result_predict).reshape(result_predict.size, 1)
print(y_pred) #print predicted output

for files, labels in test_sample:
  label_t = np.argmax(tf.convert_to_tensor(labels), axis = 2)
  y_true = np.array(label_t).reshape(label_t.size, 1) # true output
print(y_true) # print true label

for files, labels in test_sample_unbatch:  #unbatched test data so that it can be plotted
  files=np.asarray(files)
  for row in files:
    row=row.reshape(1,6).flatten()
    df_test_subset.append(row)

#Plotting data for first few samples of test data set
len_df=len(df_test_subset)
time=[1/float(Sampling_Freq) *j for j in range(len_df)]
legend_X='acc_X'
legend_Y='acc_Y'
legend_Z='acc_Z'
fig = plt.figure(figsize=(18,5))
# acc columns are the first 3 columns acc_X, acc_Y and acc_Z
X_component=[df_test_subset[i][0] for i in range(len_df) ] 
Y_component=[df_test_subset[i][1] for i in range(len_df) ] 
Z_component=[df_test_subset[i][2] for i in range(len_df) ]
plt.plot(time,X_component,color='y',label=legend_X)
plt.plot(time,Y_component,color='c',label=legend_Y)
plt.plot(time,Z_component,color='m',label=legend_Z)
plt.legend(loc="upper right")
plt.ylabel('Acceleration') 
plt.xlabel('Time in seconds (s)') 
title = 'Acc_TestData'
path = '/content/drive/My Drive/HAR_Images/' + title + '.png'
plt.savefig(path, bbox_inches='tight')
plt.show()
# gyro columns are the last 3 columns gyro_X, gyro_Y and gyro_Z
legend_X='gyro_X'
legend_Y='gyro_Y'
legend_Z='gyro_Z'
X_component=[df_test_subset[i][3] for i in range(len_df) ] 
Y_component=[df_test_subset[i][4] for i in range(len_df) ] 
Z_component=[df_test_subset[i][5] for i in range(len_df) ]
fig = plt.figure(figsize=(18,5))
plt.plot(time,X_component,color='y',label=legend_X)
plt.plot(time,Y_component,color='c',label=legend_Y)
plt.plot(time,Z_component,color='m',label=legend_Z)
plt.legend(loc="upper right")
title = 'Gyro_TestData'
plt.ylabel('Gyro Angular Velocity [rad/s]')
plt.xlabel('Time in seconds (s)') 
path = '/content/drive/My Drive/HAR_Images/' + title + '.png'
plt.savefig(path, bbox_inches='tight')
plt.show()

####Test Data Visualization ends here********************************************************************************
