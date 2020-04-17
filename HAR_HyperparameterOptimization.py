#HyperParameter Optimization starts here **********************************************************************

%load_ext tensorboard

!rm -rf ./logs/ 
import tensorflow as tf
from tensorboard.plugins.hparams import api as hp

#Set hyperparameters as optimizer, number of LSTM neurons for 1st RNN layer, drouput rate  
HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['adam', 'RMSProp']))
HP_LSTM_NEURONS = hp.HParam('LSTMNeurons', hp.Discrete([128,256]))
HP_DROPOUT = hp.HParam('dropout', hp.RealInterval(0.1, 0.2))

#Choose metrics as accuracy
METRIC_ACCURACY = 'accuracy'

#Create logs to log mertics for different hyperparams
with tf.summary.create_file_writer('logs/hparam_tuning').as_default():
  hp.hparams_config(
    hparams=[HP_OPTIMIZER, HP_LSTM_NEURONS, HP_DROPOUT],
    metrics=[hp.Metric(METRIC_ACCURACY, display_name='Accuracy')],
  )

#Define model with the hyperparameters
def HP_mdl():
  inputs = keras.Input(shape=(N_WindowSize,N_features)) 
  x = layers.LSTM(hparams[HP_LSTM_NEURONS],return_sequences=True)(inputs)
  x = layers.Dropout(hparams[HP_DROPOUT])(x)
  x = layers.LSTM(128,return_sequences=True)(x)
  x = layers.Dense(64,activation='relu')(x)
  x = layers.Dropout(hparams[HP_DROPOUT])(x)
  x = layers.Dense(32,activation='relu')(x)
  outputs = layers.Dense(12,activation='softmax')(x)
  mdl = keras.Model(inputs=inputs, outputs=outputs, name='DR_model')
  return mdl

#Function to write into the logs
#Input: file name, hyperparameter used
def run(run_dir, hparams):
  with tf.summary.create_file_writer(run_dir).as_default():
    hp.hparams(hparams) 
    accuracy = train_test_model(hparams)
    tf.summary.scalar(METRIC_ACCURACY, accuracy, step=1)

N_epochs = 5

def train_test_model(hparams):
  mdl = HP_mdl()
  opt = hparams[HP_OPTIMIZER]
  mdl.compile(loss=tf.keras.losses.categorical_crossentropy,
              optimizer=opt,
              metrics=['accuracy'])
  #Train model
  history = mdl.fit(train_ds, epochs = N_epochs,
                    validation_data = validate_ds)
  _, accuracy = mdl.evaluate(test_ds)
  return accuracy

#In each session write into the logs for different combinations of various hyperparameters
session_num = 0
for optimizer in HP_OPTIMIZER.domain.values:
  for lstm_neurons in HP_LSTM_NEURONS.domain.values:
    for dropout_rate in (HP_DROPOUT.domain.min_value, HP_DROPOUT.domain.max_value):
      hparams = {
          HP_OPTIMIZER: optimizer,
          HP_LSTM_NEURONS: lstm_neurons,
          HP_DROPOUT:dropout_rate, 
      }
      run_name = "run-%d" % session_num
      print('--- Starting trial: %s' % run_name)
      print({h.name: hparams[h] for h in hparams})
      run('logs/hparam_tuning/' + run_name, hparams)
      session_num += 1
#*Hyperparameter Optimization ends************************************************************************************
