import tensorflow as tf

from tensorflow import keras
import kerastuner as kt

import utils
import scoring

import os, sys
import tempfile
import json
import argparse
from argparse import ArgumentParser
from tqdm import tqdm

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import time
from datetime import timedelta, datetime

import sklearn
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# from gtk.keysyms import minutes

# Global Configuration
#_____________________________________________________________________________

# # Set CPU as available physical device
my_devices = tf.config.experimental.list_physical_devices(device_type='CPU')
tf.config.experimental.set_visible_devices(devices= my_devices, device_type='CPU')

physical_devices = tf.config.list_physical_devices('CPU')
print("Num CPUs:", len(physical_devices))

# 
# # To find out which devices your operations and tensors are assigned to
# tf.debugging.set_log_device_placement(True)


mpl.rcParams['figure.figsize'] = (12, 10)
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

EPOCHS = 1000
BATCH_SIZE = 32
SHUFFLE = True
REPEATITION=20

# print(tf.__version__)
# print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    
# get_ipython().magic('load_ext tensorboard')
# get_ipython().system('rm -rf log_dir')




def main(commandLine=None):
    """ Main function"""
    parser = ArgumentParser()
    # General switches
    
    parser.add_argument('-d', '--debug', help='Turn on debug output', action='store_true')
    
    parser.add_argument('-tr', '--trainning', help='Run in the training mode', action='store_true')
    parser.add_argument('-it', '--inputTestFile', help='Input testing dataset', action='store_true')
    
    parser.add_argument('-o', '--outputPath', help='Output path for pdf/jpg/submission', default='./output/' + datetime.now().strftime("%Y_%m_%d-%H_%M_%S"))
    
    parser.add_argument('-r', '--restore', help='Restrore model/weight from checkpoint', action='store_true')
    parser.add_argument('-rckp', '--restoreFromCkpt', help='Path to restrore model/weight from checkpoint ', default='./ckpt')
    parser.add_argument('-rr', '--retrain', help='Restrore model/weight from checkpoint and retrain', action='store_true')
    
    parser.add_argument('-c', '--checkPointPath', help='Output path for checkpoint', default='./ckpt')
    parser.add_argument('-l', '--tbLogPath', help='Output path for tensorboard', default='./logs')
    
    parser.add_argument('-pr', '--printData', help='Print dataset', action='store_false')
    
    parser.add_argument('-ev', '--evaluate', help='Run in the evaluate mode with validation dataset', action='store_false')
#     parser.add_argument('-et', '--evaluate', help='Run in the evaluate mode with test dataset', action='store_false')
    
    
#     parser.add_argument('-pt', '--evaluate', help='Run in the prediction mode with test dataset', action='store_false')
    
    parser.add_argument('-s', '--submit', help='Generate submission files', action='store_true')
    
    parser.add_argument('-tu', '--tune', help='HyperParameter tuning', action='store_true')
    parser.add_argument('-v', '--version', help='output subffix', default='')
    
    
    start = time.time()
    tqdm.write("Start time: %s (Wall clock time)" % datetime.now())
    
    opts = None
    if commandLine:
        opts = parser.parse_args(commandLine)
    else:
        opts = parser.parse_args()
    
       
    opts.outputPath = os.path.abspath(opts.outputPath)
    print(opts.outputPath)
    if not os.path.exists(opts.outputPath):
        os.makedirs(opts.outputPath)   
        
    opts.tbLogPath = os.path.abspath(opts.outputPath + '/' + opts.tbLogPath)
    if not os.path.exists(opts.tbLogPath):
        os.makedirs(opts.tbLogPath)  
        
    opts.checkPointPath = os.path.abspath(opts.outputPath + '/' + opts.checkPointPath)
    if not os.path.exists(opts.checkPointPath):
        os.makedirs(opts.checkPointPath) 
        
    
    
    readStartTime = time.time()
    
    use_columns = utils.BEST_FEATURE_COLUMNS 
#     use_columns = utils.SIMPLE_FEATURE_COLUMNS
    
    
    columns = use_columns + ["id", "label", "weight"] #, "sWeight", "kinWeight"]
    DATA_PATH = "/data/atlas/users/jjteoh/mlhep2020_muID/"
    train = pd.read_csv(os.path.join(DATA_PATH, "train.csv.gz"), index_col="id", usecols=columns)
#     train = pd.read_csv(os.path.join(DATA_PATH, "train_1_percent.csv"), index_col="id", usecols=columns)
    

    
    
    testHasLable = False 
    
    if opts.inputTestFile :
        print('loading dedicated test file.......')
        train_df, val_df = train_test_split(train, test_size=0.25, shuffle=True, random_state=2342234)
        test_df = pd.read_csv(os.path.join(DATA_PATH, "test-features.csv.gz"), index_col="id", usecols=use_columns + ["id"])       
    else:
        train_df, test_df = train_test_split(train, test_size=0.2)
        train_df, val_df = train_test_split(train_df, test_size=0.2)
        testHasLable = True

    if opts.printData:
        train.head(5)

#     print('testhaslabel: ---- ', testHasLable)
    readTime = time.time() - readStartTime
       


    preprocessingStartTime = time.time()
# train_ds = utils.df_to_dataset(train_df, shuffle=SHUFFLE, batch_size=BATCH_SIZE, repeatitions = REPEATITION)
# val_ds = utils.df_to_dataset(val_df, shuffle=SHUFFLE, batch_size=BATCH_SIZE, repeatitions = REPEATITION)
# test_ds = utils.df_to_dataset(test_df, shuffle=SHUFFLE, batch_size=BATCH_SIZE, repeatitions = REPEATITION)


    # Form np arrays of labels and features.
    train_labels = np.array(train_df.pop('label'))
    val_labels = np.array(val_df.pop('label'))
    if testHasLable: test_labels = np.array(test_df.pop('label'))
    
    
    train_weights = np.array(train_df.pop('weight'))
    val_weights = np.array(val_df.pop('weight'))
    if testHasLable: test_weights = np.array(test_df.pop('weight'))
    
    
    train_features = np.array(train_df, dtype='float32')
    val_features = np.array(val_df, dtype='float32')
    test_features = np.array(test_df, dtype='float32')

    
# Normalize the input features using the sklearn StandardScaler. This will set the mean to 0 and standard deviation to 1.
# Note: The StandardScaler is only fit using the train_features to be sure the model is not peeking at the validation or test sets.

    scaler = StandardScaler()
    train_features = scaler.fit_transform(train_features)
    val_features = scaler.transform(val_features)
    test_features = scaler.transform(test_features)
    
    
    train_ds = utils.make_ds(train_features, train_labels, weights=None, shuffle=SHUFFLE, batch_size=BATCH_SIZE, repeatitions = REPEATITION)
    val_ds = utils.make_ds(val_features, val_labels, weights=None,  shuffle=SHUFFLE, batch_size=BATCH_SIZE, repeatitions = REPEATITION)
    if testHasLable: test_ds = utils.make_ds(test_features, test_labels, weights=None, shuffle=SHUFFLE, batch_size=BATCH_SIZE, repeatitions = REPEATITION)
#     print(val_ds)
#     return

# val_ds = tf.data.Dataset.from_tensor_slices((val_features, val_labels))#.cache()
# val_ds = val_ds.batch(BATCH_SIZE).prefetch(2) 

    print('Training labels shape:', train_labels.shape)
    print('Validation labels shape:', val_labels.shape)
    if testHasLable: print('Test labels shape:', test_labels.shape)
    
    print('Training features shape:', train_features.shape)
    print('Validation features shape:', val_features.shape)
    print('Test features shape:', test_features.shape)


    preprocessingTime = time.time() - preprocessingStartTime
     


# for feature_batch, label_batch in train_ds.take(1):
#   print('Every feature:', list(feature_batch.keys()))
#   print('A batch of PT:', feature_batch['PT'])
#   print('A batch of targets:', label_batch )

# feature_columns = []
# feature_batch, label_batch = next(iter(train_ds))

# for header in list(feature_batch.keys()):
#     feature_columns.append(tf.feature_column.numeric_column(header))


# # print(feature_columns)
# feature_layer = tf.keras.layers.DenseFeatures(feature_columns, dtype='float64')


# In[5]:


    
#     tf.keras.backend.set_floatx('float64')
    strategy = tf.distribute.MirroredStrategy()
    
    latest_ckpt = None
    latest_model_ckpt = None
    model = None
    if opts.restore:

        latest_ckpt = tf.train.latest_checkpoint(opts.restoreFromCkpt)
        print('found latest checkpoint----: ', latest_ckpt)
        
        latest_model_ckpt = utils.latest_saved_model(opts.restoreFromCkpt)
    
    if latest_model_ckpt is not None and opts.restore:
        print('.....loading model from checkpoint: ', latest_model_ckpt)
        model = tf.keras.models.load_model(latest_model_ckpt)
        model_history = model.history
    elif latest_ckpt is not None and opts.restore:
        model = make_model(strategy)
        model.load_weights(latest_ckpt).assert_consumed()
        print("Restored from {}".format(latest_ckpt))
        model_history = model.history
    elif not opts.tune:
        model = make_model(strategy)


    hpTuningStartTime = time.time()
    tuner = None
    if opts.tune:
        tuner = kt.Hyperband(make_model_hp_tunning,
                     objective = kt.Objective("val_auc", direction="max"), 
                     max_epochs = 10,
                     factor = 3,
                     directory = opts.outputPath + '/HPtuning',
                     project_name = 'hp_tuning')   

# model.summary()
        tuner.search(train_ds, epochs = 30, validation_data = val_ds, 
                        callbacks=[ tf.keras.callbacks.EarlyStopping(monitor='val_auc',patience=5, restore_best_weights=True)]
                        )

        # Get the optimal hyperparameters
        best_hps = tuner.get_best_hyperparameters(num_trials = 1)[0]

        print('The hyperparameter search is complete. The optimal number of units in the first densely-connected layer are:')
        print('layer1: ' , best_hps.get('units1'))
        print('layer2: ' , best_hps.get('units2'))
        print('layer3: ' , best_hps.get('units3'))
        print('layer4: ' , best_hps.get('units4'))
#         print('layer5: ' , best_hps.get('units5'))
#         print('layer6: ' , best_hps.get('units6'))
#         print('layer7: ' , best_hps.get('units7'))
        print('The optimal learning rate: ' , best_hps.get('learning_rate'))
        print('')
        
        
        model = tuner.hypermodel.build(best_hps)

    hpTuningTime = time.time() - hpTuningStartTime    


# Save the weights using the `checkpoint_path` format
# model.save_weights(checkpoint_path.format(epoch=0, val_auc=0))



    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_auc',
        verbose=1,
        patience=30,
        mode='max',
        restore_best_weights=True)




    checkpoint_path = opts.checkPointPath+"/model-{epoch:02d}_{val_auc:.4f}.ckpt"
    # /model-{epoch:02d}-{val_auc:.2f}.ckpt"

    # Create a callback that saves the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                     monitor='val_auc',
                                                     save_weights_only=False,
                                                     verbose=1,
                                                     mode='max',
                                                     save_best_only=True)
    
    cp_callback_weightOnly = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                     monitor='val_auc',
                                                     save_weights_only=True,
                                                     verbose=1,
                                                     mode='max',
                                                     save_best_only=True)
    
    
    tensorboard_callback = tf.keras.callbacks.TensorBoard( log_dir=opts.tbLogPath + '/' + datetime.now().strftime("%Y_%m_%d-%H_%M_%S"),
                                                    histogram_freq=1,
                                                    write_graph=True,
                                                    write_images=False,
                                                    update_freq='epoch',
                                                    profile_batch=2,
                                                    embeddings_freq=0,
                                                    embeddings_metadata=None,
                                                )


    trainingStartTime = time.time()
    
    
     
    model_history = None    

###TO-DO Write call back to save training history for every epoch
#https://github.com/tensorflow/tensorflow/issues/27861
#https://stackoverflow.com/questions/50127527/how-to-save-training-history-on-every-epoch-in-keras

# if (latest_ckpt is None  and latest_model_ckpt is  None) :
# model_history = model.fit(
#     train_features,
#     train_labels,
#     batch_size=BATCH_SIZE,
#     steps_per_epoch = 20,
#     epochs=EPOCHS,
#     shuffle=True,
#     callbacks = [early_stopping, cp_callback, tensorboard_callback],
#     validation_data=val_ds)
        #     validation_data=(val_features, val_labels))

# if (latest_ckpt is None  and latest_model_ckpt is  None and not RESTORE) :
    
    if not opts.restore or opts.retrain:
        print('start training ------')
        model_history = model.fit(
        train_ds,
    #     batch_size=BATCH_SIZE,
        steps_per_epoch = 50,
        epochs=EPOCHS,
        shuffle=False,
        callbacks = [early_stopping, cp_callback, tensorboard_callback],
        validation_data=val_ds)
        
    
# model.fit(train_features,
#     train_labels,
#           validation_data=val_ds,
#           epochs=10)
# loss, accuracy = model.evaluate(test_ds)
# print("Accuracy", accuracy)


    trainingTime = time.time() - trainingStartTime
    



# with open(outputFolder+"history.json", 'w') as fp:
# json.dumps(str(model.history) )
    if model_history is not None:
        plot_metrics(model_history, opts.outputPath)


# In[9]:
    predict_eval_StartTime = time.time()

    train_predictions_baseline = model.predict(train_features, batch_size=BATCH_SIZE)
    val_predictions_baseline = model.predict(val_features, batch_size=BATCH_SIZE)
    test_predictions_baseline = model.predict(test_features, batch_size=BATCH_SIZE)





    baseline_results = model.evaluate(val_features, val_labels,
                                      batch_size=BATCH_SIZE, verbose=0)
    
    
    predict_eval_Time = time.time() - predict_eval_StartTime
    
    for name, value in zip(model.metrics_names, baseline_results):
      print(name, ': ', value)
    # print(test_predictions_baseline)
    
    plot_cm(val_labels, val_predictions_baseline, opts.outputPath)


# In[10]:





# In[11]:


    plot_roc("Train Baseline", train_labels, train_predictions_baseline, opts.outputPath, color=colors[0])    
    plot_roc("Validation Baseline", val_labels, val_predictions_baseline, opts.outputPath, color=colors[2], linestyle='-.')
    if testHasLable: 
        plot_roc("Test Baseline", test_labels, test_predictions_baseline, opts.outputPath, color=colors[1], linestyle='--')
    
    


    print('')
    
    rejection90 = scoring.rejection90(val_labels, val_predictions_baseline.flatten(), sample_weight=None)
    print('----------scoring-----rejection@90=  ', rejection90)
    print('')

    test_predictions = model.predict(test_features, batch_size=BATCH_SIZE)



    if opts.submit:
        tqdm.write("Preparing submission file......." )
        compression_opts = dict(method='zip',
                                archive_name='submission.csv')  
        pd.DataFrame(data={"prediction": test_predictions.flatten()}, index=test_df.index).to_csv(
            opts.outputPath+"/submission.zip", index_label=utils.ID_COLUMN, compression=compression_opts)
        
        
        # In[16]:
        
        
        submission = pd.read_csv(opts.outputPath + "/submission.zip")
        print(submission.head(5))

    
    tqdm.write("End time: %s (Wall clock time)" % datetime.now())
    execTime = time.time() - start
    
    tqdm.write("Reading input took: %s secs (Wall clock time)" % timedelta(seconds=round(readTime))) 
    tqdm.write("Data preprocessing took: %s secs (Wall clock time)" % timedelta(seconds=round(preprocessingTime)))
       
    tqdm.write("HP tuning took: %s secs (Wall clock time)" % timedelta(seconds=round(hpTuningTime)))
    tqdm.write("Training took: %s secs (Wall clock time)" % timedelta(seconds=round(trainingTime)))
    tqdm.write("Prediction & evualation took: %s secs (Wall clock time)" % timedelta(seconds=round(predict_eval_Time)))
    
    tqdm.write("Total execution took: %s secs (Wall clock time)" % timedelta(seconds=round(execTime)))


# class ClearTrainingOutput(tf.keras.callbacks.Callback):
#   def on_train_end(*args, **kwargs):
#     IPython.display.clear_output(wait = True)


#_____________________________________________________________________________
def make_model(strategy , output_bias=None):
    
    with strategy.scope():
#         tf.keras.backend.set_floatx('float64')
        METRICS = [
          keras.metrics.BinaryAccuracy(name='accuracy'),
          keras.metrics.Precision(name='precision'),
          keras.metrics.Recall(name='recall'),
          keras.metrics.AUC(name='auc'),
          ]
        
        model = keras.Sequential([
    #       keras.layers.Dense(1280, activation='relu', input_shape=(train_features.shape[-1],), name="layer1" ),
    #       feature_layer,
#         keras.layers.Dense(1280, activation='relu',  name="layer1"),
#         keras.layers.Dropout(.5, name="dropout1"),
#         keras.layers.Dense(640, activation='relu',  name="layer2"),
#         keras.layers.Dropout(.5, name="dropout2"),
#         keras.layers.Dense(320, activation='relu',  name="layer3"),
#         keras.layers.Dropout(.5, name="dropout3"),
#         keras.layers.Dense(160, activation='relu',  name="layer4"),
#         keras.layers.Dropout(.5, name="dropout4"),
#         keras.layers.Dense(80, activation='relu',  name="layer5"),
#         keras.layers.Dropout(.5, name="dropout5"),
#         keras.layers.Dense(40, activation='relu',  name="layer6"),
#         keras.layers.Dropout(.5, name="dropout6"),
#         keras.layers.Dense(20, activation='relu',  name="layer7"),
#         keras.layers.Dropout(.5, name="dropout1"),
#         keras.layers.Dense(1,  name="layer8"),
#         keras.layers.Dense(1, activation='sigmoid',
#                            bias_initializer=output_bias),

              keras.layers.Dense(units = 500, activation='relu', kernel_regularizer=keras.regularizers.l2(0.005),  name="layer1"),
              keras.layers.Dropout(.2, name="dropout1"),
              keras.layers.Dense(units = 100, activation='relu', kernel_regularizer=keras.regularizers.l2(0.005),  name="layer2"),
              keras.layers.Dropout(.2, name="dropout2"),
              keras.layers.Dense(units = 20, activation='relu', kernel_regularizer=keras.regularizers.l2(0.005),  name="layer3"),
              keras.layers.Dropout(.125, name="dropout3"),
              keras.layers.Dense(units = 5, activation='relu', kernel_regularizer=keras.regularizers.l2(0.005),  name="layer4"),
              keras.layers.Dropout(.1, name="dropout4"),
        #       keras.layers.Dense(1, activation='relu',  name="layer8"),
              keras.layers.Dense(1, activation='sigmoid')
          ])
      
    model.compile(
       tf.keras.optimizers.RMSprop(learning_rate=0.0005),
       loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
       metrics=METRICS)
    
#     model.compile(
# #         optimizer=keras.optimizers.Adam(lr=1e-3),
#         optimizer='Adadelta',
#         loss=keras.losses.BinaryCrossentropy(),
#         metrics=METRICS )

#     model.compile(optimizer='adam',
#                     loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
#                     metrics=METRICS 
#     )

    return model


#_____________________________________________________________________________
def make_model_hp_tunning(hp, output_bias=None):
    
    
#         tf.keras.backend.set_floatx('float64')
    METRICS = [
      keras.metrics.BinaryAccuracy(name='accuracy'),
      keras.metrics.Precision(name='precision'),
      keras.metrics.Recall(name='recall'),
      keras.metrics.AUC(name='auc'),
      ]
    
    hp_units1 = hp.Int('units1', min_value = 100, max_value = 500, step = 32)
    hp_units2 = hp.Int('units2', min_value = 20, max_value = 100, step = 32)
    hp_units3 = hp.Int('units3', min_value = 10, max_value = 50, step = 32)
    hp_units4 = hp.Int('units4', min_value = 1, max_value = 20, step = 32)   

    model = keras.Sequential([
   

     keras.layers.Dense(units = hp_units1, activation='relu',  name="layer1"),
     keras.layers.Dropout(.5, name="dropout1"),
     keras.layers.Dense(units = hp_units2, activation='relu',  name="layer2"),
     keras.layers.Dropout(.5, name="dropout2"),
     keras.layers.Dense(units = hp_units3, activation='relu',  name="layer3"),
     keras.layers.Dropout(.5, name="dropout3"),
     keras.layers.Dense(units = hp_units4, activation='relu',  name="layer4"),
     keras.layers.Dropout(.5, name="dropout4"),
    #       keras.layers.Dense(1, activation='relu',  name="layer8"),
     keras.layers.Dense(1, activation='sigmoid',
                        bias_initializer=output_bias),
     ])
    
#       hp_units1 = hp.Int('units1', min_value = 500, max_value = 1000, step = 32)
#     hp_units2 = hp.Int('units2', min_value = 200, max_value = 500, step = 32)
#     hp_units3 = hp.Int('units3', min_value = 100, max_value = 400, step = 32)
#     hp_units4 = hp.Int('units4', min_value = 50, max_value = 160, step = 32)   
#     hp_units5 = hp.Int('units5', min_value = 30, max_value = 80, step = 32) 
#     hp_units6 = hp.Int('units6', min_value = 10, max_value = 40, step = 32) 
#     hp_units7 = hp.Int('units7', min_value = 5, max_value = 20, step = 32) 


#     model = keras.Sequential([
# #       keras.layers.Dense(1280, activation='relu', input_shape=(train_features.shape[-1],), name="layer1" ),
# #       feature_layer,
# #       keras.layers.Dense(units = hp_units1, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001), name="layer1"),
#       keras.layers.Dense(units = hp_units1, activation='relu',  name="layer1"),
#       keras.layers.Dense(units = hp_units2, activation='relu',  name="layer2"),
#       keras.layers.Dropout(.5, name="dropout1"),
#       keras.layers.Dense(units = hp_units3, activation='relu',  name="layer3"),
#       keras.layers.Dropout(.5, name="dropout2"),
#       keras.layers.Dense(units = hp_units4, activation='relu',  name="layer4"),
#       keras.layers.Dropout(.5, name="dropout3"),
#       keras.layers.Dense(units = hp_units5, activation='relu',  name="layer5"),
#       keras.layers.Dropout(.5, name="dropout4"),
# #       keras.layers.Dense(1, activation='relu',  name="layer8"),
#       keras.layers.Dense(1, activation='sigmoid',
#                          bias_initializer=output_bias),
#       ])


     # Tune the learning rate for the optimizer 
     # Choose an optimal value from 0.01, 0.001, or 0.0001
    hp_learning_rate = hp.Choice('learning_rate', values = [0.0005, 0.001, 0.0015, 0.002] ) #[1e-2, 1e-3, 1e-4]
    
    model.compile(
       tf.keras.optimizers.RMSprop(learning_rate=hp_learning_rate),
       loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
       metrics=METRICS)
    
#     model.compile(
#         optimizer=keras.optimizers.Adam(lr=1e-3),
#         loss=keras.losses.BinaryCrossentropy(),
#         metrics=METRICS )

#   model.compile(optimizer='adam',
#                 loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
#                 metrics=['accuracy']
# )

    return model
#_____________________________________________________________________________ 
def plot_metrics(history, outputPath):
    metrics =  ['loss', 'auc', 'precision', 'recall']
    for n, metric in enumerate(metrics):
        name = metric.replace("_"," ").capitalize()
        plt.subplot(2,2,n+1)
        plt.plot(history.epoch,  history.history[metric], color=colors[0], label='Train')
        plt.plot(history.epoch, history.history['val_'+metric],
                 color=colors[0], linestyle="--", label='Val')
        plt.xlabel('Epoch')
        plt.ylabel(name)
        if metric == 'loss':
          plt.ylim([0, plt.ylim()[1]])
        elif metric == 'auc':
          plt.ylim([0.5,1])
        else:
          plt.ylim([0,1])
        
        plt.legend()
    plt.savefig(outputPath +'/'+ metric+'.jpg')

#_____________________________________________________________________________ 
def plot_cm(labels, predictions,  outputPath = ".", p=0.5):
    cm = confusion_matrix(labels, predictions > p)
    plt.figure(figsize=(5,5))
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title('Confusion matrix @{:.2f}'.format(p))
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    plt.savefig(outputPath + '/Confusion_matrix.jpg')
    
    print('Legitimate Transactions Detected (True Negatives): ', cm[0][0])
    print('Legitimate Transactions Incorrectly Detected (False Positives): ', cm[0][1])
    print('Fraudulent Transactions Missed (False Negatives): ', cm[1][0])
    print('Fraudulent Transactions Detected (True Positives): ', cm[1][1])
    print('Total Fraudulent Transactions: ', np.sum(cm[1]))
  
  
    
#_____________________________________________________________________________  
def plot_roc(name, labels, predictions, outputPath, **kwargs):
    fp, tp, _ = sklearn.metrics.roc_curve(labels, predictions)
    plt.figure(200)
    plt.plot(100*fp, 100*tp, label=name, linewidth=2, **kwargs)
    plt.xlabel('False positives [%]')
    plt.ylabel('True positives [%]')
    #   plt.xlim([-0.5,20])
    #   plt.ylim([0,100.5])
    plt.grid(True)
    plt.legend(loc='lower right')
    ax = plt.gca()
    ax.set_aspect('equal')
    plt.savefig(outputPath + '/ROC.jpg')
    

  
#_____________________________________________________________________________
if __name__ == '__main__':
    main()
    #EOF