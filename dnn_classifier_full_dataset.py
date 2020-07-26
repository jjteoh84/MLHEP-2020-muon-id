
# coding: utf-8

# In[1]:


import tensorflow as tf
print(tf.__version__)
from tensorflow import keras

import utils
import scoring

import os
import tempfile
import json

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import datetime

import sklearn
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


mpl.rcParams['figure.figsize'] = (12, 10)
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

EPOCHS = 1000
BATCH_SIZE = 32
SHUFFLE = True
REPEATITION=50


log_dir = "./logs/fit_100percent_Data/"

#output folder for saving checkpoint
outputFolder = './output_100percent_Data'
if not os.path.exists(outputFolder):
    os.makedirs(outputFolder)
    
get_ipython().magic('load_ext tensorboard')
get_ipython().system('rm -rf log_dir')


# In[2]:


columns = utils.SIMPLE_FEATURE_COLUMNS + ["id", "label", "weight"] #, "sWeight", "kinWeight"]
DATA_PATH = "~/share/data/I-coopetition-muon-id/"
train = pd.read_csv(os.path.join(DATA_PATH, "train.csv.gz"), index_col="id", usecols=columns)
test_df = pd.read_csv(os.path.join(DATA_PATH, "test-features.csv.gz"), index_col="id", usecols=utils.SIMPLE_FEATURE_COLUMNS + ["id"])


# columns = utils.SIMPLE_FEATURE_COLUMNS + ["id", "label", "weight"] #,  "kinWeight"]
# DATA_FOLDER = "~/share/data/1.6.2-boosting/"
# data = pd.read_csv(os.path.join(DATA_FOLDER, "train_1_percent.csv"), index_col="id", usecols=columns)


# new_key_label=[]
# for name in data.columns:
#     if '[' in name:
#         name = name.replace('[', '_').replace(']', '')

#     new_key_label.append(name)

# data.columns = new_key_label


# In[3]:


# data.head(5)


# In[4]:


# Use a utility from sklearn to split and shuffle our dataset.
## for full dataset: ~/share/data/I-coopetition-muon-id/
train_df, val_df = train_test_split(train, test_size=0.25, shuffle=True, random_state=2342234)

##for small dataset i.e  ~/share/data/1.6.2-boosting/
# train_df, test_df = train_test_split(data, test_size=0.2)
# train_df, val_df = train_test_split(train_df, test_size=0.2)



# train_ds = utils.df_to_dataset(train_df, shuffle=SHUFFLE, batch_size=BATCH_SIZE, repeatitions = REPEATITION)
# val_ds = utils.df_to_dataset(val_df, shuffle=SHUFFLE, batch_size=BATCH_SIZE, repeatitions = REPEATITION)
# test_ds = utils.df_to_dataset(test_df, shuffle=SHUFFLE, batch_size=BATCH_SIZE, repeatitions = REPEATITION)


# Form np arrays of labels and features.
train_labels = np.array(train_df.pop('label'))
val_labels = np.array(val_df.pop('label'))
# test_labels = np.array(test_df.pop('label'))


train_weights = np.array(train_df.pop('weight'))
val_weights = np.array(val_df.pop('weight'))
# test_weights = np.array(test_df.pop('weight'))


train_features = np.array(train_df)
val_features = np.array(val_df)
test_features = np.array(test_df)

# print(train_features)
# print(train_labels.shape)

# Normalize the input features using the sklearn StandardScaler. This will set the mean to 0 and standard deviation to 1.
# Note: The StandardScaler is only fit using the train_features to be sure the model is not peeking at the validation or test sets.

scaler = StandardScaler()
train_features = scaler.fit_transform(train_features)
val_features = scaler.transform(val_features)
test_features = scaler.transform(test_features)


train_ds = utils.make_ds(train_features, train_labels, train_weights, shuffle=SHUFFLE, batch_size=BATCH_SIZE, repeatitions = REPEATITION)
val_ds = utils.make_ds(val_features, val_labels, val_weights,  shuffle=SHUFFLE, batch_size=BATCH_SIZE, repeatitions = REPEATITION)
# test_ds = utils.make_ds(test_features, test_labels, test_weights, shuffle=SHUFFLE, batch_size=BATCH_SIZE, repeatitions = REPEATITION)

# val_ds = tf.data.Dataset.from_tensor_slices((val_features, val_labels))#.cache()
# val_ds = val_ds.batch(BATCH_SIZE).prefetch(2) 

print('Training labels shape:', train_labels.shape)
print('Validation labels shape:', val_labels.shape)
# print('Test labels shape:', test_labels.shape)

print('Training features shape:', train_features.shape)
print('Validation features shape:', val_features.shape)
print('Test features shape:', test_features.shape)
# print(test_ds)


# In[5]:




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


METRICS = [
#       keras.metrics.TruePositives(name='tp'),
#       keras.metrics.FalsePositives(name='fp'),
#       keras.metrics.TrueNegatives(name='tn'),
#       keras.metrics.FalseNegatives(name='fn'),
      keras.metrics.BinaryAccuracy(name='accuracy'),
      keras.metrics.Precision(name='precision'),
      keras.metrics.Recall(name='recall'),
      keras.metrics.AUC(name='auc'),
]

tf.keras.backend.set_floatx('float64')
def make_model(metrics = METRICS, output_bias=None):

  model = keras.Sequential([
#       keras.layers.Dense(1280, activation='relu', input_shape=(train_features.shape[-1],), name="layer1" ),
#       feature_layer,
      keras.layers.Dense(1280, activation='relu', name="layer1" ),
      keras.layers.Dense(640, activation='relu', name="layer2"),
      keras.layers.Dropout(.1, name="dropout1"),
      keras.layers.Dense(320, activation='relu', name="layer3"),
      keras.layers.Dropout(.05, name="dropout2"),
      keras.layers.Dense(160, activation='relu', name="layer4"),
      keras.layers.Dropout(.025, name="dropout3"),
      keras.layers.Dense(80, activation='relu', name="layer5"),
      keras.layers.Dropout(.0125, name="dropout4"),
      keras.layers.Dense(40, activation='relu', name="layer6"),
      keras.layers.Dense(20, activation='relu', name="layer7"),
      keras.layers.Dense(1, name="layer8"),
      keras.layers.Dense(1, activation='sigmoid',
                         bias_initializer=output_bias),
  ])

#   model.compile(optimizer='rmsprop',
#               loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
#               metrics=metrics)
    
  model.compile(
#       optimizer=keras.optimizers.Adam(lr=1e-3),
      optimizer='rmsprop',
      loss=keras.losses.BinaryCrossentropy(),
      metrics=metrics)

#   model.compile(optimizer='adam',
#                 loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
#                 metrics=['accuracy'])



  return model



RESTORE = False
latest_ckpt = tf.train.latest_checkpoint(outputFolder)
print('found latest checkpoint----: ', latest_ckpt)

latest_model_ckpt = utils.latest_saved_model(outputFolder)

if latest_model_ckpt is not None and RESTORE:
    print('.....loading model from checkpoint: ', latest_model_ckpt)
    model = tf.keras.models.load_model(latest_model_ckpt)
    model_history = model.history
elif latest_ckpt is not None and RESTORE:
    model = make_model()
    model.load_weights(latest_ckpt).assert_consumed()
    print("Restored from {}".format(latest_ckpt))
    model_history = model.history
else:
    model = make_model()


# model.summary()


# Save the weights using the `checkpoint_path` format
# model.save_weights(checkpoint_path.format(epoch=0, val_auc=0))



early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_auc',
    verbose=1,
    patience=20,
    mode='max',
    restore_best_weights=True)




checkpoint_path = outputFolder+"/model-{epoch:02d}_{val_auc:.4f}.ckpt"
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


tensorboard_callback = tf.keras.callbacks.TensorBoard( log_dir=log_dir + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
                                                histogram_freq=1,
                                                write_graph=True,
                                                write_images=False,
                                                update_freq='epoch',
                                                profile_batch=2,
                                                embeddings_freq=0,
                                                embeddings_metadata=None,
                                                )


# In[7]:


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
if not RESTORE:
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



# In[8]:


def plot_metrics(history):
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

# with open(outputFolder+"history.json", 'w') as fp:
# json.dumps(str(model.history) )
    
plot_metrics(model_history)


# In[9]:


train_predictions_baseline = model.predict(train_features, batch_size=BATCH_SIZE)
val_predictions_baseline = model.predict(val_features, batch_size=BATCH_SIZE)
test_predictions_baseline = model.predict(test_features, batch_size=BATCH_SIZE)


def plot_cm(labels, predictions, p=0.5):
  cm = confusion_matrix(labels, predictions > p)
  plt.figure(figsize=(5,5))
  sns.heatmap(cm, annot=True, fmt="d")
  plt.title('Confusion matrix @{:.2f}'.format(p))
  plt.ylabel('Actual label')
  plt.xlabel('Predicted label')

  print('Legitimate Transactions Detected (True Negatives): ', cm[0][0])
  print('Legitimate Transactions Incorrectly Detected (False Positives): ', cm[0][1])
  print('Fraudulent Transactions Missed (False Negatives): ', cm[1][0])
  print('Fraudulent Transactions Detected (True Positives): ', cm[1][1])
  print('Total Fraudulent Transactions: ', np.sum(cm[1]))


baseline_results = model.evaluate(val_features, val_labels,
                                  batch_size=BATCH_SIZE, verbose=0)

for name, value in zip(model.metrics_names, baseline_results):
  print(name, ': ', value)
# print(test_predictions_baseline)

plot_cm(val_labels, val_predictions_baseline)


# In[10]:


def plot_roc(name, labels, predictions, **kwargs):
  fp, tp, _ = sklearn.metrics.roc_curve(labels, predictions)

  plt.plot(100*fp, 100*tp, label=name, linewidth=2, **kwargs)
  plt.xlabel('False positives [%]')
  plt.ylabel('True positives [%]')
#   plt.xlim([-0.5,20])
#   plt.ylim([0,100.5])
  plt.grid(True)
  ax = plt.gca()
  ax.set_aspect('equal')


# In[11]:


plot_roc("Train Baseline", train_labels, train_predictions_baseline, color=colors[0])
# plot_roc("Test Baseline", test_labels, test_predictions_baseline, color=colors[1], linestyle='--')
plot_roc("Validation Baseline", val_labels, val_predictions_baseline, color=colors[2], linestyle='-.')

plt.legend(loc='lower right')


# In[27]:


# print(val_predictions_baseline)
# print(val_predictions_baseline.shape)
# val_predictions_baseline.flatten()
# print(val_predictions_baseline.flatten())
# print(val_labels)
# # val_weights = np.array(val_df.copy('weight'))
# # val_weights = val_df['weight'].copy().values
# # print(val_weights)

# print(len(val_predictions_baseline.flatten()))
# print(len(val_labels))
# print(len(val_weights))


# In[12]:


scoring.rejection90(val_labels, val_predictions_baseline.flatten(), sample_weight=val_weights)


# In[ ]:


# model.fit(train.loc[:, utils.SIMPLE_FEATURE_COLUMNS].values, train.label, sample_weight=train.kinWeight.values)


# In[13]:


# predictions = model.predict_proba(test.loc[:, utils.SIMPLE_FEATURE_COLUMNS].values)[:, 1]
test_predictions = model.predict(test_features, batch_size=BATCH_SIZE)


# In[15]:



compression_opts = dict(method='zip',
                        archive_name='submission.csv')  
pd.DataFrame(data={"prediction": test_predictions.flatten()}, index=test_df.index).to_csv(
    "submission.zip", index_label=utils.ID_COLUMN, compression=compression_opts)


# In[16]:


submission = pd.read_csv("./submission.zip")
submission.head(5)

