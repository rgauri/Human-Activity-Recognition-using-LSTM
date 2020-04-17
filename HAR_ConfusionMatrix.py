#Confusion Matrix and ROC Curves***************************************************************************

y_pred = []
y_true = []

#Get predicted labels for entire test dataset
for files, labels in test_ds.take(-1):
    result_predict = mdl.predict(files)
    result_predict = np.asarray(result_predict)
    # print(result_predict.shape)
    # print(tf.convert_to_tensor(result_predict[0][0]))
    result_predict = np.argmax(result_predict, axis=2)
    y_pred.append(np.array(result_predict).reshape(result_predict.size, 1))
    # print(np.asarray(y_pred).size)
    # print(labels[0][0])
    label_t = np.argmax(tf.convert_to_tensor(labels), axis=2)
    y_true.append(np.array(label_t).reshape(label_t.size, 1)) #true label
    # print(np.asarray(y_true).size)

y_pred = np.asarray(y_pred)
y_true = np.asarray(y_true)

y_pred_concat = []
y_true_concat = []

#Create numpy array of predicted and true labels
for index in range(0, len(y_pred)):

  if (index == 0):
    y_pred_concat = y_pred[index]
    y_true_concat = y_true[index]

  else:
    y_pred_concat = np.concatenate((y_pred_concat, y_pred[index]), axis=0)
    y_true_concat = np.concatenate((y_true_concat, y_true[index]), axis=0)

print(len(y_pred_concat))
print(len(y_true_concat))

#*Implementation 1******************************************************************************************************
import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
import seaborn as sns
import math
from sklearn.metrics import classification_report, confusion_matrix

labels=['WALKING',
'WALKING_UPSTAIRS',
'WALKING_DOWNSTAIRS',
'SITTING',
'STANDING',
'LAYING',
'STAND_TO_SIT',
'SIT_TO_STAND',
'SIT_TO_LIE',
'LIE_TO_SIT',
'STAND_TO_LIE',
'LIE_TO_STAND'
]

#Confusion matrix based on true labels and predicted lables
confusion_matrix = metrics.confusion_matrix(y_true_concat, y_pred_concat)
#Normalized confusion matrix obtained by dividing every element by sum of all elements in confusion matrix
normalised_confusion_matrix = np.array(confusion_matrix, dtype=np.float32)/np.sum(confusion_matrix) * 100

#Plot confusion matrix
plt.figure(figsize=(16, 14))
sns.heatmap(confusion_matrix, xticklabels=labels, yticklabels=labels, annot=True, fmt="d");
plt.title("Confusion matrix")
plt.ylabel('True label')
plt.xlabel('Predicted label')
title = 'HAR Confusion matrix'
path = '/content/drive/My Drive/HAR_Images/' + title + '.png'
plt.savefig(path, bbox_inches='tight')
plt.show();

#Plot normalized confusion matrix
plt.figure(figsize=(16, 14))
sns.heatmap(normalised_confusion_matrix, xticklabels=labels, yticklabels=labels, annot=True, fmt='0.2g' );
plt.title("Normalized Confusion matrix")
plt.ylabel('True label')
plt.xlabel('Predicted label')
title = 'HAR Normalized Confusion matrix'
path = '/content/drive/My Drive/HAR_Images/' + title + '.png'
plt.savefig(path, bbox_inches='tight')
plt.show();

print(classification_report(y_true_concat, y_pred_concat, target_names=labels))
#*Implementation 1 ends here********************************************************************************************

#*Implementation 2******************************************************************************************************

import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn import metrics

labels=['WALKING',
'WALKING_UPSTAIRS',
'WALKING_DOWNSTAIRS',
'SITTING',
'STANDING',
'LAYING',
'STAND_TO_SIT',
'SIT_TO_STAND',
'SIT_TO_LIE',
'LIE_TO_SIT',
'STAND_TO_LIE',
'LIE_TO_STAND'
]

#Function to plot normalized confusion matrix from confusion matrix
#Input: confusion matrix, number of classes, title, color map 
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# confusion matrix
cm_normalize=True
print_cm=True
cm_cmap  =plt.cm.Greens

cm = metrics.confusion_matrix(y_true_concat, y_pred_concat)
#results['confusion_matrix'] = cm
if print_cm:
    print('--------------------')
    print('| Confusion Matrix |')
    print('--------------------')
    print('\n {}'.format(cm))

# plot confusin matrix
plt.figure(figsize=(8,8))
plt.grid(b=False)
plot_confusion_matrix(cm, classes=labels, normalize=True, title='Normalized confusion matrix', cmap = cm_cmap)
plt.show()
#*Implementation 2 ends here**************************************************************************************

#* ROC AUC Curve Implementation**************************************************************************************

from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import roc_auc_score

labels=['WALKING',           
'WALKING_UPSTAIRS',  
'WALKING_DOWNSTAIRS',
'SITTING',           
'STANDING',          
'LAYING',            
'STAND_TO_SIT',      
'SIT_TO_STAND',      
'SIT_TO_LIE',        
'LIE_TO_SIT',        
'STAND_TO_LIE',      
'LIE_TO_STAND'      
]

#Function to calculate Region of Convergence based on true and predicted labels
#Input: True label, Predicted label
#Output: ROC curve
def multiclass_roc_auc_score(y_test, y_pred, average="macro"):
    lb = LabelBinarizer()
    lb.fit(y_test)
    y_test = lb.transform(y_test)
    y_pred = lb.transform(y_pred)

    for (idx, c_label) in enumerate(labels): # all_labels: no of the labels
        fpr, tpr, thresholds = roc_curve(y_test[:,idx].astype(int), y_pred[:,idx])
        plt.plot(fpr, tpr, label = '%s (AUC:%0.2f)'  % (c_label, auc(fpr, tpr)))
    
    title = 'HAR ROC Curve'
    path = '/content/drive/My Drive/HAR_Images/' + title + '.png'
    plt.savefig(path, bbox_inches='tight')
    plt.plot(fpr, fpr, 'b-', label = 'Random Guessing')
    return roc_auc_score(y_test, y_pred, average=average)

ROC = multiclass_roc_auc_score(y_true_concat, y_pred_concat)
print(ROC)
#Confusion Matrix and ROC ends here****************************************************************************
