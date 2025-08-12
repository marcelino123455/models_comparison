from bert_sklearn import BertClassifier
from bert_sklearn import BertRegressor
from bert_sklearn import load_model
from sklearn import metrics
from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    classification_report, accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix
)

import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
csv_path = "../../data/spotify_dataset_sin_duplicados_4.csv"
RANDOM_STATE = 42
A = ['text', 'song', 'Artist(s)', 'Album', 'Similar Artist 1', 'Genre']


TESTING = False
COL_TF_IDF = A


if TESTING:
    _SAMPLE_SIZE = 100
else:
    _SAMPLE_SIZE = None



def load_data(csv_path, sample_size=100, columns_tf_idfizable = ['text']):
    df = pd.read_csv(csv_path)
    if sample_size and sample_size < len(df):
        sampled_idx = df.sample(n=sample_size, random_state=RANDOM_STATE).index
        df = df.loc[sampled_idx].reset_index(drop=True)
    df['text'] = df[columns_tf_idfizable].fillna('').agg(' '.join, axis=1)
    # df['label'] = (df['Explicit'].str.lower() == 'yes').astype(int)
    df['label'] = df['Explicit']
    print(df["label"].value_counts())

 


    print(df['text'].shape)
    print(df['label'].shape)

    return df['text'],df['label']



X, y = load_data(csv_path=csv_path, sample_size=_SAMPLE_SIZE,columns_tf_idfizable=COL_TF_IDF)



# Split data
print("\nSplitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

print("\nData con el spliting...")
print(f"En TRAIN: {y_train.shape}")
print(f"En TEST: {y_test.shape}")
        
X_train = X_train.astype(str).tolist()
X_test  = X_test.astype(str).tolist()
# y_train = y_train.astype(str).tolist()
# y_test  = y_test.astype(str).tolist()

print(type(y_train[0]))


# define model
model = BertClassifier()         # text/text pair classification
model.num_labels = 2
# model.epochs = 3
# model.bert_model =  'brunokreiner/lyrics-bert'
# model = BertRegressor()        # text/text pair regression
# model = BertTokenClassifier()  # token sequence classification


# finetune model
model.fit(X_train, y_train)

# make predictions
y_pred = model.predict(X_test)

# make probabilty predictions
# y_pred = model.predict_proba(X_test)

# score model on test data
model.score(X_test, y_test)

# save model to disk
save_dir = './data'
os.makedirs(save_dir, exist_ok=True)
savefile = os.path.join(save_dir, 'mymodel.bin')
model.save(savefile)

# load model from disk
# new_model = load_model(savefile)

# do stuff with new model
# new_model.score(X_test, y_test)

# --- Matriz de confusión ---
def map_labels_to_int(labels):
    mapping = { 'Yes':1, 'No':0}  
    return [mapping[str(label).strip()] for label in labels]

y_test_int = map_labels_to_int(y_test)
y_pred_int = map_labels_to_int(y_pred)

print("\nReporte de clasificación:")
print(classification_report(y_test_int, y_pred_int, target_names=['Not Explicit', 'Explicit']))


cm = confusion_matrix(y_test_int, y_pred_int, labels=[0, 1])
print("Matriz de confusión (0 = Not Explicit, 1 = Explicit):")
print(cm)

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Not Explicit', 'Explicit'],
            yticklabels=['Not Explicit', 'Explicit'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')

conf_matrix_path = os.path.join(save_dir, 'confusion_matrix.png')
plt.savefig(conf_matrix_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Matriz de confusión guardada en: {conf_matrix_path}")

## Metricas

acc = accuracy_score(y_test_int, y_pred_int)
prec = precision_score(y_test_int, y_pred_int, )
rec = recall_score(y_test_int, y_pred_int)
f1 = f1_score(y_test_int, y_pred_int)
print("Mrtricas: ")
print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1-score:  {f1:.4f}")