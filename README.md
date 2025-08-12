# ml-project-models

Subir el csv y los embbeding en formato .npy en la carpeta data



# Relevant packages

python -m pip install asreview sentence-transformers tf-keras


# For Bert Classifier

git clone -b master https://github.com/charles9n/bert-sklearn
cd bert-sklearn
pip3 install .

# To fix bug: 
in models_comparison/baseline_clean/Bert_classifier/bert-sklearn/bert_sklearn/sklearn.py

put: 
self.label2id = {"Yes": 1, "No": 0}
self.id2label = {1: "Yes", 0: "No"}
after
        y_pred = np.argmax(self.predict_proba(X), axis=1)

and in 
models_comparison/baseline_clean/Bert_classifier/bert-sklearn/bert_sklearn/data/data.py

put in th eline 72: 
self.label2id = {"Yes": 1, "No": 0}
                label_id = self.label2id[label]

                
