# flake8: noqa: E501
#
# En este dataset se desea pronosticar el default (pago) del cliente el próximo
# mes a partir de 23 variables explicativas.
#
#   LIMIT_BAL: Monto del credito otorgado. Incluye el credito individual y el
#              credito familiar (suplementario).
#         SEX: Genero (1=male; 2=female).
#   EDUCATION: Educacion (0=N/A; 1=graduate school; 2=university; 3=high school; 4=others).
#    MARRIAGE: Estado civil (0=N/A; 1=married; 2=single; 3=others).
#         AGE: Edad (years).
#       PAY_0: Historia de pagos pasados. Estado del pago en septiembre, 2005.
#       PAY_2: Historia de pagos pasados. Estado del pago en agosto, 2005.
#       PAY_3: Historia de pagos pasados. Estado del pago en julio, 2005.
#       PAY_4: Historia de pagos pasados. Estado del pago en junio, 2005.
#       PAY_5: Historia de pagos pasados. Estado del pago en mayo, 2005.
#       PAY_6: Historia de pagos pasados. Estado del pago en abril, 2005.
#   BILL_AMT1: Historia de pagos pasados. Monto a pagar en septiembre, 2005.
#   BILL_AMT2: Historia de pagos pasados. Monto a pagar en agosto, 2005.
#   BILL_AMT3: Historia de pagos pasados. Monto a pagar en julio, 2005.
#   BILL_AMT4: Historia de pagos pasados. Monto a pagar en junio, 2005.
#   BILL_AMT5: Historia de pagos pasados. Monto a pagar en mayo, 2005.
#   BILL_AMT6: Historia de pagos pasados. Monto a pagar en abril, 2005.
#    PAY_AMT1: Historia de pagos pasados. Monto pagado en septiembre, 2005.
#    PAY_AMT2: Historia de pagos pasados. Monto pagado en agosto, 2005.
#    PAY_AMT3: Historia de pagos pasados. Monto pagado en julio, 2005.
#    PAY_AMT4: Historia de pagos pasados. Monto pagado en junio, 2005.
#    PAY_AMT5: Historia de pagos pasados. Monto pagado en mayo, 2005.
#    PAY_AMT6: Historia de pagos pasados. Monto pagado en abril, 2005.
#
# La variable "default payment next month" corresponde a la variable objetivo.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# clasificación están descritos a continuación.
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import precision_score, balanced_accuracy_score, recall_score, f1_score, confusion_matrix
from sklearn.svm import SVC
from sklearn.decomposition import PCA
import json
import gzip
import joblib
import os
import pickle
# Paso 1.
# Realice la limpieza de los datasets:
# - Renombre la columna "default payment next month" a "default".
# - Remueva la columna "ID".
# - Elimine los registros con informacion no disponible.
# - Para la columna EDUCATION, valores > 4 indican niveles superiores
#   de educación, agrupe estos valores en la categoría "others".
# - Renombre la columna "default payment next month" a "default"
# - Remueva la columna "ID".
train = pd.read_csv('/content/files/input/test_default_of_credit_card_clients.csv')
test = pd.read_csv('/content/files/input/train_default_of_credit_card_clients.csv')

train.rename(columns={"default payment next month": "default"}, inplace=True)
test.rename(columns={"default payment next month": "default"}, inplace=True)

train.drop(columns=["ID"], inplace=True)
test.drop(columns=["ID"], inplace=True)

train.dropna(inplace=True)
test.dropna(inplace=True)

train['EDUCATION'] = np.where(train['EDUCATION'] > 4, 4, train['EDUCATION'])
test['EDUCATION'] = np.where(test['EDUCATION'] > 4, 4, test['EDUCATION'])
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
x_train = train.drop(columns=["default"])
y_train = train["default"]
x_test = test.drop(columns=["default"])
y_test = test["default"]

# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Descompone la matriz de entrada usando PCA. El PCA usa todas las componentes.
# - Estandariza la matriz de entrada.
# - Selecciona las K columnas mas relevantes de la matrix de entrada.
# - Ajusta una maquina de vectores de soporte (svm).
categorical_features = ['SEX', 'EDUCATION', 'MARRIAGE']
numerical_features = [col for col in train.columns if col not in categorical_features + ['default']]

categorical_transformer = OneHotEncoder(handle_unknown='ignore')

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer, categorical_features),
        ('num', 'passthrough', numerical_features)
    ]
)

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),                
    ('pca', PCA()),                                
    ('scaler', StandardScaler()),                  
    ('feature_selection', SelectKBest(f_classif)), 
    ('classifier', SVC(kernel='linear', random_state=42))  
])



# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use la función de precision
# balanceada para medir la precisión del modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
from sklearn.model_selection import GridSearchCV

# Definir los parámetros que se van a probar en el GridSearch con más variación
param_grid = {
    'pca__n_components': [10, 20, x_train.shape[1] - 2],  # Variar el número de componentes principales
    'feature_selection__k': [10, 12, 15],                 # Variar las características seleccionadas
    'classifier__kernel': ['linear', 'rbf', 'poly'],      # Probar diferentes kernels en SVM
    'classifier__C': [0.1, 1, 10],                        # Regularización para SVM
    'classifier__gamma': ['scale', 0.1, 0.01]             # Parámetro gamma del kernel SVM
}

# Crear el modelo con GridSearch
model = GridSearchCV(
    pipeline,                      # El pipeline definido previamente
    param_grid,                    # Los hiperparámetros a probar
    cv=10,                         # Número de splits para validación cruzada
    scoring="balanced_accuracy",   # Métrica de precisión balanceada
    n_jobs=-1,                     # Usar todos los núcleos disponibles
    refit=True                     # Refinar el modelo con el mejor resultado
)

# Ajustar el modelo con el conjunto de entrenamiento
model.fit(x_train, y_train)

# Ver los mejores parámetros encontrados
print(f"Mejores hiperparámetros: {model.best_params_}")

# Paso 6.
# Calcule las metricas de precision, precision balanceada, recall,
# y f1-score para los conjuntos de entrenamiento y prueba.
# Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# Este diccionario tiene un campo para indicar si es el conjunto
# de entrenamiento o prueba. Por ejemplo:
#
# {'dataset': 'train', 'precision': 0.8, 'balanced_accuracy': 0.7, 'recall': 0.9, 'f1_score': 0.85}
# {'dataset': 'test', 'precision': 0.7, 'balanced_accuracy': 0.6, 'recall': 0.8, 'f1_score': 0.75}
from sklearn.model_selection import GridSearchCV

# Definir los parámetros que se van a probar en el GridSearch con más variación
param_grid = {
    'pca__n_components': [10, 20, x_train.shape[1] - 2],  # Variar el número de componentes principales
    'feature_selection__k': [10, 12, 15],                 # Variar las características seleccionadas
    'classifier__kernel': ['linear', 'rbf', 'poly'],      # Probar diferentes kernels en SVM
    'classifier__C': [0.1, 1, 10],                        # Regularización para SVM
    'classifier__gamma': ['scale', 0.1, 0.01]             # Parámetro gamma del kernel SVM
}

# Crear el modelo con GridSearch
model = GridSearchCV(
    pipeline,                      # El pipeline definido previamente
    param_grid,                    # Los hiperparámetros a probar
    cv=10,                         # Número de splits para validación cruzada
    scoring="balanced_accuracy",   # Métrica de precisión balanceada
    n_jobs=-1,                     # Usar todos los núcleos disponibles
    refit=True                     # Refinar el modelo con el mejor resultado
)

# Ajustar el modelo con el conjunto de entrenamiento
model.fit(x_train, y_train)

# Ver los mejores parámetros encontrados
print(f"Mejores hiperparámetros: {model.best_params_}")

# Paso 7.
# Calcule las matrices de confusion para los conjuntos de entrenamiento y
# prueba. Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'cm_matrix', 'dataset': 'train', 'true_0': {"predicted_0": 15562, "predicte_1": 666}, 'true_1': {"predicted_0": 3333, "predicted_1": 1444}}
# {'type': 'cm_matrix', 'dataset': 'test', 'true_0': {"predicted_0": 15562, "predicte_1": 650}, 'true_1': {"predicted_0": 2490, "predicted_1": 1420}}
#
cm_train = confusion_matrix(y_train, y_train_pred)
cm_matrix_train = {
    'type': 'cm_matrix',
    'dataset': 'train',
    'true_0': {"predicted_0": int(cm_train[0, 0]), "predicted_1": int(cm_train[0, 1])},
    'true_1': {"predicted_0": int(cm_train[1, 0]), "predicted_1": int(cm_train[1, 1])}
}

cm_test = confusion_matrix(y_test, y_test_pred)
cm_matrix_test = {
    'type': 'cm_matrix',
    'dataset': 'test',
    'true_0': {"predicted_0": int(cm_test[0, 0]), "predicted_1": int(cm_test[0, 1])},
    'true_1': {"predicted_0": int(cm_test[1, 0]), "predicted_1": int(cm_test[1, 1])}
}

metrics = [
    cm_matrix_train,
    cm_matrix_test
]


output_path = '/content/files/output/metrics.json'
with open(output_path, 'a') as f: 
    for metric in metrics:
        f.write(json.dumps(metric) + '\n')
