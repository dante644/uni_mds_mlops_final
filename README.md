# Modelo Avanzado de Credit Scoring - Banco de Taiwán (BOT)



## 1. Definición del Caso de Uso (IA/ML/MLOps)
**Problema Específico:** Predicción de la probabilidad de incumplimiento de pago (*default*) en clientes de tarjetas de crédito para optimizar la toma de decisiones financieras.

El Banco de Taiwán busca transformar su gestión de riesgos mediante un modelo de **Machine Learning de Clasificación Binaria**. El objetivo es identificar patrones de comportamiento en el historial crediticio y factores demográficos que preceden a un impago, permitiendo a la institución actuar de manera preventiva.

---

## 2. Descripción del Proyecto

### Contexto del Problema
El sector financiero en Taiwán requiere herramientas de alta precisión para competir en un mercado volátil. Contamos con un conjunto de datos histórico (abril 2005 - septiembre 2005) que detalla:
* **Factores Demográficos:** Género, educación, estado civil y edad.
* **Historial de Crédito:** Monto del crédito otorgado.
* **Historial de Pagos:** Registros de puntualidad en los últimos 6 meses.
* **Estado de Cuenta:** Montos de facturación y pagos realizados.



### Limitaciones
* **Ventana Temporal:** Los datos representan un periodo específico de 2005; el modelo debe ser validado frente a cambios en el comportamiento del consumidor moderno.
* **Desbalance de Datos:** Generalmente, los casos de incumplimiento son significativamente menores a los de cumplimiento, lo que requiere técnicas de remuestreo (SMOTE) o ajuste de pesos en el algoritmo.
* **Interpretabilidad:** Debido a regulaciones bancarias, el modelo no solo debe ser preciso, sino también explicable (identificar por qué se rechaza un crédito).

### Objetivos
* **Desarrollar** un modelo predictivo robusto basado en algoritmos avanzados (ej. XGBoost, Random Forest o Redes Neuronales).
* **Mejorar** la precisión en la evaluación del riesgo crediticio en comparación con métodos estadísticos tradicionales.
* **Fortalecer** la competitividad del Banco de Taiwán en el mercado mediante una gestión de cartera más inteligente.

### Beneficios y Resultados Esperados
1.  **Reducción de la Tasa de Incumplimiento:** Disminución directa de las pérdidas por cuentas incobrables.
2.  **Optimización de la Cartera:** Capacidad de ofrecer mejores condiciones a clientes de bajo riesgo, aumentando la retención.
3.  **Automatización:** Reducción de tiempos en el análisis manual de solicitudes de crédito.
4.  **Dashboard de Riesgo:** Visualización clara de las variables que más impactan en el riesgo crediticio actual.

---

## 3. Métricas de Éxito (KPIs de ML)
Para que la solución sea considerada exitosa, el modelo debe alcanzar o superar los siguientes umbrales:

| Métrica | Objetivo | Justificación |
| :--- | :--- | :--- |
| **AUC-ROC** | $> 0.78$ | Capacidad del modelo para distinguir entre un cliente que pagará y uno que no. |
| **F1-Score** | Balanceado | Asegura un equilibrio entre precisión y sensibilidad ante datos desbalanceados. |

---

## 4. Tecnologías Sugeridas
* **Lenguaje:** Python 3.x
* **Librerías:** Pandas, Scikit-learn, XGBoost/LightGBM, Matplotlib/Seaborn para visualización.
* **Entorno:** Visual Code / Git Hub.

---


## 5. Adquisición de datos

| Variable | Descripción |
| :--- | :--- | 
| ID | ID del cliente (índice)  |
| LIMIT_BAL | Monto del crédito otorgado (NT dollar)  |
| SEX | Género (masculino; femenino)|
| EDUCATION | Nivel educativo (escuela de posgrado; universidad; escuela secundaria; otros)   |
| MARRIAGE | Estado civil (casado; soltero; otros)  |
| AGE | Edad en años. |
| PAY_0 | Estado de pago en septiembre de 2005 (-2 = sin consumo; -1 = pago puntual; 0 = uso de crédito renovable; 1 = retraso en el pago de un mes; 2 = retraso en el pago de dos meses; y así sucesivamente)  |
| PAY_2 | Estado de pago en agosto de 2005 (misma escala que la anterior)  |
| PAY_3 | Estado de pago en julio de 2005 (misma escala que la anterior) |
| PAY_4 | Estado de pago en junio de 2005 (misma escala que la anterior)  |
| PAY_5 | Estado de pago en mayo de 2005 (misma escala que la anterior) arriba)  |
| PAY_6 | Estado de pago en abril de 2005 (misma escala que arriba)  |
| BILL_AMT1 | Importe del estado de cuenta en septiembre de 2005 (dólar NT)  |
| BILL_AMT2 | Importe del estado de cuenta en agosto de 2005 (dólar NT)  |
| BILL_AMT3 | Importe del estado de cuenta en julio de 2005 (dólar NT)  |
| BILL_AMT4 | Importe del estado de cuenta en junio de 2005 (dólar NT) |
| BILL_AMT5 | Importe del estado de cuenta en mayo de 2005 (dólar NT)  |
| BILL_AMT6 | Importe del estado de cuenta en abril de 2005 (dólar NT)  |
| PAY_AMT1 | Importe pagado en septiembre de 2005 (dólar NT)  |
| PAY_AMT2 | Importe pagado en agosto de 2005 (dólar NT)  |
| PAY_AMT3 | Importe pagado en julio de 2005 (dólar NT)  |
| PAY_AMT4 | Monto pagado en junio de 2005 (dólar NT)  |
| PAY_AMT5 | Monto pagado en mayo de 2005 (dólar NT)  |
| PAY_AMT6 | Monto pagado en abril de 2005 (dólar NT)  |
| default_payment_next_month | Pago predeterminado (1 = sí; 0 = no) |

---


## 6. Estructura del Proyecto

```
Credit Scoring/
├── data/
│   ├── raw/                    # Dataset original
│   │   └── cleaned_credit_card_data.csv
│   └── training/               # Datos procesados
│        └── processed_credit_card_data.csv
├── imagen/                     # Imagenes de analisis 
│   ├── boxplot_variables.png
│   ├── correlacion_final.png
│   └── correlacion_inicial.png
├── mlruns/                     # Run ID del MLflow
├── models/                     # Modelos entrenados
│   ├── adaboost.pkl
│   ├── gradient_boosting.pkl
│   ├── knn.pkl
│   ├── logistic_regression.pkl
│   ├── naive_bayes.pkl
│   ├── random_forest.pkl
│   └── svm.pkl
├── reports/                     # Resultados
│   ├── cm_AdaBoost.png
│   ├── cm_Gradient_Boosting.png
│   ├── cm_KNN.png
│   ├── cm_Logistic_Regression.png
│   ├── cm_Naive_Bayes.png
│   ├── cm_Random_Forest.png
│   ├── cm_SVM.png
│   ├── final_metrics.csv
│   └── prediction_result.json
├── src/                        # Código modular
│   ├── data_preparation.py
│   ├── experiments.py
│   └── prediction.py
├── requirements.txt
├── README.md
├── terminal.txt                # Ejecuciones que se hicieron
└── .gitignore
```

---

## 7. Proceso del proyecto

### 7.1 Instalación del entorno virtual

```bash
# Ejecutar en el terminal de Powershell

# Habilitar la ejecución de scripts de forma segura y temporal y activar el entorno virtual
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process

# Se usara la versión 3.11 para construir una carpeta llamada .venv que contenga una copia aislada de Python de esa versión,
# debido que algunas librerias no son compatibles con las versiones recientes, entrando en conflicto.
py -3.11 -m venv .venv

# Este comando se utiliza para activar el entorno virtual específicamente en PowerShell
.\.venv\Scripts\Activate.ps1

# Nota: Si se presenta conflicto de versiones, usar este comando que elimina el entorno virtual
# y crear nuevamente el entorno virtual
Remove-Item -Recurse -Force .venv

```

### 7.2 Instalación de librerias

```bash
# Ejecutar en el terminal de Powershell

# Nota: Si presenta un error en el pip o no detecta el pip (gestor de paquetes de Python), activar este comando actualiza el pip
python -m pip install --upgrade pip

# Este comando se utiliza para instalar automáticamente todas las librerías con sus versiones especificas que figuran en ese txt
pip install -r requirements.txt

```

### 7.3 Tratamiento de Datos

```bash
# Ejecutar en el terminal de Powershell

py src/data_preparation.py
```
Se usa la data alojada data/raw/cleaned_credit_card_data.csv, donde se realiza la busqueda de nulos (no presenta la data), se elimina la variable ID (no aporta al analisis del modelo), transformaciones de variables a categoricas ('SEX', 'EDUCATION', 'MARRIAGE', 'default_payment_next_month') y a enteras ('PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6'), se realiza un análisis correlacional (arroja una imagen en imagen/correlacion_inicial.png) donde presenta la correlación de las variables, se presenta la variables que presenta alta correlación ("BILL_AMT1", "BILL_AMT3", "BILL_AMT4", "BILL_AMT6") y quedaria la correlación final (arroja una imagen en imagen/correlacion_final.png) y por ultimo se hace un análisis que variables que presenta outliers (arroja una imagen en imagen/boxplot_variables.png) para realizar el tratamiento de las variables que presentaban outliers ('BILL_AMT2', 'BILL_AMT5', 'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6') para que al final se obtenga una data limpia y se envie a la carpeta siguiente data/training/processed_credit_card_data.csv

### 7.4 Modelamiento - Evaluación

```bash
# Ejecutar en el terminal de Powershell

py src/experiments.py
```

Se realiza el modelamiento con la data ya preparada con los siguientes modelos: Logistic_Regression, Random_Forest, SVM, Gradient_Boosting, AdaBoost, KNN, Naive_Bayes. Un 80% para el tratamiento y 20% para el test, donde los resultados aparecen en la carpeta reports: la matriz de confusión de cada variable, los modelos en formato .pkl (excepto el random_forest.pkl debido que el archivo pesa 25MB y no se puede colocar el GitHub) y las metricas en formato csv donde integra las metricas de cada modelo y Run ID (identificador de cada modelo para realizar el despligue y por consiguiente el servicio en el mlflow)

Observando, el mejor modelo es Gradient_Boosting.

| Modelo | F1-Score | Accuracy | Run_ID |
|--------|----------|----------|--------|
| **Gradient_Boosting** | **0.7995453627211427** | **0.8208333333333333** | **62d737358b96407b82b8bdd76aef4b98** |
| Random_Forest | 0.7949377236570891 | 0.8141666666666667 | 668f238435ab44d9a08cd0f03b45f1dd |
| AdaBoost | 0.7933341293532338 | 0.8186666666666667 | b93fe74931344e919068fb1d7e6207b2 |
| KNN | 0.7174260659393438 | 0.7545 | b71a745034dd47dea49d88d56f585429 |
| Logistic_Regression | 0.6859041959551291 | 0.7813333333333333 | 201938fa783a4a0ca160556983ca7a99 |
| SVM | 0.6851928823180812 | 0.7811666666666667 | fabf51ecb11d47abb1774c6ea0dece7d |
| Naive_Bayes | 0.6793083546301553 | 0.6548333333333334 | 5762cccf20c644d69b9eb06e547919d7 |

Nota: RUN_ID varian cuando se ejecuta otra vez, estos codigos del Run_ID son los resultados que salieron para el proyecto.

```bash
# Ejecutar en el terminal de Powershell - dejarlo abierto
mlflow ui

```

Abres para ver el entorno de MLflow: **http://127.0.0.1:5000**

Dentro de la interfaz de MLflow, cada ejecución de un modelo se visualiza como un registro detallado que centraliza sus parámetros (configuraciones de entrada como el learning rate), sus métricas (resultados de rendimiento del Accuracy y F1-Score), y sus artefactos, que incluyen el archivo binario del modelo entrenado junto con los archivos de entorno (conda.yaml) necesarios para su réplica.

Nota: En la ejecución de este script, arroja la carpeta llamada mlruns, donde refleja el Run ID del mlflow de cada modelo, pero no se pudo alojar en GitHub los artefactos de cada modelo debido que pesan 25MB (conda.yaml, MLmodel, model.pkl, python_env.yaml y requirements).


### 7.5 Servicio

```bash
# En el terminal del Git Bash - dejarlo abierto
mlflow ui --port 5000
```

Este es el servidor de visualización, el comando mlflow ui levanta la interfaz gráfica en el puerto 5000 para que pueda, como desarrollador, visualizar, comparar y administrar todos los experimentos, parámetros y métricas registrados en la base de datos local a través del navegador. Se deja abierto porque es un proceso activo que actúa como servidor de lectura de tus archivos de seguimiento, permitiéndote identificar visualmente cuál es el mejor Run ID antes de decidir cuál llevarás a la siguiente etapa.


```bash
# En otro terminal del Git Bash - dejarlo abierto
mlflow models serve -m runs:/668f238435ab44d9a08cd0f03b45f1dd/model --port 5001 --no-conda
```

Abre: **http://127.0.0.1:5001**

Este es el serving, el comando mlflow models serve convierte un modelo específico (identificado por su Run ID) en una API activa y funcional en el puerto 5001, permitiendo que otras aplicaciones le envíen datos y reciban predicciones en tiempo real. Al usar la bandera --no-conda, le indicas que ejecute el modelo utilizando las librerías de tu entorno actual (.venv) en lugar de crear uno nuevo, y se mantiene abierto porque funciona como un "operador" a la espera de peticiones externas; si cierras la terminal, el modelo deja de "escuchar" y el servicio de predicción se apaga.

Nota: El Run_ID es del modelo Random_Forest, ahi se debe poner el Run_ID del modelo que se planea usar para la predicción.

### 7.6 Despliegue del modelo

El uso de un API REST cliente es el paso decisivo para que el modelo de Machine Learning deje de ser un archivo estático y se convierta en un servicio dinámico. Al utilizar herramientas como Python o curl, actúas como el "solicitante" que envía datos estructurados al servidor de MLflow (que configuraste en el puerto 5001), permitiendo que cualquier aplicación externa obtenga predicciones en tiempo real sin necesidad de conocer la lógica interna del modelo.

Se usaron dos los métodos para la predicción:

Al utilizar Python (específicamente la librería requests) como cliente de la API REST permite integrar las predicciones de forma automatizada y elegante dentro de flujos de trabajo profesionales. Los resultados de la predicción figuran en la carpeta reports con el archivo: prediction_result.json.

```bash
# Ejecutar en otro terminal de Powershell
py src/prediction.py

```

**O**

El uso de curl como cliente de la API REST es la forma más directa y pura de interactuar con el modelo desde la terminal de comandos de Git Bash y solo colocariamos el siguiente codigo para la predicción solicitada.

```bash
# En el terminal del Git Bash
curl -X POST -H "Content-Type: application/json" \
  --data '{
    "dataframe_split": {
      "columns": [
      "LIMIT_BAL", "AGE", "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6", "BILL_AMT2", "BILL_AMT5", "PAY_AMT1", "PAY AMT2", "PAY_AMT3", "PAY AMT4", "PAY_ AMT5", "PAY AMT6"
      ],
    "data": [
    [20000, 24, 2, 2, -1, -1, -2, -2, 3102, 0, 0, 689, 0, 0, 0, 0]
      ]
     }
   }' \
http://127.0.0.1:5001/invocations

```

***Resultado de la predición: 1***

---

## 8. Conclusiones

Tras un exhaustivo proceso de experimentación y evaluación de diversos algoritmos de clasificación, los resultados consolidados en la plataforma MLflow confirman que Gradient Boosting es el modelo con mejor desempeño para este caso de uso. Este algoritmo alcanzó un F1-Score de 0.7995 y un Accuracy de 0.8208, superando a otras alternativas como Random Forest y AdaBoost. La superioridad de este modelo radica en su capacidad para manejar relaciones no lineales complejas entre el historial de pagos y el perfil demográfico de los clientes del Banco de Taiwán.

### Resultados Esperados

Con la implementación de este modelo avanzado de credit scoring, se uso el modelo Random_Forest, donde el Banco de Taiwán proyecta una mejora significativa en la precisión de sus evaluaciones de riesgo crediticio. La integración del modelo mediante scripts de predicción permita identificar de forma preventiva a los clientes con alta probabilidad de incumplimiento, lo que se traducirá en una reducción directa de la tasa de morosidad y un fortalecimiento de la competitividad de la institución en el mercado financiero. Asimismo, la automatización del proceso de evaluación optimizará la gestión de la cartera de clientes, permitiendo una toma de decisiones más ágil y basada en datos.

### Lecciones Aprendidas

El desarrollo del proyecto permitió validar que los métodos de ensamble, específicamente los basados en refuerzo (boosting), ofrecen una ventaja predictiva superior frente a modelos lineales tradicionales como la Regresión Logística o SVM en el contexto de riesgo bancario. Una lección clave fue la importancia de utilizar métricas como el F1-Score para balancear la precisión y la sensibilidad, especialmente dado que los casos de incumplimiento suelen ser menos frecuentes en los datos. Además, la implementación técnica mediante peticiones curl y scripts de Python demostró la viabilidad de desplegar estos modelos en entornos de producción en tiempo real.

### Limitaciones del Proyecto

A pesar de los resultados positivos, el proyecto presenta limitaciones importantes, principalmente relacionadas con la temporalidad de los datos, que corresponden al periodo entre abril y septiembre de 2005. Esta brecha de tiempo implica que los patrones de comportamiento actuales podrían haber variado, afectando la precisión del modelo si no se actualiza con datos recientes. Por otro lado, la naturaleza de "caja negra" de los modelos de Gradient Boosting puede presentar desafíos ante regulaciones que exijan una explicabilidad detallada sobre por qué se deniega un crédito a un cliente específico.

### Para Futuras Investigaciones

Para dar continuidad a este trabajo, se recomienda explorar la incorporación de técnicas de explicabilidad (como SHAP o LIME) que permitan desglosar el impacto de cada variable en la predicción final. También es fundamental investigar la inclusión de variables externas, como indicadores macroeconómicos de Taiwán, para enriquecer el contexto del modelo. Finalmente, se propone establecer un flujo de aprendizaje continuo donde el sistema se re-entrene periódicamente con datos nuevos, garantizando que la capacidad predictiva del Banco de Taiwán se mantenga vigente frente a las nuevas dinámicas del mercado crediticio.

---
