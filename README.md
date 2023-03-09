# Machine Learning :woman_student: :computer:

Repositorio de los ejercicios de pair programming de estadística y Machine Learning  desarrollados *Bootcamp* de [Adalab](https://adalab.es/#) de [Analistas de Datos](https://adalab.es/bootcamp-data/): :woman_technologist:

### Índice
- [Estadística](#estadistica)
    - [Estadística Estructura del repositorio](#estadistica-estructura-del-repositorio)
    - [Estadística Lineal Biblioteca](#estadistica-bibliotecas)
- [Regresion Lineal](#regresion-lineal)
    - [Regresion Lineal Estructura del repositorio](#regresion-lineal-estructura-del-repositorio)
    - [Regresion Lineal Biblioteca](#regresion-lineal-bibliotecas)
- [Regresion Logistica](#regresion-logistica)
    - [Regresion Logistica Estructura del repositorio](#regresion-logistica-estructura-del-repositorio)
    - [Regresion Logistica Biblioteca](#regresion-logistica-biblioteca)

***
- **datos** - [Carpeta](https://github.com/paulafuenteg/Machine-Learning/tree/main/datos)  
    Aquí encontramos todos los ficheros que hemos ido utilizando a lo largo de todos los ejercicios  

### **`Estadística`**

Utilizamos el *DataFrame* [Sephora Website](https://www.kaggle.com/datasets/raghadalharbi/all-products-available-on-sephora-website?resource=download)

|Columna| Tipo de dato | Descripcion |
|-------|--------------|-------------|
|**id**| int |	ID del producto.
|**brand**	| string |	Marca del producto
|**category**	| string |	La categoría del producto
|**name**	| string | Nombre del producto
|**size**	| string |	Tamaño del producto
|**rating**	| Float |	La valoración del producto
|**love**| int |    Numero de personas que les gusta el producto
|**price**	| float |	Precio del producto
|**value_price**| float|	Valor del producto
|**URL**| string|	Link del producto
|**MarketingFlags**| Bool |	Si se venden exclusivamente online
|**options**| string |	 Opciones de color y tamaños
|**details**| string |	 Detalles del producto
|**how_to_use**| string |	Instrucciones del producto
|**ingredients**| string |	 Ingredientes
|**online_only**| int |	 Si el producto se vende solo online
|**exclusive**| int |	 Si el producto se vende exclusivamente en la web de Sephora
|**limited_edition**| int |	 Si el producto es edición limitada
|**limited_time_offer**| int |	 Si el producto tiene un tiempo limitado de oferta

### **Estadística Estructura del Repositorio**:

- **Estadística** - [Carpeta](https://github.com/paulafuenteg/Machine-Learning/tree/main/Estadistica)  
En los siguientes ficheros podemos encontar nuestro estudio sobre el dataset de la web de Sephora, explorando sus datos principalmente de los estadísticos de los precios de sus productos.
    - [Lecc01-Introducción](https://github.com/paulafuenteg/Machine-Learning/blob/main/Estadistica/modulo-2-Estadistica-1.ipynb)
    - [Lecc02-Lecc02-Cuartiles_estadistica_contigencia](https://github.com/paulafuenteg/Machine-Learning/blob/main/Estadistica/modulo-2-Estadistica-2.ipynb)
    - [Lecc03-Contigencia_correlacion_sesgos_int_confianza](https://github.com/paulafuenteg/Machine-Learning/blob/main/Estadistica/modulo-2-Estad%C3%ADstica-3.ipynb)
    - [Lecc04-Repaso](https://github.com/paulafuenteg/Machine-Learning/blob/main/Estadistica/modulo-2-repaso-conceptos-resumidos.ipynb) - En este repaso se utiliza la base de datos de [Top 1000s in IMDB](https://www.kaggle.com/datasets/ramjasmaurya/top-250s-in-imdb)

### **Estadística Bibliotecas:**
```
# Tratamiento de datos
import numpy as np
import pandas as pd

# Test estadisticos
from scipy.stats import skew
import scipy.stats as st

# Gráficos
import matplotlib.pyplot as plt
import seaborn as sns

```

### **`Regresion Lineal`**

Empezamos con explorar el *Dataframe* que tenemos y decidir cual será nuestra variable respuesta.    

Utilizamos el *DataFrame* [*Global Disaster Risk*](https://www.kaggle.com/datasets/tr1gg3rtrash/global-disaster-risk-index-time-series-dataset)

|Columna| Tipo de dato | Descripcion |
|-------|--------------|-------------|
|**Region**| String|	Nombre de la region.
|**WRI**	| Decimal |	*World Risk Score* (Puntuaciones de riesgo de las regiones)
|**Exposure**	| Decimal |	Riesgo/exposición a peligros naturales como terremotos, huracanes, inundaciones, sequías y aumento del nivel del mar.
|**Vulnerability**	| Decimal | Vulnerabilidad en función de la infraestructura, la nutrición, la situación de la vivienda y las condiciones del marco económico.
|**Susceptibility**	| Decimal |	Susceptibilidad según la infraestructura, la nutrición, la situación de la vivienda y las condiciones del marco económico.
|**Lack of Coping Capabilities**	| Decimal |	Preparación ante desastres, atención medica, seguridad social.
|**Lack of Adaptive Capacities**| Decimal |	Capacidades de adaptácion ante eventos naturales, cambio climático y otro desafíos.
|**Year**	| Decimal |	Años.
|**WRI Category**| String|	Categoria calculada en base al *WRI*.
|**Exposure Category**| String|	Categoria calculada en base al *Exposure*.
|**Vulnerability Categoy**| String|	Categoria calculada en base al *Vulnerability*.
|**Susceptibility Category**| String|	 Categoria calculada en base al *Susceptibility*.

---

### **Regresion Lineal Estructura del Repositorio**:

- **deepl** - [Carpeta](https://github.com/paulafuenteg/Machine-Learning/tree/main/datos)
Aquí encontramos los ficheros en lo que hemos realizado la traducción de la columna *region*.  
Enlace con toda la info sobre deepL [deepl-Python](https://github.com/DeepLcom/deepl-python).

- **Regresion Lineal** - [Carpeta](https://github.com/Adalab/DA-promoC-Mod3-sprint1-VannayPaula/tree/main/Regresion%20Lineal)  
En los siguientes ficheros podemos encontar nuestro estudio sobre los datos, utilizando la metodologia EDA, averiguamos si hay nulos, *outliers*, realizamos graficas.  
Averiguamos correlaciones, normalizamos, estandardizamos y aplicamos el *encoding* a los datos.  
Aplicamos la Regresion lineal, *Decision Tree* y *Random Forest*.  


    - [Lecc01-Intro_ML](https://github.com/Adalab/DA-promoC-Mod3-sprint1-VannayPaula/blob/main/Regresion%20Lineal/Lecc01-Intro_ML.ipynb)
    - [Lecc02-Test_Estadisticos](https://github.com/Adalab/DA-promoC-Mod3-sprint1-VannayPaula/blob/main/Regresion%20Lineal/Lecc02-Test_Estadisticos.ipynb)
    - [Lecc03-Correlación_Covarianza](https://github.com/Adalab/DA-promoC-Mod3-sprint1-VannayPaula/blob/main/Regresion%20Lineal/Lecc03-Correlacion_Covarianza.ipynb)
    - [Lecc04-Asunciones](https://github.com/Adalab/DA-promoC-Mod3-sprint1-VannayPaula/blob/main/Regresion%20Lineal/Lecc04-Asunciones.ipynb)
    - [Lecc05-Normalización](https://github.com/Adalab/DA-promoC-Mod3-sprint1-VannayPaula/blob/main/Regresion%20Lineal/Lecc05-Normalizaci%C3%B3n.ipynb)
    - [Lecc06-Estandardizacion](https://github.com/Adalab/DA-promoC-Mod3-sprint1-VannayPaula/blob/main/Regresion%20Lineal/Lecc06-Estandarizacion.ipynb)
    - [Lecc07-Anova](https://github.com/Adalab/DA-promoC-Mod3-sprint1-VannayPaula/blob/main/Regresion%20Lineal/Lecc07-Anova.ipynb)
    - [Lecc08-Encoding](https://github.com/Adalab/DA-promoC-Mod3-sprint1-VannayPaula/blob/main/Regresion%20Lineal/Lecc08-Encoding.ipynb)
    - [Lecc09-Regresion_lineal_Intro](https://github.com/Adalab/DA-promoC-Mod3-sprint1-VannayPaula/blob/main/Regresion%20Lineal/Lecc09-Regresion_lineal_Intro.ipynb)
    - [Lecc10-Regresion_lineal_Metricas](https://github.com/Adalab/DA-promoC-Mod3-sprint1-VannayPaula/blob/main/Regresion%20Lineal/Lecc10-Regresion-lineal_Metricas.ipynb)
    - [Lecc11-Decision_tree](https://github.com/Adalab/DA-promoC-Mod3-sprint1-VannayPaula/blob/main/Regresion%20Lineal/Lecc11-Decision_Tree.ipynb)
    - [Lecc12-Random_Forest](https://github.com/Adalab/DA-promoC-Mod3-sprint1-VannayPaula/blob/main/Regresion%20Lineal/Lecc12-Random_Forest.ipynb)

---


### **Regresion Lineal Bibliotecas:**

```
#Traducción columna region
import deepL

# Tratamiento de datos
import numpy as np
import pandas as pd

# Test estadisticos
import researchpy as rp
from scipy import stats
from scipy.stats import kstest
from scipy.stats import levene
from scipy.stats import skew
from scipy.stats import kurtosistest
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.multivariate.manova import MANOVA
from sklearn.preprocessing import StandardScaler

# Gráficos
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

# Transformación de los datos / modelado / evaluación / cross evaluacion
import math 
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn import tree
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import GridSearchCV
from tqdm import tqdm
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn import metrics

# Codificación de las variables numéricas
from sklearn.preprocessing import LabelEncoder # para realizar el Label Encoding 
from sklearn.preprocessing import OneHotEncoder  # para realizar el One-Hot Encoding

# Configuración warnings
import warnings
warnings.filterwarnings('once')
```
---

### **`Regresion Logistica`**


Empezamos con explorar el Dataframe que tenemos y decidir cual será nuestra variable respuesta.  
Utilizamos el *DataFrame* [Fraude de Tarjeta de Credito](https://www.kaggle.com/datasets/dhanushnarayananr/credit-card-fraud)

|Column| Type | Description |
|-------|--------------|-------------|
|distance_from_home| float64|	Distancia desde casa donde occurrió la transacción
|distance_from_last_transaction| float64|	Distancia desde donde occurrió la uñtima transacción  
|ratio_to_median_purchase_price| float64|	Ratio entre el precio de la transacción y el precio de la compra media
|repeat_retailer| float64|	¿La transacción se realizó desde el mismo vendidore/tienda? 
|used_chip| float64|	¿La transacción se realizó con el chip? 
|used_pin_number| float64|	¿La transacción se realizó utilizando el pin?  
|online_order | float64| ¿La transacción se realizó en internet? 
|fraud | float64| ¿La transacción es una fraude? 

---


### **Regresion Logistica Estructura del Repositorio**

En los siguientes ficheros podemos encontar nuestro estudio sobre los datos, utilizando la metodologia EDA, averiguamos la distribución de los datos, los balanceamos, estandarizamos y utilizamos la matriz de correlación.  
Aplicamos ambos el *Decision Tree* y el *Random Forest*.

> 🔺🔺 **ATENCÍON** 🔺🔺  
> Estos ultimos dos `jupiters`,están ejecutado directamente en el `google colab`.  

- **Regresion Logistica** - [Carpeta](https://github.com/Adalab/DA-promoC-Mod3-sprint1-VannayPaula/tree/main/Regresion%20Logistica) 

    - [Lecc01-EDA](https://github.com/Adalab/DA-promoC-Mod3-sprint1-VannayPaula/blob/main/Regresion%20Logistica/Lecc01-EDA.ipynb)  
    - [Lecc02-Preparacion_Datos](https://github.com/Adalab/DA-promoC-Mod3-sprint1-VannayPaula/blob/main/Regresion%20Logistica/Lecc02-Preparacion_Datos.ipynb)  
    - [Lecc03-Ajuste](https://github.com/Adalab/DA-promoC-Mod3-sprint1-VannayPaula/blob/main/Regresion%20Logistica/Lecc03-Ajuste.ipynb)  
    - [Lecc04-Metricas](https://github.com/Adalab/DA-promoC-Mod3-sprint1-VannayPaula/blob/main/Regresion%20Logistica/Lecc04-Metricas.ipynb)  
    - [Lecc05-Decision_Tree](https://github.com/Adalab/DA-promoC-Mod3-sprint1-VannayPaula/blob/main/Regresion%20Logistica/Lecc05-Decision_Tree.ipynb)  
    - [Lecc06-Random_Forest](https://github.com/Adalab/DA-promoC-Mod3-sprint1-VannayPaula/blob/main/Regresion%20Logistica/Lecc06-Random_Forest.ipynb)  

---    

### **Regresion Logistica Biblioteca:**

```# Tratamiento de datos
import numpy as np
import pandas as pd


# Gráficos
import matplotlib.pyplot as plt
import seaborn as sns

# Estandarización variables numéricas y Codificación variables categóricas
from sklearn.preprocessing import StandardScaler

# Gestión datos desbalanceados
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from imblearn.combine import SMOTETomek

# Para separar los datos en train y test / matriz de confusión / Modelado 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score , cohen_kappa_score, roc_curve,roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

#  Gestión de warnings
import warnings
warnings.filterwarnings("ignore")
``` 

Ejercicios realizados con la colaboración de [Giovanna Lozito](https://github.com/VannaLZ) y [Lola Rubio](https://github.com/Lolaru26) :woman_technologist: