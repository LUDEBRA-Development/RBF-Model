# 🧠 Red de Función de Base Radial (RBF) — Clasificación Supervisada

> Implementación completa de una Red Neuronal RBF para clasificación multiclase, desarrollada en Python con Jupyter Notebook. Cubre los 10 pasos del pipeline: desde la carga de datos hasta el ajuste de hiperparámetros.

---

## 📋 Tabla de Contenidos

- [Descripción del Proyecto](#descripción-del-proyecto)
- [¿Qué es una Red RBF?](#qué-es-una-red-rbf)
- [Estructura del Repositorio](#estructura-del-repositorio)
- [Datasets](#datasets)
- [Requisitos e Instalación](#requisitos-e-instalación)
- [Cómo Ejecutar el Notebook](#cómo-ejecutar-el-notebook)
- [Pipeline — Los 10 Pasos](#pipeline--los-10-pasos)
- [Parámetros Configurables](#parámetros-configurables)
- [Métricas de Evaluación](#métricas-de-evaluación)
- [Resultados Obtenidos](#resultados-obtenidos)
- [Guía de Interpretación](#guía-de-interpretación)
- [Tecnologías Utilizadas](#tecnologías-utilizadas)

---

## Descripción del Proyecto

Este proyecto implementa una **Red Neuronal de Función de Base Radial (RBF)** desde cero para resolver problemas de clasificación supervisada con 2, 3 y 4 clases. Fue desarrollado como actividad grupal del curso de Inteligencia Artificial.

La implementación sigue un pipeline estructurado en **10 pasos** que abarca desde la limpieza y análisis exploratorio de los datos hasta el re-entrenamiento del modelo con parámetros ajustados, pasando por el cálculo de métricas completas y la visualización de resultados.

---

## ¿Qué es una Red RBF?

Una **Red de Función de Base Radial** es un tipo de red neuronal de tres capas que clasifica datos basándose en la **distancia** entre un punto de entrada y un conjunto de puntos de referencia llamados **centros**.

```
[Capa de Entrada] ──→ [Capa RBF (centros + gaussiana)] ──→ [Capa de Salida]
    (variables x)          (distancias + activación)          (clase predicha)
```

### Funcionamiento paso a paso

**1. Capa de Entrada**
Recibe las variables del dato a clasificar (x1, x2, x3, ...).

**2. Capa Oculta RBF**
- Se identifican **centros** mediante el algoritmo K-Means sobre los datos de entrenamiento.
- Para cada dato, se calcula su **distancia euclidiana** a cada centro.
- Esa distancia se transforma con la **función gaussiana**:

$$\phi(x, c_j) = e^{-\frac{||x - c_j||^2}{2\sigma^2}}$$

donde:
- `x` = dato de entrada
- `c_j` = centro j-ésimo
- `σ` (sigma) = ancho de la campana gaussiana (controla el radio de influencia)
- El resultado es un valor entre 0 y 1: **1** si el dato está muy cerca del centro, **0** si está muy lejos.

**3. Capa de Salida**
Una Regresión Logística toma todas las activaciones gaussianas y decide a qué clase pertenece el dato.

### Ventajas de la RBF
- Entrenamiento rápido comparado con otras redes neuronales
- Funciona bien cuando los datos tienen **grupos bien definidos**
- Interpretable: los centros representan "prototipos" de cada región del espacio

---

## Estructura del Repositorio

```
📁 rbf-clasificacion/
│
├── 📓 RBF_Clasificacion_Completo.ipynb   # Notebook principal con todo el pipeline
│
├── 📊 dataset_rbf_1.json                 # Dataset binario (2 clases, 1000 muestras)
├── 📊 dataset_rbf_2.json                 # Dataset 3 clases (1050 muestras)
├── 📊 dataset_rbf_3.json                 # Dataset 4 clases (1200 muestras)
│
└── 📄 README.md                          # Este archivo
```

---

## Datasets

El proyecto trabaja con tres datasets en formato JSON. Cada archivo tiene la siguiente estructura:

```json
{
  "dataset": "Nombre del dataset",
  "features": ["x1", "x2", "x3", "x4"],
  "data": [
    { "input": [0.375, 1.567, 1.710, 2.307], "output": 0 },
    { "input": [1.394, -0.906, 1.288, -0.134], "output": 1 }
  ]
}
```

### Dataset 1 — Clasificación Binaria

| Característica | Valor |
|---|---|
| Clases | 2 (normal vs alterado) |
| Variables | 4 (x1, x2, x3, x4) |
| Total de muestras | 1000 |
| Muestras por clase | 500 cada una |
| Caso de uso | Diagnóstico médico: condición normal (0) vs alterada (1) |

### Dataset 2 — Tres Clases

| Característica | Valor |
|---|---|
| Clases | 3 (baja / media / alta producción) |
| Variables | 3 (x1, x2, x3) |
| Total de muestras | 1050 |
| Muestras por clase | 350 cada una |
| Caso de uso | Clasificación de niveles de producción industrial |

### Dataset 3 — Cuatro Clases

| Característica | Valor |
|---|---|
| Clases | 4 (riesgo bajo / medio-bajo / medio-alto / alto) |
| Variables | 4 (x1, x2, x3, x4) |
| Total de muestras | 1200 |
| Muestras por clase | 300 cada una |
| Caso de uso | Clasificación de niveles de riesgo |

---

## Requisitos e Instalación

### Requisitos del sistema
- Python 3.8 o superior
- Jupyter Notebook o JupyterLab

### Instalación de dependencias

Todas las librerías usadas son estándar del ecosistema de ciencia de datos en Python:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn jupyter
```

O si usas Anaconda (recomendado), ya vienen incluidas por defecto. Solo abre Anaconda Navigator y lanza Jupyter Notebook.

### Verificar instalación

```python
import numpy, pandas, matplotlib, seaborn, sklearn
print("Todo instalado correctamente ✅")
```

---

## Cómo Ejecutar el Notebook

1. Clona o descarga este repositorio en tu computador.

2. Asegúrate de que los tres archivos JSON estén en la **misma carpeta** que el notebook.

3. Abre una terminal en esa carpeta y ejecuta:
   ```bash
   jupyter notebook
   ```

4. Abre el archivo `RBF_Clasificacion_Completo.ipynb`.

5. Ejecuta las celdas en orden con `Shift + Enter` o usa el botón **"Run All"** del menú.

6. Las celdas marcadas con `✏️ PUEDES CAMBIAR ESTOS VALORES` son las únicas que debes modificar para experimentar con los parámetros.

> **Importante:** Si los archivos JSON están en una carpeta diferente al notebook, actualiza la variable `RUTA_DS1`, `RUTA_DS2` o `RUTA_DS3` con la ruta correcta.

---

## Pipeline — Los 10 Pasos

El notebook implementa los 10 pasos definidos en el enunciado del examen:

### Paso 1 — Carga del Dataset
Los archivos JSON se leen con la librería `json` de Python. Se extraen las matrices `X` (variables de entrada) y `y` (etiquetas de clase) como arrays de NumPy.

### Paso 2 — Parámetros de Entrada
Se definen en un diccionario editable antes de cada ejecución:
- Número de centros RBF
- Sigma (ancho gaussiano)
- C (regularización del clasificador)
- Tipo de partición

### Paso 3 — Estadística Descriptiva
Se calculan y muestran:
- Número de muestras totales y por clase
- Media, desviación estándar, mínimo, máximo y cuartiles por variable
- Media de cada variable separada por clase (útil para ver qué tan separados están los grupos)

### Paso 4 — Particionamiento del Dataset
Se implementan dos esquemas de partición usando `train_test_split` con `stratify=y` (garantiza la misma proporción de clases en cada partición):

| Esquema | Entrenamiento | Validación | Prueba |
|---------|--------------|-----------|--------|
| 80-10-10 | 80% | 10% | 10% |
| 70-15-15 | 70% | 15% | 15% |

### Paso 5 — Parámetros de Entrenamiento
Se confirman los parámetros antes de iniciar el entrenamiento y se explica el rol de cada uno.

### Paso 6 — Entrenamiento del Modelo RBF
El entrenamiento ocurre en 5 sub-pasos internos:
1. Normalización de datos con `StandardScaler` (media=0, desv_std=1)
2. Búsqueda de centros con `KMeans`
3. Cálculo de sigma (automático o manual)
4. Cómputo de activaciones gaussianas para todos los datos de entrenamiento
5. Entrenamiento del clasificador de salida (`LogisticRegression`) sobre las activaciones

### Paso 7 — Simulación
El modelo entrenado predice las clases de los tres conjuntos (entrenamiento, validación y prueba) usando el método `predict()`.

### Paso 8 — Métricas de Evaluación
Se calculan las cuatro métricas principales para cada conjunto:

- **Exactitud (Accuracy):** proporción de predicciones correctas sobre el total
- **Precisión (Precision):** de todos los casos predichos como clase X, cuántos lo eran realmente
- **Sensibilidad (Recall):** de todos los casos reales de clase X, cuántos fueron detectados
- **F1 Score:** media armónica de precisión y sensibilidad

Además se genera el reporte completo por clase con `classification_report`.

### Paso 9 — Gráficas de Resultados
Se generan cuatro tipos de visualizaciones:
- **Dispersión de datos y centros:** proyección 2D de los datos con los centros RBF marcados
- **Matrices de confusión:** para los tres conjuntos (train/val/test)
- **Barras de métricas:** comparación de las 4 métricas entre los tres conjuntos
- **Activaciones RBF:** heatmap de qué tanto activa cada neurona, y activación promedio por centro

### Paso 10 — Ajuste y Re-entrenamiento
Se ejecuta una segunda versión del modelo con parámetros distintos y se comparan ambas versiones en una gráfica de barras agrupadas para los tres conjuntos.

---

## Parámetros Configurables

Estos son los parámetros que puedes modificar para experimentar:

### `n_centros` — Número de centros RBF
Determina cuántas neuronas tiene la capa oculta. Cada centro es un punto representativo del espacio de datos encontrado por K-Means.

| Valor | Efecto |
|-------|--------|
| Muy bajo | El modelo es demasiado simple, puede no capturar la estructura |
| Apropiado | Buen balance entre aprendizaje y generalización |
| Muy alto | Riesgo de sobreajuste (memoriza en vez de aprender) |

**Recomendación inicial:**
- Dataset 2 clases → 6 a 12 centros
- Dataset 3 clases → 9 a 18 centros
- Dataset 4 clases → 12 a 24 centros

### `sigma` — Ancho de la función gaussiana
Controla el radio de influencia de cada neurona.

| Valor | Efecto |
|-------|--------|
| `None` | Se calcula automáticamente (recomendado para empezar) |
| Valor grande | La gaussiana es muy ancha, influye lejos del centro |
| Valor pequeño | La gaussiana es estrecha, solo influye cerca del centro |

### `C` — Regularización del clasificador
Parámetro de la Regresión Logística que controla la complejidad de la frontera de decisión.

| Valor | Efecto |
|-------|--------|
| C pequeño (0.01–0.1) | Más regularización, frontera más suave |
| C = 1.0 | Balance estándar |
| C grande (5–100) | Menos regularización, frontera más ajustada a los datos |

### `particion` — Esquema de división del dataset
Dos opciones: `'80-10-10'` o `'70-15-15'`.

---

## Métricas de Evaluación

### Matriz de Confusión
Tabla que muestra cuántas muestras de cada clase real fueron predichas como cada clase posible.

```
                 Predicho
                 Clase 0   Clase 1
Real   Clase 0  [   TP   |   FN  ]
       Clase 1  [   FP   |   TN  ]
```

- **Diagonal principal** → predicciones correctas (verde)
- **Fuera de diagonal** → errores o confusiones (rojo)

### Fórmulas de las métricas

$$\text{Exactitud} = \frac{TP + TN}{TP + TN + FP + FN}$$

$$\text{Precisión} = \frac{TP}{TP + FP}$$

$$\text{Sensibilidad} = \frac{TP}{TP + FN}$$

$$\text{F1 Score} = 2 \cdot \frac{\text{Precisión} \times \text{Sensibilidad}}{\text{Precisión} + \text{Sensibilidad}}$$

### Escala de referencia

| Rango | Calidad del modelo |
|-------|-------------------|
| > 0.95 | Excelente |
| 0.85 – 0.95 | Bueno |
| 0.70 – 0.85 | Aceptable |
| < 0.70 | Necesita mejoras |

---

## Resultados Obtenidos

Los tres datasets presentan clusters bien definidos en el espacio de características, lo que permite a la RBF obtener resultados de clasificación muy altos desde la primera versión del modelo.

| Dataset | Clases | Centros | Accuracy Train | Accuracy Val | Accuracy Test |
|---------|--------|---------|---------------|-------------|--------------|
| DS1 — V1 | 2 | 8 | ~1.000 | ~1.000 | ~1.000 |
| DS1 — V2 | 2 | 16 | ~1.000 | ~1.000 | ~1.000 |
| DS2 — V1 | 3 | 12 | ~1.000 | ~1.000 | ~1.000 |
| DS2 — V2 | 3 | 24 | ~1.000 | ~1.000 | ~1.000 |
| DS3 — V1 | 4 | 16 | ~1.000 | ~1.000 | ~1.000 |
| DS3 — V2 | 4 | 32 | ~1.000 | ~1.000 | ~1.000 |

> Los resultados exactos pueden variar ligeramente dependiendo del `random_state` y los parámetros usados. Los valores de la tabla corresponden a la ejecución con los parámetros por defecto del notebook.

---

## Guía de Interpretación

### Diagnóstico del modelo

| Síntoma | Diagnóstico | Solución |
|---------|-------------|---------|
| Train alto, Val/Test bajo | Sobreajuste (memoriza) | Reducir `n_centros`, reducir `C` |
| Train bajo, Val/Test bajo | Subajuste (no aprende) | Aumentar `n_centros`, aumentar `C` |
| Train ≈ Val ≈ Test | Modelo bien generalizado ✅ | Mantener parámetros |
| Val alto, Test bajo | Fuga de datos en validación | Revisar el particionamiento |

### Cómo leer la gráfica de centros
Los puntos `★` (estrellas negras) son los centros RBF encontrados por K-Means. Deberían estar distribuidos entre los distintos grupos de datos. Si todos los centros están en un solo grupo, aumenta `n_centros`.

### Cómo leer el heatmap de activaciones
Cada fila es un dato de prueba y cada columna es una neurona RBF. El color amarillo (valor cercano a 1) indica que ese dato está muy cerca de ese centro. Cada dato debería activar fuertemente 1 o 2 neuronas y el resto deberían estar apagadas (azul).

---

## Tecnologías Utilizadas

| Librería | Versión mínima | Uso |
|----------|---------------|-----|
| Python | 3.8+ | Lenguaje base |
| NumPy | 1.21+ | Operaciones matriciales y cálculo de distancias |
| Pandas | 1.3+ | Manejo de tablas y estadística descriptiva |
| Matplotlib | 3.4+ | Gráficas base |
| Seaborn | 0.11+ | Heatmaps y gráficas de matrices de confusión |
| scikit-learn | 0.24+ | KMeans, StandardScaler, LogisticRegression, métricas |
| Jupyter Notebook | 6.0+ | Entorno de ejecución interactivo |

---

## Autores

Proyecto desarrollado como examen práctico del curso de **Inteligencia Artificial**.

@Moo0nchild

---
