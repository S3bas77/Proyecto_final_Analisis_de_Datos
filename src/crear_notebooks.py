import os
import json
from pathlib import Path

# Crear la carpeta notebooks si no existe (según la estructura de la imagen)
notebooks_path = Path("./notebooks")
notebooks_path.mkdir(exist_ok=True)

# Contenido del notebook 01_curation.ipynb (CORREGIDO: curation en lugar de curacion)
curation_notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# 01 - Curación y Preparación de Datos\n",
                "## Proyecto Final: Análisis de Datos\n",
                "### Objetivo: Cargar, revisar y limpiar los datasets iniciales"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import pandas as pd\n",
                "import numpy as np\n",
                "import matplotlib.pyplot as plt\n",
                "import seaborn as sns\n",
                "\n",
                "print('Librerías cargadas correctamente')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 1. Carga de Datos"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Cargar datasets\n",
                "clients = pd.read_csv('../data/clients.csv')\n",
                "projects = pd.read_csv('../data/projects.csv')\n",
                "\n",
                "print('=== INFORMACIÓN DE DATASETS ===')\n",
                "print(f'Clientes: {clients.shape} - Filas: {clients.shape[0]}, Columnas: {clients.shape[1]}')\n",
                "print(f'Proyectos: {projects.shape} - Filas: {projects.shape[0]}, Columnas: {projects.shape[1]}')\n",
                "\n",
                "print('\\n=== PRIMERAS FILAS ===')\n",
                "print('\\nClientes:')\n",
                "display(clients.head())\n",
                "print('\\nProyectos:')\n",
                "display(projects.head())"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 2. Revisión de Calidad de Datos"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "print('=== INFORMACIÓN DE TIPOS DE DATOS ===')\n",
                "print('\\nClientes:')\n",
                "clients.info()\n",
                "print('\\nProyectos:')\n",
                "projects.info()\n",
                "\n",
                "print('\\n=== VALORES NULOS ===')\n",
                "print('\\nClientes:')\n",
                "print(clients.isnull().sum())\n",
                "print('\\nProyectos:')\n",
                "print(projects.isnull().sum())"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 3. Conversión de Tipos de Datos"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Convertir fechas en proyectos\n",
                "date_columns = ['start_date', 'planned_end_date', 'actual_end_date']\n",
                "for col in date_columns:\n",
                "    projects[col] = pd.to_datetime(projects[col])\n",
                "\n",
                "print('=== CONVERSIÓN DE FECHAS COMPLETADA ===')\n",
                "print('\\nTipos de datos en proyectos después de conversión:')\n",
                "print(projects[date_columns].dtypes)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 4. Validaciones de Consistencia"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Validar que las fechas sean consistentes\n",
                "projects['duration_planned'] = (projects['planned_end_date'] - projects['start_date']).dt.days\n",
                "projects['duration_actual'] = (projects['actual_end_date'] - projects['start_date']).dt.days\n",
                "\n",
                "# Identificar proyectos con fechas inconsistentes\n",
                "invalid_dates = projects[projects['duration_actual'] < 0]\n",
                "print(f'Proyectos con fechas inconsistentes: {len(invalid_dates)}')\n",
                "\n",
                "# Validar rangos de valores\n",
                "print('\\n=== VALIDACIÓN DE RANGOS ===')\n",
                "print(f'Satisfaction score range: {clients[\"satisfaction_score\"].min()} - {clients[\"satisfaction_score\"].max()}')\n",
                "print(f'Budget range: ${projects[\"budget_usd\"].min():,.2f} - ${projects[\"budget_usd\"].max():,.2f}')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 5. Guardar Datasets Curados"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Guardar datasets procesados\n",
                "clients.to_csv('../data/clients_curated.csv', index=False)\n",
                "projects.to_csv('../data/projects_curated.csv', index=False)\n",
                "\n",
                "print('✅ Datasets curados guardados correctamente:')\n",
                "print('- ../data/clients_curated.csv')\n",
                "print('- ../data/projects_curated.csv')"
            ]
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.8.0"
        }
    }
}

# Contenido del notebook 02_eda.ipynb
eda_notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# 02 - Análisis Exploratorio de Datos (EDA)\n",
                "## Proyecto Final: Análisis de Datos\n",
                "### Objetivo: Explorar y visualizar los datos para identificar patrones y relaciones"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import pandas as pd\n",
                "import numpy as np\n",
                "import matplotlib.pyplot as plt\n",
                "import seaborn as sns\n",
                "\n",
                "# Configuración de visualización\n",
                "plt.style.use('seaborn-v0_8')\n",
                "sns.set_palette(\"husl\")\n",
                "plt.rcParams['figure.figsize'] = (12, 6)\n",
                "\n",
                "print('Librerías y configuración cargadas correctamente')"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Cargar datasets curados\n",
                "clients = pd.read_csv('../data/clients_curated.csv')\n",
                "projects = pd.read_csv('../data/projects_curated.csv')\n",
                "\n",
                "# Convertir fechas nuevamente\n",
                "date_cols = ['start_date', 'planned_end_date', 'actual_end_date']\n",
                "for col in date_cols:\n",
                "    projects[col] = pd.to_datetime(projects[col])\n",
                "\n",
                "print('Datos cargados y preparados para análisis')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 1. Análisis Univariado"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "fig, axes = plt.subplots(2, 3, figsize=(18, 10))\n",
                "\n",
                "# Distribución de satisfaction_score\n",
                "sns.histplot(data=clients, x='satisfaction_score', bins=5, ax=axes[0,0])\n",
                "axes[0,0].set_title('Distribución de Satisfacción')\n",
                "\n",
                "# Distribución de tamaño de empresa\n",
                "clients['size'].value_counts().plot(kind='bar', ax=axes[0,1])\n",
                "axes[0,1].set_title('Distribución por Tamaño de Empresa')\n",
                "\n",
                "# Distribución de industria\n",
                "clients['industry'].value_counts().plot(kind='bar', ax=axes[0,2])\n",
                "axes[0,2].set_title('Distribución por Industria')\n",
                "axes[0,2].tick_params(axis='x', rotation=45)\n",
                "\n",
                "# Distribución de presupuesto\n",
                "sns.histplot(data=projects, x='budget_usd', ax=axes[1,0])\n",
                "axes[1,0].set_title('Distribución de Presupuesto')\n",
                "\n",
                "# Distribución de estado de proyectos\n",
                "projects['status'].value_counts().plot(kind='bar', ax=axes[1,1])\n",
                "axes[1,1].set_title('Estado de Proyectos')\n",
                "\n",
                "# Distribución de complejidad\n",
                "projects['complexity'].value_counts().plot(kind='bar', ax=axes[1,2])\n",
                "axes[1,2].set_title('Complejidad de Proyectos')\n",
                "\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.8.0"
        }
    }
}

# Contenido del notebook 03_modeling.ipynb
modeling_notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# 03 - Modelado y Análisis Predictivo\n",
                "## Proyecto Final: Análisis de Datos\n",
                "### Objetivo: Desarrollar modelos predictivos para renovación de contratos y retrasos en proyectos"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import pandas as pd\n",
                "import numpy as np\n",
                "import matplotlib.pyplot as plt\n",
                "import seaborn as sns\n",
                "from sklearn.model_selection import train_test_split\n",
                "from sklearn.preprocessing import StandardScaler, OneHotEncoder\n",
                "from sklearn.compose import ColumnTransformer\n",
                "from sklearn.pipeline import Pipeline\n",
                "from sklearn.ensemble import RandomForestClassifier\n",
                "from sklearn.metrics import classification_report, roc_auc_score\n",
                "\n",
                "print('Librerías cargadas correctamente')"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Cargar datos\n",
                "clients = pd.read_csv('../data/clients_curated.csv')\n",
                "projects = pd.read_csv('../data/projects_curated.csv')\n",
                "\n",
                "print('=== DATOS CARGADOS ===')\n",
                "print(f'Clientes: {clients.shape}')\n",
                "print(f'Proyectos: {projects.shape}')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 1. Modelo: Predicción de Renovación de Contratos"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Preparar características y target\n",
                "X_contract = clients[['industry', 'size', 'region', 'tickets_opened_last_year', \n",
                "                     'avg_response_time_hours', 'satisfaction_score']]\n",
                "y_contract = clients['renewed_contract']\n",
                "\n",
                "print('=== MODELO 1: RENOVACIÓN DE CONTRATOS ===')\n",
                "print(f'Características: {X_contract.shape}')\n",
                "print(f'Target distribution: {y_contract.value_counts().to_dict()}')"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Preprocesamiento\n",
                "numeric_features = ['tickets_opened_last_year', 'avg_response_time_hours', 'satisfaction_score']\n",
                "categorical_features = ['industry', 'size', 'region']\n",
                "\n",
                "preprocessor = ColumnTransformer(\n",
                "    transformers=[\n",
                "        ('num', StandardScaler(), numeric_features),\n",
                "        ('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), categorical_features)\n",
                "    ])\n",
                "\n",
                "# Crear pipeline\n",
                "model_contract = Pipeline([\n",
                "    ('preprocessor', preprocessor),\n",
                "    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))\n",
                "])\n",
                "\n",
                "# Dividir datos\n",
                "X_train, X_test, y_train, y_test = train_test_split(X_contract, y_contract, \n",
                "                                                    test_size=0.2, random_state=42, \n",
                "                                                    stratify=y_contract)\n",
                "\n",
                "print('Datos divididos para entrenamiento y prueba')"
            ]
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.8.0"
        }
    }
}

# Contenido del notebook 04_storytelling.ipynb (NUEVO - según la estructura de la imagen)
storytelling_notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# 04 - Storytelling y Presentación de Resultados\n",
                "## Proyecto Final: Análisis de Datos\n",
                "### Objetivo: Sintetizar hallazgos y crear visualizaciones para presentación"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import pandas as pd\n",
                "import numpy as np\n",
                "import matplotlib.pyplot as plt\n",
                "import seaborn as sns\n",
                "\n",
                "# Configuración para presentación\n",
                "plt.style.use('seaborn-v0_8-whitegrid')\n",
                "sns.set_palette(\"Set2\")\n",
                "plt.rcParams['figure.figsize'] = (14, 8)\n",
                "plt.rcParams['font.size'] = 12\n",
                "\n",
                "print('Librerías cargadas para storytelling')"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Cargar todos los datos necesarios\n",
                "clients = pd.read_csv('../data/clients_curated.csv')\n",
                "projects = pd.read_csv('../data/projects_curated.csv')\n",
                "\n",
                "# Convertir fechas\n",
                "date_cols = ['start_date', 'planned_end_date', 'actual_end_date']\n",
                "for col in date_cols:\n",
                "    projects[col] = pd.to_datetime(projects[col])\n",
                "\n",
                "print('Datos cargados para análisis final')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 1. Resumen Ejecutivo de Hallazgos"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "print('=== RESUMEN EJECUTIVO ===')\n",
                "print(f'• Total de clientes analizados: {len(clients)}')\n",
                "print(f'• Total de proyectos completados: {len(projects[projects[\"status\"] == \"completed\"])}')\n",
                "print(f'• Tasa promedio de satisfacción: {clients[\"satisfaction_score\"].mean():.2f}/5')\n",
                "print(f'• Presupuesto promedio por proyecto: ${projects[\"budget_usd\"].mean():,.2f}')\n",
                "print(f'• Tasa de renovación de contratos: {clients[\"renewed_contract\"].mean()*100:.1f}%')"
            ]
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.8.0"
        }
    }
}

# Guardar los notebooks CORREGIDOS según la estructura de la imagen
notebooks_to_save = {
    "01_curation.ipynb": curation_notebook,  # CORREGIDO: curation en lugar de curacion
    "02_eda.ipynb": eda_notebook,
    "03_modeling.ipynb": modeling_notebook,
    "04_storytelling.ipynb": storytelling_notebook  # NUEVO: agregado según la estructura
}

for filename, notebook_content in notebooks_to_save.items():
    filepath = notebooks_path / filename
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(notebook_content, f, indent=2, ensure_ascii=False)
    print(f'✅ {filename} creado correctamente')

print(f'\n🎉 Todos los notebooks creados en: {notebooks_path}')
print('📁 Archivos creados:')
for file in notebooks_path.glob('*.ipynb'):
    print(f'   - {file.name}')