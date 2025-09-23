import json
from pathlib import Path

# Crear carpeta notebooks
notebooks_path = Path("./notebooks")
notebooks_path.mkdir(exist_ok=True)

# Función para crear una celda de manera correcta
def create_cell(cell_type, source, metadata=None):
    cell = {
        "cell_type": cell_type,
        "metadata": metadata or {},
        "source": source if isinstance(source, list) else [source]
    }
    if cell_type == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    return cell

# Notebook 01_curacion.ipynb
curacion_cells = [
    create_cell("markdown", [
        "# 01 - Curación y Preparación de Datos\n",
        "## Proyecto Final: Análisis de Datos\n",
        "### Objetivo: Cargar, revisar y limpiar los datasets iniciales"
    ]),
    
    create_cell("code", [
        "import pandas as pd\n",
        "import numpy as np\n",
        "import matplotlib.pyplot as plt\n",
        "import seaborn as sns\n",
        "\n",
        "print('Librerías cargadas correctamente')"
    ]),
    
    create_cell("markdown", [
        "### 1. Carga de Datos"
    ]),
    
    create_cell("code", [
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
    ])
]

# Notebook 02_eda.ipynb
eda_cells = [
    create_cell("markdown", [
        "# 02 - Análisis Exploratorio de Datos (EDA)\n",
        "## Proyecto Final: Análisis de Datos\n",
        "### Objetivo: Explorar y visualizar los datos para identificar patrones y relaciones"
    ]),
    
    create_cell("code", [
        "import pandas as pd\n",
        "import numpy as np\n",
        "import matplotlib.pyplot as plt\n",
        "import seaborn as sns\n",
        "\n",
        "# Configuración de visualización\n",
        "plt.style.use('default')\n",
        "sns.set_palette(\"husl\")\n",
        "plt.rcParams['figure.figsize'] = (12, 6)\n",
        "\n",
        "print('Librerías y configuración cargadas correctamente')"
    ]),
    
    create_cell("code", [
        "# Cargar datasets curados\n",
        "clients = pd.read_csv('../data/clients_curated.csv')\n",
        "projects = pd.read_csv('../data/projects_curated.csv')\n",
        "\n",
        "# Convertir fechas\n",
        "date_cols = ['start_date', 'planned_end_date', 'actual_end_date']\n",
        "for col in date_cols:\n",
        "    projects[col] = pd.to_datetime(projects[col])\n",
        "\n",
        "print('Datos cargados y preparados para análisis')"
    ])
]

# Notebook 03_modeling.ipynb
modeling_cells = [
    create_cell("markdown", [
        "# 03 - Modelado y Análisis Predictivo\n",
        "## Proyecto Final: Análisis de Datos\n",
        "### Objetivo: Desarrollar modelos predictivos"
    ]),
    
    create_cell("code", [
        "import pandas as pd\n",
        "import numpy as np\n",
        "from sklearn.model_selection import train_test_split\n",
        "from sklearn.preprocessing import StandardScaler, OneHotEncoder\n",
        "from sklearn.compose import ColumnTransformer\n",
        "from sklearn.pipeline import Pipeline\n",
        "from sklearn.ensemble import RandomForestClassifier\n",
        "\n",
        "print('Librerías cargadas correctamente')"
    ]),
    
    create_cell("code", [
        "# Cargar datos\n",
        "clients = pd.read_csv('../data/clients_curated.csv')\n",
        "projects = pd.read_csv('../data/projects_curated.csv')\n",
        "\n",
        "print('=== DATOS CARGADOS ===')\n",
        "print(f'Clientes: {clients.shape}')\n",
        "print(f'Proyectos: {projects.shape}')"
    ])
]

# Crear la estructura completa del notebook
def create_notebook(cells, notebook_name):
    notebook = {
        "cells": cells,
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
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }
    
    filepath = notebooks_path / notebook_name
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=2, ensure_ascii=False)
    
    print(f'✅ {notebook_name} creado correctamente')
    return filepath

# Crear los notebooks
print("Creando notebooks...")
create_notebook(curacion_cells, "01_curacion.ipynb")
create_notebook(eda_cells, "02_eda.ipynb")
create_notebook(modeling_cells, "03_modeling.ipynb")

print(f'\n🎉 Todos los notebooks creados en: {notebooks_path.absolute()}')