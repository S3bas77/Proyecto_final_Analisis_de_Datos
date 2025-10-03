# Proyecto Final - Minor en Analítica de Datos
**Universidad Privada Boliviana (UPB)**  
Autor: Sebastián Pablo Chacón Mendoza  

## Descripción
Este proyecto aplica la metodología CRISP-DM para analizar la gestión de proyectos y la retención de clientes en **OS Bolivia Software Factory**, empresa real de Santa Cruz, Bolivia.  
Los datos utilizados fueron **simulados en Python** con reglas de negocio inspiradas en contextos realistas, garantizando privacidad y ausencia de datos sensibles.  

## 📂 Estructura del Proyecto
```
PROYECTO_FINAL_ANALITICA/
├── data/
│   ├── clients.csv              # Dataset original clientes (simulado)
│   ├── projects.csv             # Dataset original proyectos (simulado)
│   ├── clients_curated.csv      # Dataset limpiado clientes
│   └── projects_curated.csv     # Dataset limpiado proyectos
├── notebooks/
│   ├── 01_curation.ipynb        # Curación y preparación datos
│   ├── 02_eda.ipynb             # Análisis exploratorio (EDA)
│   ├── 03_modeling.ipynb        # Modelado predictivo
│   └── 04_storytelling.ipynb    # Storytelling e insights
├── src/
│   └── generar_dataset.py       # Script para generar datasets sintéticos
├── docs/
│   ├── images/                  # Figuras y gráficas usadas en informe
│   └── executive_summary.json   # Resumen ejecutivo
├── requirements.txt             # Dependencias del proyecto
└── README.md                    # Documentación del proyecto
```

## Instalación
1. Clonar este repositorio
2. Crear un entorno virtual (opcional pero recomendado):
   ```bash
   python3 -m venv venv
   source venv/bin/activate   # en Ubuntu
   ```
3. Instalar dependencias:
   ```bash
   pip install -r requirements.txt
   ```

## Uso
1. Generar datasets simulados:
   ```bash
   python src/generar_dataset.py
   ```
   Esto guarda `clients.csv` y `projects.csv` en la carpeta `data/`.

2. Abrir los notebooks con Jupyter:
   ```bash
   jupyter notebook
   ```

3. Ejecutar en orden:
   - `01_curation.ipynb`
   - `02_eda.ipynb`
   - `03_modeling.ipynb`
   - `04_storytelling.ipynb`

## 📊 Herramientas
- Python 3.10+
- Numpy, Pandas, Scikit-learn
- Matplotlib, Seaborn
- Jupyter Notebook
- VSCode en Ubuntu 22.04

## 📜 Nota Ética
Aunque la empresa **OS Bolivia** es real, los datos utilizados son **totalmente simulados** con fines académicos.  
El proyecto garantiza el cumplimiento de buenas prácticas en privacidad y transparencia.