import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
import os

# Usar la ruta actual del proyecto como base
base_path = "./"  # Cambiado de "/mnt/data/" a "./"

# Las carpetas ya están creadas manualmente, así que comentamos esta parte
# folders = [
#     "data", "notebooks", "src", "docs", "anexos/modelos"
# ]
# for folder in folders:
#     os.makedirs(os.path.join(base_path, folder), exist_ok=True)

# Semilla para reproducibilidad
np.random.seed(42)
random.seed(42)

# ---- Generación de tabla de clientes ----
N_clients = 100
client_ids = [f"C{1000+i}" for i in range(N_clients)]
industries = ['Retail', 'Finanzas', 'Gobierno', 'Educación', 'Salud', 'Otros']
sizes = ['Pequeña', 'Mediana', 'Grande']
regions = ['La Paz', 'Cochabamba', 'Santa Cruz', 'Oruro', 'Potosí']

clients = pd.DataFrame({
    'client_id': client_ids,
    'industry': np.random.choice(industries, size=N_clients, p=[0.2,0.2,0.2,0.15,0.15,0.1]),
    'size': np.random.choice(sizes, size=N_clients, p=[0.5,0.3,0.2]),
    'region': np.random.choice(regions, size=N_clients),
    'support_contract': np.random.choice([0,1], size=N_clients, p=[0.4,0.6]),
    'tickets_opened_last_year': np.random.poisson(lam=20, size=N_clients),
    'avg_response_time_hours': np.round(np.random.normal(loc=24, scale=8, size=N_clients),1),
    'satisfaction_score': np.random.randint(1,6, size=N_clients)
})

# Regla para renovación: depende de satisfacción y tiempo de respuesta
clients['renewed_contract'] = np.where(
    (clients['support_contract']==1) & (clients['satisfaction_score']>=3) & (clients['avg_response_time_hours']<=30),
    np.random.choice([1,0], size=N_clients, p=[0.8,0.2]),
    np.random.choice([1,0], size=N_clients, p=[0.3,0.7])
)

# ---- Generación de tabla de proyectos ----
N_projects = 200
project_ids = [f"P{2000+i}" for i in range(N_projects)]
start_dates = [datetime(2023,1,1) + timedelta(days=int(np.random.rand()*730)) for _ in range(N_projects)]
planned_durations = np.random.randint(30,180, size=N_projects)  # en días
planned_end_dates = [start_dates[i] + timedelta(days=int(planned_durations[i])) for i in range(N_projects)]

# Calcular fechas reales con retraso o no
delays = np.random.choice([0,1], size=N_projects, p=[0.65,0.35])
actual_end_dates = []
for i in range(N_projects):
    if delays[i]==0:
        # Entregado a tiempo (±10% margen)
        delta = int(planned_durations[i] * np.random.uniform(0.9,1.1))
    else:
        # Retrasado (20% a 100% más largo)
        delta = int(planned_durations[i] * np.random.uniform(1.2,2.0))
    actual_end_dates.append(start_dates[i] + timedelta(days=delta))

projects = pd.DataFrame({
    'project_id': project_ids,
    'client_id': np.random.choice(client_ids, size=N_projects),
    'start_date': start_dates,
    'planned_end_date': planned_end_dates,
    'actual_end_date': actual_end_dates,
    'budget_usd': np.round(np.random.normal(20000, 7000, size=N_projects),2),
    'dev_team_size': np.random.randint(2,15, size=N_projects),
    'complexity': np.random.choice(['Baja','Media','Alta'], size=N_projects, p=[0.3,0.5,0.2]),
    'status': np.where(delays==0,'On-time','Delayed')
})

# Ajuste del costo final
projects['final_cost_usd'] = np.round(
    projects['budget_usd'] * np.where(projects['status']=='On-time',
                                      np.random.uniform(0.9,1.1,size=N_projects),
                                      np.random.uniform(1.1,1.5,size=N_projects)),2
)

# Guardar CSVs (ahora en la carpeta data de tu proyecto)
clients.to_csv(os.path.join(base_path,"data","clients.csv"), index=False)
projects.to_csv(os.path.join(base_path,"data","projects.csv"), index=False)

print("Datasets generados exitosamente!")
print(f"Archivos guardados en: {os.path.join(base_path, 'data')}")
print(f"Clientes: {len(clients)} registros")
print(f"Proyectos: {len(projects)} registros")