import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib
import os

print("Iniciando treinamento do modelo HAPS-AI...")
CSV_FILE = 'haps_training_data.csv'

if not os.path.exists(CSV_FILE):
    print(f"❌ Erro: Arquivo '{CSV_FILE}' não encontrado. Execute haps_ai.py para gerar os dados.")
    exit()

df = pd.read_csv(CSV_FILE)
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)

if len(df) < 100:
    print(f"⚠️ Aviso: Dataset pequeno ({len(df)} amostras). O modelo pode não ser preciso.")

features = [
    'tempo_restante_execucao', 'tempo_decorrido', 'tempo_restante_deadline',
    'total_preempcoes', 'prioridade_base', 'tipo_processo', 'ciclo'
]
target = 'deadline_perdido'

X = df[features]
y = df[target]

if len(y.unique()) < 2:
    print("❌ Erro: O dataset de treino só contém uma classe. O modelo não pode aprender.")
    exit()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced', max_depth=10, min_samples_leaf=5)
model.fit(X_train, y_train)

print("\n--- Avaliação do Modelo ---")
print(classification_report(y_test, model.predict(X_test), zero_division=0))

joblib.dump(model, 'haps_model.joblib')
print("\n✅ Modelo treinado e salvo com sucesso como 'haps_model.joblib'!")