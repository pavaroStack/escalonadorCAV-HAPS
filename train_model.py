import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

print("Iniciando treinamento do modelo HAPS-AI...")

try:
    df = pd.read_csv('haps_training_data.csv')
    df.dropna(inplace=True) 
    df.drop_duplicates(inplace=True) 
except FileNotFoundError:
    print("❌ Erro: Arquivo 'haps_training_data.csv' não encontrado. Execute o haps_ai.py primeiro para gerar os dados.")
    exit()

if len(df) < 50:
    print(f"⚠️ Aviso: Dataset pequeno ({len(df)} amostras). O modelo pode não ser preciso. Rode mais simulações.")

# 2. Definir Features (X) e Target (y)
features = ['duracao_total', 'prioridade_base', 'tipo_processo', 'total_preempcoes', 'turnaround_final', 'deadline_final']
target = 'deadline_perdido'

X = df[features]
y = df[target]

if len(y.unique()) < 2:
    print("❌ Erro: O dataset de treino só contém uma classe (todas as tarefas falharam ou todas tiveram sucesso). O modelo não pode ser treinado.")
    exit()

# 3. Dividir em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# 4. Treinar o modelo Random Forest
# class_weight='balanced' é crucial para dados desbalanceados
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced', max_depth=10)
model.fit(X_train, y_train)

# 5. Avaliar o modelo
print("\n--- Avaliação do Modelo ---")
accuracy = model.score(X_test, y_test)
print(f"Acurácia no set de teste: {accuracy:.2f}")
print("\nRelatório de Classificação:")
print(classification_report(y_test, model.predict(X_test)))

# 6. Salvar o modelo treinado
joblib.dump(model, 'haps_model.joblib')
print("\n✅ Modelo treinado e salvo com sucesso como 'haps_model.joblib'!")