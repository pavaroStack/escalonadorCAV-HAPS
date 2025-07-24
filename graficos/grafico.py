import os
import json
import matplotlib.pyplot as plt
from collections import defaultdict

# MODIFICADO: Agora busca na mesma pasta do script
PASTA = "."
PASTA_SAIDA = "graficos"
os.makedirs(PASTA_SAIDA, exist_ok=True)

dados = defaultdict(list)
arquivos_validos = 0

print(f"Buscando arquivos JSON em: {PASTA}")

for arquivo in os.listdir(PASTA):
    if arquivo.endswith(".json"):
        caminho = os.path.join(PASTA, arquivo)
        print(f"Processando: {arquivo}")
        try:
            with open(caminho, "r", encoding="utf-8") as f:
                conteudo = json.load(f)
                
                # Verificação mais flexível da estrutura
                if isinstance(conteudo, list):
                    for resultado in conteudo:
                        if isinstance(resultado, dict) and "nome" in resultado:
                            nome = resultado["nome"]
                            # Coleta dados com tratamento de valores ausentes
                            dados[nome].append({
                                "turnaround_medio": resultado.get("turnaround_medio", 0),
                                "deadlines_perdidos": resultado.get("deadlines_perdidos", 0),
                                "sobrecarga_total": resultado.get("sobrecarga_total", 0)
                            })
                    arquivos_validos += 1
                    print(f"  ✅ Estrutura válida - {len(conteudo)} registros")
                else:
                    print(f"  ⚠️ Formato inválido (não é lista) - Ignorado")

        except json.JSONDecodeError:
            print(f"  ❌ ERRO: Arquivo JSON inválido ou corrompido")
        except Exception as e:
            print(f"  ❌ ERRO inesperado: {str(e)}")

print(f"\nTotal de arquivos válidos processados: {arquivos_validos}")

if not dados:
    print("\nNenhum dado válido encontrado. Possíveis causas:")
    print("- Arquivos JSON na pasta errada (pasta atual: {PASTA})")
    print("- Estrutura diferente do esperado (deve ser lista de dicionários)")
    print("- Nenhum arquivo .json encontrado")
    exit()


# Calcula as médias por escalonador
medias = {}
for nome, registros in dados.items():
    total_registros = len(registros)
    medias[nome] = {
        "turnaround_medio": sum(r["turnaround_medio"] for r in registros) / total_registros,
        "deadlines_perdidos": sum(r["deadlines_perdidos"] for r in registros) / total_registros,
        "sobrecarga_total": sum(r["sobrecarga_total"] for r in registros) / total_registros
    }

# Ordenação lógica dos escalonadores (personalizável)
ORDEM_ESCALONADORES = ["HAPS-AI", "FIFO", "Round Robin", "Prioridade Estática"]
nomes = [n for n in ORDEM_ESCALONADORES if n in medias]

# Extração dos dados para gráficos
turnarounds = [medias[n]["turnaround_medio"] for n in nomes]
deadlines = [medias[n]["deadlines_perdidos"] for n in nomes]
sobrecargas = [medias[n]["sobrecarga_total"] for n in nomes]

# Função para salvar gráficos com formatação melhorada
def salvar_grafico(titulo, valores, ylabel, nome_arquivo, formato="%.2f"):
    plt.figure(figsize=(10, 6))
    bars = plt.bar(nomes, valores, color=["#4C72B0", "#55A868", "#C44E52", "#8172B2"])
    
    # Adiciona valores nas barras
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 formato % height, ha='center', va='bottom')
    
    plt.title(titulo, fontsize=14)
    plt.ylabel(ylabel, fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    caminho_saida = os.path.join(PASTA_SAIDA, nome_arquivo)
    plt.savefig(caminho_saida, dpi=300)
    plt.close()
    print(f"✔ Gráfico salvo: {caminho_saida}")

# Gera os gráficos com formatação adequada
salvar_grafico("Turnaround Médio por Escalonador", turnarounds, 
               "Tempo (unidades)", "turnaround_medio.png")

salvar_grafico("Deadlines Perdidos por Escalonador", deadlines, 
               "Percentual (%)", "deadlines_perdidos.png", "%.1f%%")

salvar_grafico("Sobrecarga Total por Escalonador", sobrecargas, 
               "Tempo (unidades)", "sobrecarga_total.png")

print("\nProcesso concluído com sucesso!")
