import os
import json
import matplotlib.pyplot as plt
from collections import defaultdict

PASTA = "../"  

PASTA_SAIDA = "graficos"
os.makedirs(PASTA_SAIDA, exist_ok=True)

dados = defaultdict(list)

# Lê todos os arquivos JSON na pasta
for arquivo in os.listdir(PASTA):
    if arquivo.endswith(".json"):
        caminho = os.path.join(PASTA, arquivo)
        try:
            with open(caminho, "r", encoding="utf-8") as f:
                conteudo = json.load(f)

                # Lista de resultados
                if isinstance(conteudo, list):
                    for resultado in conteudo:
                        if isinstance(resultado, dict) and "nome" in resultado:
                            nome = resultado["nome"]
                            dados[nome].append({
                                "turnaround_medio": resultado.get("turnaround_medio", 0),
                                "deadlines_perdidos": resultado.get("deadlines_perdidos", 0),
                                "sobrecarga_total": resultado.get("sobrecarga_total", 0)
                            })

                # Um único dicionário
                elif isinstance(conteudo, dict) and "nome" in conteudo:
                    nome = conteudo["nome"]
                    dados[nome].append({
                        "turnaround_medio": conteudo.get("turnaround_medio", 0),
                        "deadlines_perdidos": conteudo.get("deadlines_perdidos", 0),
                        "sobrecarga_total": conteudo.get("sobrecarga_total", 0)
                    })

        except Exception as e:
            print(f"[ERRO] Falha ao ler {arquivo}: {e}")

# Calcula as médias por escalonador
medias = {}
for nome, registros in dados.items():
    total_turnaround = sum(r["turnaround_medio"] for r in registros)
    total_deadlines = sum(r["deadlines_perdidos"] for r in registros)
    total_sobrecarga = sum(r["sobrecarga_total"] for r in registros)
    n = len(registros)
    medias[nome] = {
        "turnaround_medio": total_turnaround / n,
        "deadlines_perdidos": total_deadlines / n,
        "sobrecarga_total": total_sobrecarga / n
    }

# Ordena os nomes dos escalonadores
nomes = sorted(medias.keys())

# Extrai os dados para os gráficos
turnarounds = [medias[n]["turnaround_medio"] for n in nomes]
deadlines = [medias[n]["deadlines_perdidos"] for n in nomes]
sobrecargas = [medias[n]["sobrecarga_total"] for n in nomes]

# Função para salvar gráficos
def salvar_grafico(titulo, valores, ylabel, nome_arquivo):
    plt.figure(figsize=(8, 5))
    plt.bar(nomes, valores, color="skyblue")
    plt.title(titulo)
    plt.ylabel(ylabel)
    plt.xticks(rotation=20)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    caminho_saida = os.path.join(PASTA_SAIDA, nome_arquivo)
    plt.savefig(caminho_saida)
    plt.close()
    print(f"✔ Gráfico salvo em: {caminho_saida}")

# Salvar os três gráficos
salvar_grafico("Turnaround Médio por Escalonador", turnarounds, "Turnaround Médio", "turnaround_medio.png")
salvar_grafico("Deadlines Perdidos (%) por Escalonador", deadlines, "Deadlines Perdidos (%)", "deadlines_perdidos.png")
salvar_grafico("Sobrecarga Total por Escalonador", sobrecargas, "Sobrecarga Total", "sobrecarga_total.png")
