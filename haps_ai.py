import joblib
import pandas as pd
import numpy as np
import random
import time
import copy
import os
import heapq
from collections import deque
from abc import ABC, abstractmethod
from enum import Enum
from typing import List,  Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

# ==============================================================================
# CLASSES DE CONTEXTO E ENUMERAÇÕES
# ==============================================================================

class TipoProcessoCAV(Enum):
    """Tipos de processos CAV com seus pesos base de prioridade"""
    SEGURANCA_CRITICA = 100
    TEMPO_REAL = 80
    NAVEGACAO = 60
    CONFORTO = 40
    DIAGNOSTICO = 20

class CondicaoClimatica(Enum):
    """Condições climáticas que afetam a condução"""
    CLARO = "claro"
    CHUVA = "chuva"
    NEVE = "neve"
    NEBLINA = "neblina"

class TipoVia(Enum):
    """Tipos de via de condução"""
    URBANA = "urbana"
    RODOVIA = "rodovia"
    RURAL = "rural"

class DensidadeTrafego(Enum):
    """Densidade de tráfego na via"""
    BAIXA = "baixa"
    MEDIA = "media"
    ALTA = "alta"
    CONGESTIONAMENTO = "congestionamento"

@dataclass
class ContextoConducao:
    """Contexto ambiental de condução que influencia as prioridades"""
    clima: CondicaoClimatica
    tipo_via: TipoVia
    trafego: DensidadeTrafego
    velocidade_atual: float  # km/h
    modo_autonomo: bool

    def get_fator_ajuste(self, tipo: TipoProcessoCAV) -> float:
        """
        Calcula fator de ajuste da prioridade baseado no contexto de condução
        Args:
            tipo: Tipo do processo CAV
        Returns:
            Fator multiplicativo de ajuste (>= 1.0)
        """
        fator = 1.0
        if tipo is None: # Se o tipo de processo não for especificado, retorna fator base
            return fator
        
        # Ajustes por condição climática
        if self.clima in [CondicaoClimatica.CHUVA, CondicaoClimatica.NEVE]:
            if tipo == TipoProcessoCAV.SEGURANCA_CRITICA:
                fator *= 1.3
        elif self.clima == CondicaoClimatica.NEBLINA:
            if tipo == TipoProcessoCAV.SEGURANCA_CRITICA:
                fator *= 1.2
        
        # Ajustes por densidade de tráfego
        if self.trafego in [DensidadeTrafego.ALTA, DensidadeTrafego.CONGESTIONAMENTO]:
            if tipo == TipoProcessoCAV.SEGURANCA_CRITICA:
                fator *= 1.2
            elif tipo == TipoProcessoCAV.NAVEGACAO:
                fator *= 1.1
        
        # Ajustes por velocidade
        if self.velocidade_atual > 80.0:
            if tipo == TipoProcessoCAV.SEGURANCA_CRITICA:
                fator *= 1.4
            elif tipo == TipoProcessoCAV.TEMPO_REAL:
                fator *= 1.2
        
        # Ajuste por modo autônomo
        if self.modo_autonomo:
            if tipo == TipoProcessoCAV.SEGURANCA_CRITICA:
                fator *= 1.5
        
        return fator

# ==============================================================================
# CLASSE TAREFCAV COM ATRIBUTOS OPCIONAIS E AJUSTE NO MÉTODO EXECUTAR
# ==============================================================================
class TarefaCAV:
    """Representa uma tarefa/processo em um sistema CAV"""
    
    def __init__(self, nome: str, duracao: int, prioridade: int = 1, deadline: Optional[int] = None,
                 tipo_processo: Optional[TipoProcessoCAV] = None):
        self.nome = nome
        self.duracao = duracao
        self.prioridade = prioridade
        self.deadline = deadline
        self.tipo_processo: Optional[TipoProcessoCAV] = tipo_processo
        
        # Atributos de controle de execução
        self.tempo_restante = duracao
        self.tempo_inicio: Optional[float] = None
        self.tempo_final: Optional[float] = None
        self.tempo_espera: float = 0.0
        self.tempo_resposta: Optional[float] = None
        self.iniciado = False
        self.timestamp_chegada = 0.0

        # MUDANÇA: Adicionado contador de preempções totais
        self.total_preempcoes: int = 0
        self.preempcoes_consecutivas: int = 0 # Mantemos este também

    def __str__(self):
        return f"Tarefa {self.nome} (Prioridade {self.prioridade}): {self.duracao} segundos"

    def executar(self, quantum: int, tempo_atual_simulado: float) -> float:
        if not self.iniciado:
            self.tempo_inicio = tempo_atual_simulado
            self.tempo_resposta = self.tempo_inicio - self.timestamp_chegada
            self.iniciado = True

        tempo_execucao = min(quantum, self.tempo_restante)
        self.tempo_restante -= tempo_execucao
        
        if self.tempo_restante <= 0:
            self.tempo_final = tempo_atual_simulado + tempo_execucao
            self.tempo_espera = self.tempo_final - self.timestamp_chegada - self.duracao
        
        return tempo_execucao
    
    def __repr__(self):
        type_str = self.tipo_processo.name if self.tipo_processo else "N/A"
        deadline_str = str(self.deadline) if self.deadline is not None else "N/A"
        return f"TarefaCAV(nome={self.nome}, tipo={type_str}, dur={self.duracao}, prio={self.prioridade}, deadline={deadline_str})"

# ==============================================================================
# WRAPPER PARA PRIORIDADE DINÂMICA (MANTIDO DO HAPS_AI.PY)
# ==============================================================================
class ProcessoComPrioridade:
    """Wrapper para tarefa com prioridade dinâmica para uso com heapq"""

    def __init__(self, tarefa: TarefaCAV, prioridade_dinamica: float):
        self.tarefa = tarefa
        self.prioridade_dinamica = prioridade_dinamica

    def __lt__(self, other):
        """
        Comparação para heapq: prioridade maior primeiro, em caso de empate, deadline menor primeiro (EDF)
        heapq é min-heap, então invertemos a prioridade para simular max-heap.
        Se deadlines forem None, usa apenas a prioridade dinâmica.
        """
        if abs(self.prioridade_dinamica - other.prioridade_dinamica) < 0.001:
            # Em caso de empate de prioridade, usa EDF (deadline menor primeiro) se houver deadlines
            if self.tarefa.deadline is not None and other.tarefa.deadline is not None:
                return self.tarefa.deadline < other.tarefa.deadline
            return False # Sem critério de desempate claro, considera igual ou não menor
        return self.prioridade_dinamica > other.prioridade_dinamica

    def __repr__(self):
        return f"ProcessoComPrioridade({self.tarefa.nome}, prio={self.prioridade_dinamica:.2f})"

# ==============================================================================
# CLASSE BASE ESCALONADOR E CÁLCULO DE MÉTRICAS AJUSTADO
# ==============================================================================
class EscalonadorCAV(ABC):
    """Classe base para escalonadores CAV"""
    
    def __init__(self):
        self.tarefas: List[TarefaCAV] = []
        self.sobrecarga_total: float = 0.0
        self.trocas_contexto: int = 0
        self.tempo_simulacao_final: float = 0.0 # Adicionado para registrar o tempo total simulado

    def adicionar_tarefa(self, tarefa: TarefaCAV):
        """Adiciona uma tarefa ao escalonador"""
        self.tarefas.append(tarefa)

    @abstractmethod
    def escalonar(self):
        """Método abstrato para implementar a lógica de escalonamento"""
        pass

    def registrar_sobrecarga(self, tempo_sobrecarga: float):
        """Registra sobrecarga de troca de contexto"""
        self.sobrecarga_total += tempo_sobrecarga
        self.trocas_contexto += 1

    def exibir_sobrecarga(self):
        """Exibe estatísticas de sobrecarga"""
        print(f"\n=== SOBRECARGA DO SISTEMA ===")
        print(f"Sobrecarga total: {self.sobrecarga_total:.2f}s")
        print(f"Trocas de contexto: {self.trocas_contexto}")
        if self.trocas_contexto > 0:
            print(f"Sobrecarga média por troca: {self.sobrecarga_total/self.trocas_contexto:.3f}s")

    def calcular_metricas(self):
        """Calcula e exibe métricas de desempenho"""
        if not self.tarefas:
            print("\n=== MÉTRICAS DE DESEMPENHO ===\nNenhuma tarefa para calcular métricas.")
            return
            
        qtd_tarefas_completadas = [t for t in self.tarefas if t.tempo_final is not None]
        
        # Deadlines perdidos só são contados se a tarefa tiver um deadline definido
        qtd_deadlines_perdidos = sum(1 for t in qtd_tarefas_completadas
                                   if t.deadline is not None and (t.tempo_final - t.timestamp_chegada) > t.deadline)
        
        if qtd_tarefas_completadas:
            tempo_medio_espera = sum(t.tempo_espera for t in qtd_tarefas_completadas) / len(qtd_tarefas_completadas)
            tempo_medio_resposta = sum(t.tempo_resposta for t in qtd_tarefas_completadas
                                     if t.tempo_resposta is not None) / len(qtd_tarefas_completadas)
            tempo_medio_turnaround = sum(t.tempo_final - t.timestamp_chegada for t in qtd_tarefas_completadas) / len(qtd_tarefas_completadas)
        else:
            tempo_medio_espera = 0
            tempo_medio_resposta = 0
            tempo_medio_turnaround = 0

        print(f"\n=== MÉTRICAS DE DESEMPENHO ===")
        print(f"Tarefas totais: {len(self.tarefas)}")
        print(f"Tarefas completadas: {len(qtd_tarefas_completadas)}")
        print(f"Tempo médio de espera: {tempo_medio_espera:.2f}s")
        print(f"Tempo médio de resposta: {tempo_medio_resposta:.2f}s")
        print(f"Tempo médio de turnaround: {tempo_medio_turnaround:.2f}s")
        print(f"Deadlines perdidos: {qtd_deadlines_perdidos}")
        taxa_sucesso = ((len(qtd_tarefas_completadas) - qtd_deadlines_perdidos) / len(self.tarefas) * 100) if self.tarefas else 0
        print(f"Taxa de sucesso: {taxa_sucesso:.1f}%")

# ==============================================================================
# ESCALONADORES SIMPLES (AJUSTADOS PARA USAR tempo_atual_simulado no tarefa.executar)
# ==============================================================================
# Substitua sua classe EscalonadorHAPS inteira por esta:

class EscalonadorHAPS(EscalonadorCAV):
    """
    Escalonador Híbrido e Adaptativo com Machine Learning (Scikit-learn).
    """
    def __init__(self, contexto: ContextoConducao):
        super().__init__()
        self.contexto = contexto
        # As filas agora usam deque, pois a prioridade é calculada no momento da seleção.
        self.filas_por_tipo = {
            TipoProcessoCAV.SEGURANCA_CRITICA: deque(),
            TipoProcessoCAV.TEMPO_REAL: deque(),
            TipoProcessoCAV.NAVEGACAO: deque(),
            TipoProcessoCAV.CONFORTO: deque(),
            TipoProcessoCAV.DIAGNOSTICO: deque()
        }
        self.quantum_base_por_tipo = {
            TipoProcessoCAV.SEGURANCA_CRITICA: 4,
            TipoProcessoCAV.TEMPO_REAL: 6,
            TipoProcessoCAV.NAVEGACAO: 8,
            TipoProcessoCAV.CONFORTO: 10,
            TipoProcessoCAV.DIAGNOSTICO: 12
        }
        # Carrega o modelo de ML treinado
        try:
            self.model = joblib.load('haps_model.joblib')
            print("✅ Modelo de Machine Learning (Random Forest) carregado com sucesso.")
        except FileNotFoundError:
            self.model = None
            print("⚠️ Modelo de ML 'haps_model.joblib' não encontrado. Usando lógica de fallback (prioridade base).")

    def calcular_prioridade_ml(self, tarefa: TarefaCAV, tempo_atual_simulado: float) -> float:
        """Usa o modelo de ML para calcular uma pontuação de risco para a tarefa."""
        if self.model is None or tarefa.deadline is None:
            # Lógica de fallback se o modelo não existir ou a tarefa não tiver deadline
            return tarefa.prioridade

        # Prepara o vetor de features em tempo real, no mesmo formato do treino
        turnaround_atual = tempo_atual_simulado - tarefa.timestamp_chegada
        features = np.array([[
            tarefa.duracao, tarefa.prioridade, tarefa.tipo_processo.value,
            tarefa.total_preempcoes, turnaround_atual, tarefa.deadline
        ]])
        
        # Usa o modelo para prever a PROBABILIDADE de falha (classe 1)
        prob_falha = self.model.predict_proba(features)[0][1]
        
        # Bônus de urgência para reagir a deadlines iminentes
        tempo_restante_deadline = tarefa.deadline - tempo_atual_simulado
        bonus_urgencia = 0
        if tempo_restante_deadline < (tarefa.duracao * 0.5): # Reação agressiva se o tempo for curto
            bonus_urgencia = 1000 / max(0.1, tempo_restante_deadline)

        # A prioridade final é uma combinação do risco previsto e da urgência imediata
        return (prob_falha * 5000) + bonus_urgencia

    def adicionar_tarefa_fila(self, tarefa: TarefaCAV):
        """Apenas adiciona a tarefa à sua fila correspondente."""
        if tarefa.tipo_processo and tarefa.tipo_processo in self.filas_por_tipo:
            self.filas_por_tipo[tarefa.tipo_processo].append(tarefa)

    def selecionar_proxima_tarefa(self, tempo_atual_simulado: float) -> Optional[TarefaCAV]:
        """O cérebro do HAPS-AI: avalia os candidatos e escolhe o mais crítico usando ML."""
        candidatos = []
        for fila in self.filas_por_tipo.values():
            if fila:
                candidatos.append(fila[0]) # Pega o primeiro de cada fila sem remover

        if not candidatos:
            return None

        # Calcula a pontuação de risco para cada candidato
        scores = {c.nome: self.calcular_prioridade_ml(c, tempo_atual_simulado) for c in candidatos}
        
        # Encontra o nome da tarefa com a maior pontuação
        melhor_candidato_nome = max(scores, key=scores.get)
        
        # Encontra e remove a tarefa escolhida da sua fila original
        for fila in self.filas_por_tipo.values():
            if fila and fila[0].nome == melhor_candidato_nome:
                melhor_tarefa = fila.popleft()
                print(f"[HAPS-AI] Decisão: {melhor_tarefa.nome} | Prio Calculada: {scores[melhor_tarefa.nome]:.2f}")
                return melhor_tarefa
        return None

    def escalonar(self):
        print("\n=== INICIANDO ESCALONAMENTO HAPS-AI (com Scikit-learn) ===")
        tempo_atual_simulado = 0.0
        
        for tarefa in self.tarefas:
            tarefa.timestamp_chegada = tempo_atual_simulado
            self.adicionar_tarefa_fila(tarefa)
        
        while any(self.filas_por_tipo.values()):
            tarefa_atual = self.selecionar_proxima_tarefa(tempo_atual_simulado)
            if tarefa_atual is None: break

            quantum = self.quantum_base_por_tipo.get(tarefa_atual.tipo_processo, 8)
            tempo_executado = tarefa_atual.executar(quantum, tempo_atual_simulado)
            tempo_atual_simulado += tempo_executado

            tempo_sobrecarga = 0.01
            self.registrar_sobrecarga(tempo_sobrecarga)
            tempo_atual_simulado += tempo_sobrecarga

            if tarefa_atual.tempo_restante > 0:
                tarefa_atual.total_preempcoes += 1
                self.adicionar_tarefa_fila(tarefa_atual)
        
        self.tempo_simulacao_final = tempo_atual_simulado
        self.salvar_historico_para_treino()
        self.calcular_metricas()
        self.exibir_sobrecarga()
        
    def salvar_historico_para_treino(self, arquivo_csv='haps_training_data.csv'):
        novos_dados = []
        for tarefa in self.tarefas:
            if tarefa.tempo_final is not None and tarefa.deadline is not None:
                turnaround = tarefa.tempo_final - tarefa.timestamp_chegada
                deadline_perdido = 1 if turnaround > tarefa.deadline else 0
                dados = {
                    'duracao_total': tarefa.duracao, 'prioridade_base': tarefa.prioridade,
                    'tipo_processo': tarefa.tipo_processo.value, 'total_preempcoes': tarefa.total_preempcoes,
                    'turnaround_final': turnaround, 'deadline_final': tarefa.deadline,
                    'deadline_perdido': deadline_perdido
                }
                novos_dados.append(dados)
        
        if not novos_dados: return
        df = pd.DataFrame(novos_dados)
        try:
            if not os.path.exists(arquivo_csv):
                df.to_csv(arquivo_csv, index=False)
            else:
                df.to_csv(arquivo_csv, mode='a', header=False, index=False)
        except IOError as e:
            print(f"Erro ao salvar dados: {e}")

class EscalonadorFIFO(EscalonadorCAV):
    def escalonar(self):
        """Escalonamento FIFO para veículos autônomos"""
        print("\n=== ESCALONAMENTO FIFO ===")
        tempo_atual_simulado = 0.0 # Inicia o tempo simulado
        for tarefa in self.tarefas:
            # Reseta o estado da tarefa para esta simulação (feito também no deepcopy)
            tarefa.tempo_restante = tarefa.duracao
            tarefa.iniciado = False
            tarefa.tempo_inicio = None
            tarefa.tempo_final = None
            tarefa.tempo_espera = 0.0
            tarefa.tempo_resposta = None
            tarefa.timestamp_chegada = tempo_atual_simulado # A tarefa chega no tempo atual simulado

            print(f"[FIFO] Iniciando {tarefa.nome} no tempo simulado {tempo_atual_simulado:.2f}s")
            
            # Executa a tarefa completamente
            tempo_executado_no_quantum = tarefa.executar(tarefa.duracao, tempo_atual_simulado)
            time.sleep(0.01) # Simulação acelerada (tempo real para rodar o código)

            tempo_atual_simulado += tempo_executado_no_quantum # Avança o tempo simulado
            tarefa.tempo_final = tempo_atual_simulado # Define o tempo final baseado no tempo simulado
            tarefa.tempo_espera = tarefa.tempo_final - tarefa.timestamp_chegada - tarefa.duracao # Recalcula espera

            print(f"[FIFO] Tarefa {tarefa.nome} finalizada no tempo simulado {tarefa.tempo_final:.2f}s.")
            self.registrar_sobrecarga(0.05) # Sobrecarga simulada

        self.tempo_simulacao_final = tempo_atual_simulado # Registra o tempo total simulado
        self.calcular_metricas()
        self.exibir_sobrecarga()

class EscalonadorSJF(EscalonadorCAV):
    def escalonar(self):
        """Escalonamento SJF para veículos autônomos."""
        print("\n=== ESCALONAMENTO SJF ===")
        tempo_atual_simulado = 0.0  # Inicia o tempo simulado
        #Ordena a lista de tarefas com base na sua duração.
        tarefas_ordenadas = sorted(self.tarefas, key=lambda tarefa: tarefa.duracao)
        for tarefa in tarefas_ordenadas:
            # Reseta o estado da tarefa para esta simulação
            tarefa.tempo_restante = tarefa.duracao
            tarefa.iniciado = False
            tarefa.tempo_inicio = None
            tarefa.tempo_final = None
            tarefa.tempo_espera = 0.0
            tarefa.tempo_resposta = None
            tarefa.timestamp_chegada = 0.0

            print(f"[SJF] Iniciando '{tarefa.nome}' no tempo simulado {tempo_atual_simulado:.2f}s")

            #Executa a tarefa completamente
            tempo_executado = tarefa.executar(tarefa.duracao, tempo_atual_simulado)
            time.sleep(0.01)  # Simulação acelerada

            tempo_atual_simulado += tempo_executado  # Avança o tempo simulado
            tarefa.tempo_final = tempo_atual_simulado  # Define o tempo final
            tarefa.tempo_espera = tarefa.tempo_final - tarefa.timestamp_chegada - tarefa.duracao

            print(f"[SJF] Tarefa {tarefa.nome} finalizada no tempo simulado {tarefa.tempo_final:.2f}s.")
            self.registrar_sobrecarga(0.05)  # Sobrecarga simulada

        self.tempo_simulacao_final = tempo_atual_simulado  # Registra o tempo total simulado
        self.calcular_metricas()
        self.exibir_sobrecarga() 

class EscalonadorRoundRobin(EscalonadorCAV):
    def __init__(self):
        super().__init__()
        self.quantum = 1

    def escalonar(self):
        print(f"\n=== ESCALONAMENTO ROUND ROBIN (Q={self.quantum}) ===")
        fila = deque(self.tarefas)
        tempo_atual_simulado = 0.0 # Inicia o tempo simulado

        # Resetar tarefas para esta simulação específica
        for t in self.tarefas:
            t.tempo_restante = t.duracao
            t.iniciado = False
            t.tempo_inicio = None
            t.tempo_final = None
            t.tempo_espera = 0.0
            t.tempo_resposta = None
            t.timestamp_chegada = 0.0 # Todas chegam no tempo 0 para RR inicial

        while fila:
            tarefa = fila.popleft()
            
            # Se a tarefa ainda não foi iniciada, define o tempo de chegada simulado
            if not tarefa.iniciado:
                tarefa.timestamp_chegada = tempo_atual_simulado

            if tarefa.tempo_restante > 0:
                print(f"[RR] Executando {tarefa.nome} por {self.quantum} unidades (restante: {tarefa.tempo_restante}) no tempo simulado {tempo_atual_simulado:.2f}s")
                
                tempo_executado_no_quantum = tarefa.executar(self.quantum, tempo_atual_simulado) # Passa tempo simulado
                time.sleep(0.01) # Simulação acelerada

                tempo_atual_simulado += tempo_executado_no_quantum # Avança o tempo simulado
                
                self.registrar_sobrecarga(0.05)

                if tarefa.tempo_restante > 0:
                    fila.append(tarefa) # Reinsere na fila se não terminou
                    print(f"[RR] {tarefa.nome} preemptada, recolocando na fila (tempo simulado {tempo_atual_simulado:.2f}s).")
                else:
                    tarefa.tempo_final = tempo_atual_simulado
                    tarefa.tempo_espera = tarefa.tempo_final - tarefa.timestamp_chegada - tarefa.duracao
                    print(f"[RR] {tarefa.nome} COMPLETADA no tempo simulado {tarefa.tempo_final:.2f}s.")
            else:
                # Tarefa já concluída, não deveria estar na fila, mas por segurança
                if tarefa.tempo_final is None: # Se chegou aqui e já devia estar finalizada
                    tarefa.tempo_final = tempo_atual_simulado
                    tarefa.tempo_espera = tarefa.tempo_final - tarefa.timestamp_chegada - tarefa.duracao

        self.tempo_simulacao_final = tempo_atual_simulado
        self.calcular_metricas()
        self.exibir_sobrecarga()

class EscalonadorPrioridade(EscalonadorCAV):
    def escalonar(self):
        """Escalonamento por Prioridade (menor número = maior prioridade)"""
        print("\n=== ESCALONAMENTO POR PRIORIDADE ===")
        # Ordena as tarefas pela prioridade (menor número = maior prioridade)
        # Pode usar heapq aqui se preferir um heap real para grandes volumes
        self.tarefas.sort(key=lambda tarefa: tarefa.prioridade) 
        
        tempo_atual_simulado = 0.0 # Inicia o tempo simulado

        # Resetar tarefas para esta simulação específica
        for t in self.tarefas:
            t.tempo_restante = t.duracao
            t.iniciado = False
            t.tempo_inicio = None
            t.tempo_final = None
            t.tempo_espera = 0.0
            t.tempo_resposta = None
            t.timestamp_chegada = 0.0 # Todas chegam no tempo 0

        for tarefa in self.tarefas:
            tarefa.timestamp_chegada = tempo_atual_simulado # A tarefa chega no tempo atual simulado

            print(f"[PRIO] Iniciando {tarefa.nome} (Prio: {tarefa.prioridade}) no tempo simulado {tempo_atual_simulado:.2f}s.")
            
            # Executa a tarefa completamente
            tempo_executado_no_quantum = tarefa.executar(tarefa.duracao, tempo_atual_simulado)
            time.sleep(0.01) # Simulação acelerada

            tempo_atual_simulado += tempo_executado_no_quantum # Avança o tempo simulado
            tarefa.tempo_final = tempo_atual_simulado # Define o tempo final baseado no tempo simulado
            tarefa.tempo_espera = tarefa.tempo_final - tarefa.timestamp_chegada - tarefa.duracao # Recalcula espera

            print(f"[PRIO] Tarefa {tarefa.nome} finalizada no tempo simulado {tarefa.tempo_final:.2f}s.")
            self.registrar_sobrecarga(0.08) # Sobrecarga simulada

        self.tempo_simulacao_final = tempo_atual_simulado
        self.calcular_metricas()
        self.exibir_sobrecarga()


# ==============================================================================
# FUNÇÕES DE CRIAÇÃO DE TAREFAS
# ==============================================================================

def criar_tarefas_simples() -> List[TarefaCAV]:
    """
    Cria um conjunto de tarefas simples sem TipoProcesso ou Deadline
    para compatibilidade com algoritmos que não usam esses atributos.
    """
    tarefas = [
        TarefaCAV("Tarefa_A", random.randint(5, 10), prioridade=1),
        TarefaCAV("Tarefa_B", random.randint(3, 7), prioridade=2),
        TarefaCAV("Tarefa_C", random.randint(8, 12), prioridade=3),
        TarefaCAV("Tarefa_D", random.randint(4, 9), prioridade=1),
        TarefaCAV("Tarefa_E", random.randint(6, 11), prioridade=2),
    ]
    return tarefas

def criar_tarefas_cav_completas() -> List[TarefaCAV]:
    """Cria conjunto diversificado de tarefas CAV com todos os atributos."""
    tarefas = [
        # Tarefas críticas de segurança
        TarefaCAV("Detecção_Obstáculo", random.randint(5, 10), 95, random.randint(8, 15), TipoProcessoCAV.SEGURANCA_CRITICA),
        TarefaCAV("Sistema_Frenagem", random.randint(4, 7), 98, random.randint(6, 12), TipoProcessoCAV.SEGURANCA_CRITICA),
        TarefaCAV("Controle_Estabilidade", random.randint(6, 9), 90, random.randint(10, 15), TipoProcessoCAV.SEGURANCA_CRITICA),
        
        # Tarefas de tempo real
        TarefaCAV("Controle_Motor", random.randint(8, 15), 85, random.randint(15, 25), TipoProcessoCAV.TEMPO_REAL),
        TarefaCAV("Sensor_LiDAR", random.randint(6, 12), 80, random.randint(12, 20), TipoProcessoCAV.TEMPO_REAL),
        TarefaCAV("Processamento_Camera", random.randint(10, 18), 75, random.randint(20, 30), TipoProcessoCAV.TEMPO_REAL),
        
        # Tarefas de navegação
        TarefaCAV("GPS_Localização", random.randint(7, 14), 70, random.randint(20, 35), TipoProcessoCAV.NAVEGACAO),
        TarefaCAV("Planejamento_Rota", random.randint(12, 20), 65, random.randint(30, 45), TipoProcessoCAV.NAVEGACAO),
        TarefaCAV("Mapeamento_HD", random.randint(15, 25), 60, random.randint(40, 60), TipoProcessoCAV.NAVEGACAO),
        
        # Tarefas de conforto
        TarefaCAV("Controle_Climatização", random.randint(5, 10), 50, random.randint(40, 60), TipoProcessoCAV.CONFORTO),
        TarefaCAV("Sistema_Audio", random.randint(4, 8), 45, random.randint(50, 80), TipoProcessoCAV.CONFORTO),
        TarefaCAV("Interface_Usuario", random.randint(8, 15), 40, random.randint(45, 70), TipoProcessoCAV.CONFORTO),
        
        # Tarefas de diagnóstico
        TarefaCAV("Monitoramento_Sistema", random.randint(10, 18), 30, random.randint(80, 120), TipoProcessoCAV.DIAGNOSTICO),
        TarefaCAV("Log_Eventos", random.randint(4, 8), 25, random.randint(90, 150), TipoProcessoCAV.DIAGNOSTICO),
        TarefaCAV("Telemetria", random.randint(7, 14), 20, random.randint(100, 180), TipoProcessoCAV.DIAGNOSTICO),
    ]
    return tarefas

# ==============================================================================
# FUNÇÃO DE ANÁLISE DE DESEMPENHO AJUSTADA PARA O NOVO FORMATO DE SAÍDA
# ==============================================================================
def analisar_desempenho(escalonadores_com_tempos: List[Tuple[str, EscalonadorCAV, float]]):
    """
    Analisa e exibe o desempenho dos escalonadores com foco em métricas de simulação.
    """
    print("\n" + "="*140 + "\n")
    print("ANÁLISE COMPARATIVA DE DESEMPENHO DOS ESCALONADORES".center(140))
    print("\n" + "="*140)

    resultados = []
    for nome, escalonador, tempo_exec_real in escalonadores_com_tempos:
        tarefas_completadas = [t for t in escalonador.tarefas if t.tempo_final is not None]
        
        if not tarefas_completadas:
            resultados.append({
                'Algoritmo': nome,
                'Tempo Simulado (s)': escalonador.tempo_simulacao_final,
                'Turnaround Médio (s)': 0,
                'Deadlines Perdidos (%)': 0,
                'Trocas de Contexto': escalonador.trocas_contexto,
                'Sobrecarga Total (s)': escalonador.sobrecarga_total,
                'Tempo Real (s)': tempo_exec_real
            })
            continue

        turnarounds = [t.tempo_final - t.timestamp_chegada for t in tarefas_completadas]
        turnaround_medio = sum(turnarounds) / len(turnarounds)

        tarefas_com_deadline = [t for t in tarefas_completadas if t.deadline is not None]
        deadlines_perdidos = sum(1 for t in tarefas_com_deadline if (t.tempo_final - t.timestamp_chegada) > t.deadline)
        
        percentual_deadlines_perdidos = (deadlines_perdidos / len(tarefas_com_deadline) * 100) if tarefas_com_deadline else 0

        resultados.append({
            'Algoritmo': nome,
            'Tempo Simulado (s)': escalonador.tempo_simulacao_final,
            'Turnaround Médio (s)': turnaround_medio,
            'Deadlines Perdidos (%)': percentual_deadlines_perdidos,
            'Trocas de Contexto': escalonador.trocas_contexto,
            'Sobrecarga Total (s)': escalonador.sobrecarga_total,
            'Tempo Real (s)': tempo_exec_real
        })

    # Imprimir tabela de resultados
    headers = resultados[0].keys()
    col_widths = {key: max(len(str(key)), max((len(f"{x[key]:.2f}") if isinstance(x[key], float) else len(str(x[key]))) for x in resultados)) for key in headers}
    header_line = " | ".join(f"{h:<{col_widths[h]}}" for h in headers)
    print(header_line)
    print("-" * len(header_line))
    for res in resultados:
        row_line = " | ".join(f"{ (f'{v:.2f}' if isinstance(v, float) else v) :<{col_widths[k]}}" for k, v in res.items())
        print(row_line)

    # Análise e destaques
    print("\n" + "-"*50)
    print("DESTAQUES DA COMPARAÇÃO".center(50))
    print("-" * 50)
    
    melhor_tempo_simulado = min(resultados, key=lambda x: x['Tempo Simulado (s)'])
    melhor_turnaround = min(resultados, key=lambda x: x['Turnaround Médio (s)'])
    melhor_deadlines = min(resultados, key=lambda x: x['Deadlines Perdidos (%)'])
    menor_sobrecarga = min(resultados, key=lambda x: x['Sobrecarga Total (s)'])

    print(f"🏆 Menor Tempo de Simulação: {melhor_tempo_simulado['Algoritmo']} ({melhor_tempo_simulado['Tempo Simulado (s)']:.2f}s)")
    print(f"⏱️ Menor Turnaround Médio:   {melhor_turnaround['Algoritmo']} ({melhor_turnaround['Turnaround Médio (s)']:.2f}s)")
    print(f"🎯 Menor % de Deadlines Perdidos: {melhor_deadlines['Algoritmo']} ({melhor_deadlines['Deadlines Perdidos (%)']:.2f}%)")
    print(f"⚙️ Menor Sobrecarga Total:   {menor_sobrecarga['Algoritmo']} ({menor_sobrecarga['Sobrecarga Total (s)']:.2f}s)")
    
    print("\nNota: 'Tempo Real (s)' é o tempo de execução do script Python, enquanto 'Tempo Simulado (s)' é a métrica de eficiência do escalonador.")


# ==============================================================================
# FUNÇÃO PRINCIPAL DE COMPARAÇÃO
# ==============================================================================

def executar_comparacao_algoritmos(contexto_teste: ContextoConducao):
    """
    Executa comparação entre diferentes algoritmos de escalonamento.
    Permite usar tarefas completas ou simples.
    """
    print("=" * 80)
    print("COMPARAÇÃO DE ALGORITMOS DE ESCALONAMENTO CAV")
    print(f"Contexto de Teste: Clima={contexto_teste.clima.name}, Via={contexto_teste.tipo_via.name}, Tráfego={contexto_teste.trafego.name}")
    print("=" * 80)
    
    # Usa tarefas completas para testar todos os recursos dos escalonadores
    tarefas_base = criar_tarefas_cav_completas()

    escalonadores = [
        ("HAPS-AI", EscalonadorHAPS(contexto_teste)),
        ("FIFO", EscalonadorFIFO()),
        ("SJF", EscalonadorSJF()),
        ("Round Robin", EscalonadorRoundRobin()),
        ("Prioridade Estática", EscalonadorPrioridade()),
        
    ]
    
    escalonadores_com_tempos = []

    for nome, escalonador_instancia in escalonadores:
        print(f"\n{'='*20} TESTANDO {nome.upper()} {'='*20}")
        
        tarefas_copia = copy.deepcopy(tarefas_base)
        
        # Reseta o estado das tarefas e as adiciona ao escalonador
        escalonador_instancia.tarefas.clear()
        for tarefa in tarefas_copia:
            escalonador_instancia.adicionar_tarefa(tarefa)

        tempo_inicio_real_algoritmo = time.time()
        escalonador_instancia.escalonar()
        tempo_fim_real_algoritmo = time.time()
        tempo_exec_real_algoritmo = tempo_fim_real_algoritmo - tempo_inicio_real_algoritmo
        
        escalonadores_com_tempos.append((nome, escalonador_instancia, tempo_exec_real_algoritmo))
    
    analisar_desempenho(escalonadores_com_tempos)

if __name__ == "__main__":
    # Contexto de condução para o teste
    contexto_exemplo = ContextoConducao(
        clima=CondicaoClimatica.CHUVA,
        tipo_via=TipoVia.RODOVIA,
        trafego=DensidadeTrafego.ALTA,
        velocidade_atual=95.0,
        modo_autonomo=True
    )
    
    # Executa a comparação entre os algoritmos com o contexto criado
    executar_comparacao_algoritmos(contexto_exemplo)

if __name__ == "__main__":
    contexto_exemplo = ContextoConducao(
        clima=CondicaoClimatica.CHUVA,
        tipo_via=TipoVia.RODOVIA,
        trafego=DensidadeTrafego.ALTA,
        velocidade_atual=95.0,
        modo_autonomo=True
    )
    
    # Executa a comparação entre os algoritmos
    executar_comparacao_algoritmos(contexto_exemplo)