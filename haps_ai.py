import joblib
import pandas as pd
import numpy as np
import random
import time
import copy
import os
import json
import matplotlib.pyplot as plt
from collections import deque
from abc import ABC, abstractmethod
from enum import Enum
from typing import List,  Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
nome_arquivo = "dados"
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
    Versão Final com Scikit-learn e Coleta de Snapshots para resolver o Paradoxo do Vidente.
    """
    def __init__(self, contexto: ContextoConducao):
        super().__init__()
        self.contexto = contexto

        #faz uma fila/pilha de cada tipo e joga num dicionário
        self.filas_por_tipo = { tipo: deque() for tipo in TipoProcessoCAV } 
        self.quantum_base = 4  # Quantum simplificado
        
        # Snapshots guardados
        self.log_snapshots = []
        
        try:
            self.model = joblib.load('haps_model.joblib')
            print("✅ Modelo de ML carregado.")
        except FileNotFoundError:
            self.model = None
            print("⚠️ Modelo de ML não encontrado. Usando prioridade base como fallback.")

    def _preparar_features(self, tarefa: TarefaCAV, tempo_atual: float, ciclo: int) -> np.ndarray:
        """Cria o vetor de features para uma tarefa em um dado momento."""
        tempo_decorrido = tempo_atual - tarefa.timestamp_chegada
        tempo_restante_deadline = (tarefa.deadline - tempo_atual) if tarefa.deadline is not None else 1000 # Um valor alto para "sem deadline"
        
        return np.array([[
            tarefa.tempo_restante,
            tempo_decorrido,
            tempo_restante_deadline,
            tarefa.total_preempcoes,
            tarefa.prioridade,
            tarefa.tipo_processo.value,
            ciclo
        ]])

    def calcular_prioridade_ml(self, tarefa: TarefaCAV, tempo_atual: float, ciclo: int) -> float:
        """Usa o modelo para prever o risco de falha e define a prioridade."""
        if self.model is None or tarefa.deadline is None:
            return tarefa.prioridade

        features = self._preparar_features(tarefa, tempo_atual, ciclo)
        prob_falha = self.model.predict_proba(features)[0][1] # Pega a probabilidade da classe '1' (falha)
        
        return prob_falha

    def adicionar_tarefa_fila(self, tarefa: TarefaCAV):
        if tarefa.tipo_processo in self.filas_por_tipo:
            self.filas_por_tipo[tarefa.tipo_processo].append(tarefa)

    def selecionar_proxima_tarefa(self, tempo_atual: float, ciclo: int) -> Optional[TarefaCAV]:
        candidatos = [fila[0] for fila in self.filas_por_tipo.values() if fila]
        if not candidatos:
            return None

        # Calcula a prioridade (risco) para cada candidato
        prioridades = [self.calcular_prioridade_ml(c, tempo_atual, ciclo) for c in candidatos]
        
        # Escolhe o candidato com a maior prioridade (maior risco de falha)
        melhor_candidato = candidatos[np.argmax(prioridades)]
        
        # Remove o candidato escolhido da sua fila
        self.filas_por_tipo[melhor_candidato.tipo_processo].popleft()
        
        return melhor_candidato

    def escalonar(self):
        self.log_snapshots = [] # Limpa o log para a nova simulação
        tempo_atual_simulado = 0.0
        ciclo = 1

        for tarefa in self.tarefas:
            tarefa.timestamp_chegada = tempo_atual_simulado
            self.adicionar_tarefa_fila(tarefa)
        
        while any(self.filas_por_tipo.values()):
            # Para cada tarefa na fila, tiramos um "snapshot" do seu estado atual
            for fila in self.filas_por_tipo.values():
                for tarefa_na_fila in fila:
                    features_atuais = self._preparar_features(tarefa_na_fila, tempo_atual_simulado, ciclo)[0]
                    snapshot = dict(zip(self.get_feature_names(), features_atuais))
                    snapshot['task_id'] = id(tarefa_na_fila) 
                    self.log_snapshots.append(snapshot)
            
            tarefa_atual = self.selecionar_proxima_tarefa(tempo_atual_simulado, ciclo)
            if tarefa_atual is None: 
                break

            quantum = self.quantum_base  # Usando o quantum simplificado
            tempo_executado = tarefa_atual.executar(quantum, tempo_atual_simulado)
            tempo_atual_simulado += tempo_executado


            tempo_sobrecarga = 0.01  # Custo fixo por ciclo
            self.registrar_sobrecarga(tempo_sobrecarga)
            tempo_atual_simulado += tempo_sobrecarga # Cálculo limpo e direto

            # Se a tarefa não terminou, ela é preemptada e volta para a fila
            if tarefa_atual.tempo_restante > 0:
                tarefa_atual.total_preempcoes += 1
                self.adicionar_tarefa_fila(tarefa_atual)
            
            ciclo += 1

        # Finaliza a simulação e salva os resultados
        self.tempo_simulacao_final = tempo_atual_simulado
        self._processar_e_salvar_snapshots() # Salva os dados para futuro treino
        self.calcular_metricas()
        self.exibir_sobrecarga() # Exibe a sobrecarga total que foi acumulada

    def _processar_e_salvar_snapshots(self, arquivo_csv='haps_training_data.csv'):
        """Pega todos os snapshots, adiciona o resultado final e salva em CSV."""
        if not self.log_snapshots: return
        
        # Mapeia o resultado final de cada tarefa
        resultado_final = {}
        for tarefa in self.tarefas:
            if tarefa.deadline is not None:
                turnaround = tarefa.tempo_final - tarefa.timestamp_chegada if tarefa.tempo_final is not None else float('inf')
                resultado_final[id(tarefa)] = 1 if turnaround > tarefa.deadline else 0
        
        # Adiciona a etiqueta 'deadline_perdido' a cada snapshot
        for snapshot in self.log_snapshots:
            task_id = snapshot.pop('task_id')
            snapshot['deadline_perdido'] = resultado_final.get(task_id, 0)
            
        df = pd.DataFrame(self.log_snapshots)
        
        try:
            if not os.path.exists(arquivo_csv):
                df.to_csv(arquivo_csv, index=False)
            else:
                df.to_csv(arquivo_csv, mode='a', header=False, index=False)
            print(f"📈 Snapshots de treinamento salvos em '{arquivo_csv}'.")
        except IOError as e:
            print(f"Erro ao salvar snapshots: {e}")
            
    def get_feature_names(self):
        """Helper para garantir consistência nos nomes das features."""
        return [
            'tempo_restante_execucao', 'tempo_decorrido', 'tempo_restante_deadline',
            'total_preempcoes', 'prioridade_base', 'tipo_processo', 'ciclo' 
        ]

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
def analisar_desempenho(escalonadores: List[Tuple[str, EscalonadorCAV]]):
    """Analisa e exibe o desempenho dos escalonadores"""
    print("\n" + "="*60)
    print("ANÁLISE DE DESEMPENHO DOS ESCALONADORES")
    print("="*60)

    
    resultados = []
    for item in escalonadores:

        nome = item[0]
        escalonador = item[1]
        tempo_exec = item[2]
        turnarounds = []
        deadlines_perdidos = 0

        for tarefa in escalonador.tarefas:

            if  tarefa.tempo_final is not None:
                turnaround =  tarefa.tempo_final - tarefa.timestamp_chegada
                turnarounds.append(turnaround)

                if turnaround > tarefa.deadline:
                    deadlines_perdidos += 1
        
        deadlines_perdidos = (deadlines_perdidos/len(turnarounds))*100

        turnaround_medio = sum(turnarounds) / len(turnarounds) if turnarounds else 0
        
        trocas_contexto = escalonador.trocas_contexto
        sobrecarga_total = escalonador.sobrecarga_total

        resultados.append({
            'nome': nome,
            'turnaround_medio': turnaround_medio,
            'tempo_executado': tempo_exec,
            'deadlines_perdidos': deadlines_perdidos,
            'trocas_contexto': trocas_contexto,
            'sobrecarga_total': sobrecarga_total
        })

    # Exibir resultados
    print(f"{'Algoritmo':<20} {'Tempo Total (s)':<18} {'Turnaround Médio (s)':<22} {'Deadlines Perdidos':<20} {'Trocas Contexto':<17} {'Sobrecarga Total (s)':<20}")
    print("-" * 65)
    for res in resultados:
        print(f"{res['nome']:<25} {res['tempo_executado']:<20.2f} {res['turnaround_medio']:<20.2f} {res['deadlines_perdidos']:<20.0f} {res['trocas_contexto']:<20} {res['sobrecarga_total']:<20.2f}")
    return resultados

def executar_comparacao_algoritmos():
    
    """Executa comparação entre diferentes algoritmos de escalonamento"""
    print("=" * 80)
    print("COMPARAÇÃO DE ALGORITMOS DE ESCALONAMENTO CAV")
    print("=" * 80)
    
    # Contexto adverso para teste
    contexto_teste = ContextoConducao(
        clima=CondicaoClimatica.CHUVA,
        tipo_via=TipoVia.RODOVIA,
        trafego=DensidadeTrafego.ALTA,
        velocidade_atual=95.0,
        modo_autonomo=True
    )
    
    # Cria tarefas base
    tarefas_base = criar_tarefas_cav_completas()
    
    # Lista de escalonadores para comparar
    escalonadores = [
        ("HAPS-AI", EscalonadorHAPS(contexto_teste)),
        ("FIFO", EscalonadorFIFO()),
        ("Round Robin", EscalonadorRoundRobin()),
        ("Prioridade Estática", EscalonadorPrioridade())
    ]
    
    resultados = {}
    tempos_execucao = {}

    for nome, escalonador in escalonadores:
        print(f"\n{'='*20} TESTANDO {nome} {'='*20}")
        
        # Copia tarefas para cada escalonador (reset de estado)
        
        tarefas_copia = copy.deepcopy(tarefas_base)
        
        for tarefa in tarefas_copia:
            escalonador.adicionar_tarefa(tarefa)

        inicio = time.time()
        escalonador.escalonar()
        fim = time.time()
        tempo_exec = fim - inicio
        tempos_execucao[nome] = tempo_exec
    
    escalonadores_com_tempos = []
    for nome, escalonador in escalonadores:
        escalonadores_com_tempos.append((
            nome,
            escalonador,
            tempos_execucao[nome]
        ))
    
    # Analisar desempenho
    return analisar_desempenho(escalonadores_com_tempos)

# ===== EXEMPLO DE USO PRINCIPAL =====
def rodar_varias_simulacoes(n=10):
    for i in range(n):
        resultado = executar_comparacao_algoritmos()
        with open(f"{nome_arquivo}{i}.json", "w", encoding="utf-8") as f:
            json.dump(resultado, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    # Cenário de teste adverso
    contexto_adverso = ContextoConducao(
        clima=CondicaoClimatica.CHUVA,
        tipo_via=TipoVia.RODOVIA,
        trafego=DensidadeTrafego.CONGESTIONAMENTO,
        velocidade_atual=85.0,
        modo_autonomo=True
    )
    
    print("HAPS-AI CAV SCHEDULER - DEMONSTRAÇÃO")
    print("Sistema de Escalonamento Híbrido e Adaptativo para Veículos Autônomos")
    print("=" * 80)
    
    # Cria conjunto de tarefas diversificadas
    tarefas_teste = criar_tarefas_cav_completas()
    
    # Instancia e executa o escalonador HAPS-AI
    #escalonador_haps = EscalonadorHAPS(contexto_adverso)
    #for tarefa in tarefas_teste:
    #    escalonador_haps.adicionar_tarefa_fila(tarefa)
    
    #escalonador_haps.escalonar()

    resultado = rodar_varias_simulacoes(10)