import random
import time
import copy
import os
import json
from collections import deque
from abc import ABC, abstractmethod
from enum import Enum
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime

# ==============================================================================
# CLASSES DE CONTEXTO E ENUMERAÇÕES (MANTIDAS DO HAPS_AI.PY)
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
        self.deadline = deadline  # Agora é opcional
        self.tipo_processo: Optional[TipoProcessoCAV] = tipo_processo # Agora é opcional
        
        # Atributos de controle de execução
        self.tempo_restante = duracao
        self.tempo_inicio: Optional[float] = None
        self.tempo_final: Optional[float] = None
        self.tempo_espera: float = 0.0
        self.tempo_resposta: Optional[float] = None
        self.iniciado = False
        self.timestamp_chegada = 0.0 # Sem time.time() aqui, será setado pelo escalonador

        # Histórico de execução para algoritmos preditivos (ainda que não usado pelos simples)
        self.historico_exec: List[float] = []

    def __str__(self):
        return f"Tarefa {self.nome} (Prioridade {self.prioridade}): {self.duracao} segundos"

    def executar(self, quantum: int, tempo_atual_simulado: float) -> float:
        """
        Executa a tarefa por um quantum de tempo.
        Args:
            quantum: Tempo de execução em unidades
            tempo_atual_simulado: O tempo atual na simulação do escalonador.
        Returns:
            Tempo real de execução consumido
        """
        if not self.iniciado:
            self.tempo_inicio = tempo_atual_simulado # Usa o tempo simulado
            self.tempo_resposta = self.tempo_inicio - self.timestamp_chegada
            self.iniciado = True
        
        tempo_execucao = min(quantum, self.tempo_restante)
        self.tempo_restante -= tempo_execucao
        
        if self.tempo_restante <= 0:
            self.tempo_final = tempo_atual_simulado + tempo_execucao # Tempo final no simulação
            self.tempo_espera = self.tempo_final - self.timestamp_chegada - self.duracao
        
        self.historico_exec.append(tempo_execucao)
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
class EscalonadorHAPS(EscalonadorCAV):
    def __init__(self, contexto: ContextoConducao, arquivo_historico: str = "haps_ai_historico.json" ):
        super().__init__()
        self.contexto = contexto
        self. arquivo_historico = arquivo_historico
        self.historico_desempenho: Dict[ str, Dict[ str, Any ] ] = {}

        #Parâmetros para o modelo preditivo
        self.alpha = 0.1 #fator de aprendizado
        self.modelo_pred: Dict[ str, Tuple [ float, float] ] = {} #hash_tarefa -> (intercept, slope)

        #Se os pesos já existirem previamente, os carrega no modelo preditivo  
        self._carregar_historico_ia()

        #Filas hierárquicas para cada tipo de processo
        self.fila_seguranca: List[ProcessoComPrioridade] = []
        self.fila_tempo_real: List[ProcessoComPrioridade] = []
        self.fila_navegacao: List[ProcessoComPrioridade] = []
        self.fila_diagnostico: List[ProcessoComPrioridade] = []
        self.fila_conforto: List[ProcessoComPrioridade] = []

        #Mapeamento dessas filas
        self.filas_por_tipo = {
            TipoProcessoCAV.SEGURANCA_CRITICA: self.fila_seguranca,
            TipoProcessoCAV.TEMPO_REAL: self.fila_tempo_real,
            TipoProcessoCAV.NAVEGACAO: self.fila_navegacao,
            TipoProcessoCAV.DIAGNOSTICO: self.fila_diagnostico,
            TipoProcessoCAV.CONFORTO: self.fila_conforto
        }

        #Quantum base por tipo de processo
        self.quantum_base_por_tipo = {
            TipoProcessoCAV.SEGURANCA_CRITICA: 2,
            TipoProcessoCAV.TEMPO_REAL: 3,
            TipoProcessoCAV.NAVEGACAO: 4,
            TipoProcessoCAV.DIAGNOSTICO: 5,
            TipoProcessoCAV.CONFORTO: 6
        }

        # Guarda as métricas para comparação com outros algoritmos
        self.metricas_execucao = {}


        self.contador_ciclos = 0
        self.intervalo_reavaliacao = 3 #Reavaliar prioridades a cada 3 ciclos

    def _carregar_historico_ia(self):
        try:
            if os.path.exists(self.arquivo_historico):
                with open(self.arquivo_historico, 'r', encoding='utf-8') as f:
                    dados_historico = json.load(f)
                    self.historico_desempenho = dados_historico.get('historico_aprendizado', {})

                    #Carregar o modelo, caso exista
                    modelo_pred_salvo = dados_historico.get('modelo_pred', {})
                    for hash_tarefa, params in modelo_pred_salvo.items():
                        #Garantir que os parâmetros sejam tuplas
                        if isinstance(params, list):
                            self.modelo_pred[hash_tarefa] = tuple(params)
                        else:
                            self.modelo_pred[hash_tarefa] = params

                    print(f"[IA] Histórico carregado: {len(self.historico_desempenho)} tarefas aprendidas")
                    print(f"[IA] Modelo preditivo carregado: {len(self.modelo_pred)} entradas")
                    self._exibir_estatisticas_ia()

            else:
                print(f"[IA] Arquivo de histórico não encontrado, iniciando novo aprendizado")
                self.historico_desempenho = {}
                self.modelo_pred = {}
        except(json.JSONcodeError, IOError) as e:
            print(f"[ERRO] galha ao carregar histórico IA: {e}")
            print("[IA] Iniciando com histórico vazio por segurança")
            time.sleep(1) #!VER DEPOIS
            self.modelo_pred = {}

    def _salvar_historico_ia(self):
        try:
            contexto_dict ={
                'clima': self.contexto.clima.value,
                'tipo_via': self.contexto.tipo.via.value,
                'trafego': self.contexto.trafego.value,
                'velocidade_atual': self.contexto.velocidade_atual,
                'modo_autonomo': self.contexto.modo_autonomo
            }
            
            dados_persistencia = {
                'versao_esquema': '1.0',
                'timestamp_salvamento': datetime.now().isonformat(),
                'contexto_ultimo_uso': contexto_dict,
                'historico_aprendizado': self.historico_desempenho,
                'estatisticas_sessao': {
                    'tarefas_processadas': len(self.tarefas),
                    'sobrecarga_total': self.sobrecarga_total,
                    'trocas_contexto': self.trocas_contexto
                },
                'modelo_pred': self.modelo_pred
            }

            with open(self.arquivo_historico, 'w', encoding='utf-8') as f:
                json.dump(dados_persistencia, f, indent=2, ensure_ascii=False)
            
            print(f"[IA] Histórico persistido: {len(self.historico_desempenho)} tarefas aprendidas")
            print(f"[IA] Modelo preditivo persistido: {len(self.modelo_pred)} entradas")

        except IOError as e:
            print(f"[ERRO] Falha ao salvar histórico IA: {e}")
            time.sleep(1) #! VER LINHA
    
    def _exibir_estatisticas_ia(self):
        if not self.historico_desempenho:
            return
        
        print(f"[IA] === ESTATÍSTICAS DE APRENDIZADO ===")

        tarefas_criticas = sum(1 for dados in self.historico_desempenho.values()
                               if dados.get('fator_aprendizado', 1.0) > 1.1)
        tarefas_problematicas = sum(1 for dados in self.historico_desempenho.values()
                               if dados.get('fator_aprendizado', 1.0) < 0.9)
        
        print(f"[IA] Tarefas com desempenho superior: {tarefas_criticas}")
        print(f"[IA] Tarefas que precisam de atenção: {tarefas_problematicas}")

    def _gerar_hash_tarefa(self, tarefa: TarefaCAV) -> str:
        """
        Gera o hash de tarefas para otimização de localização incluindo o contexto

        inclui nome, tipo e contexto básico para rastreamento
        """
        contexto_hash = f"{self.contexto.clima.value}_{self.contexto.trafego.value}_{self.contexto.modo_autonomo}"
        return f"{tarefa.nome}_{tarefa.tipo_processo.nome}_{contexto_hash}"
    
    def _atualizar_aprendizado_ia(self, tarefa: TarefaCAV, sucesso: bool, tempo_execucao: float):
        """
        Atualiza aprendizado com base na execução

        Argumentos:
            Tarefa: Tarefa executada
            Sucesso: se a tarefa foi completada dentro do deadline
            tempo_execucao: Tempo de execução da tarefa
        """
        hash_tarefa = self._gerar_hash_tarefa(tarefa)
        timestamp_atual = datetima.now().isonformat()

        if hash_tarefa not in self.historico_desempenho:
            self.historico_desempenho[hash_tarefa] = {
                'fator_aprendizado': 1.0,
                'execucoes_totais': 0,
                'execucoes_sucesso': 0,
                'tempo_medio_execucao': 0.0,
                'contextos_execucao': [],
                'primeira_execucao': timestamp_atual
            }
        
        dados_tarefa = self.historico_desempenho[hash_tarefa]
        dados_tarefa['execucoes_totais'] += 1
        dados_tarefa['ultimas_atualizacao'] = timestamp_atual

        if sucesso:
            dados_tarefa['execucoes_sucesso'] += 1
        
        alpha = 0.3
        if dados_tarefa['tempo_medio_execucao'] == 0:
            dados_tarefa['tempo_medio_execucao'] = tempo_execucao
        else:
            dados_tarefa['tempo_medio_execucao'] = (
                alpha * tempo_execucao +
                ( 1 - alpha ) * dados_tarefa['tempo_medio_execucao']
            )

        contexto_execucao = {
            'clima': self.contexto.clima.value,
            'trafego': self.contexto.trafego.value,
            'velocidade': self.contexto.velocidade_atual,
            'sucesso': sucesso,
            'tempo_execucao': tempo_execucao,
            'timestamp': timestamp_atual
        }

        #atualiza os contextos de execução sempre para no máximo 10 runs

        dados_tarefa['contextos_execucao'].append(contexto_execucao)
        
        if len(dados_tarefa['contextos_execucao'] > 10):
            dados_tarefa['contextos_execucao'].pop(0)

        #calcula novo fator de aprendizado baseado na taxa sucesso
        taxa_sucesso = dados_tarefa['execucoes_sucesso'] / dados_tarefa['execucoes_totais']

        if taxa_sucesso >= 0.9:
            #excelente desempenho
            dados_tarefa['fator_aprendizado'] = 1.5, 1.0 + (taxa_sucesso - 0.9) * 2
        elif taxa_sucesso >= 0.7:
            dados_tarefa['fator_aprendizado'] = 1.0 + (taxa_sucesso - 0.7)
    

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
    Analisa e exibe o desempenho dos escalonadores.
    Agora inclui o "Tempo Simulado Total" do escalonador e formatação melhorada.
    """
    print("\n" + "="*140 + "\n") # Aumentar a largura total para acomodar mais colunas
    print("ANÁLISE DE DESEMPENHO DOS ESCALONADORES".center(140))
    print("\n" + "="*140)

 
    resultados_coletados = []
    for nome, escalonador, tempo_exec_real_algoritmo in escalonadores_com_tempos:
        qtd_tarefas_completadas = [t for t in escalonador.tarefas if t.tempo_final is not None]
        
        qtd_deadlines_perdidos = sum(1 for t in qtd_tarefas_completadas
                                   if t.deadline is not None and (t.tempo_final - t.timestamp_chegada) > t.deadline)
        
        turnarounds = [t.tempo_final - t.timestamp_chegada for t in qtd_tarefas_completadas]
        turnaround_medio = sum(turnarounds) / len(turnarounds) if turnarounds else 0

        if qtd_tarefas_completadas:
            tarefas_com_deadline = [t for t in qtd_tarefas_completadas if t.deadline is not None]
            if tarefas_com_deadline:
                percentual_deadlines_perdidos = (qtd_deadlines_perdidos / len(tarefas_com_deadline)) * 100
            else:
                percentual_deadlines_perdidos = 0
        else:
            percentual_deadlines_perdidos = 0

        resultados_coletados.append({
            'Algoritmo': nome,
            'Tempo Total Real (s)': tempo_exec_real_algoritmo,
            'Tempo Simulado Total (s)': escalonador.tempo_simulacao_final,
            'Turnaround Médio (s)': turnaround_medio,
            'Deadlines Perdidos (%)': percentual_deadlines_perdidos,
            'Trocas Contexto': escalonador.trocas_contexto,
            'Sobrecarga Total (s)': escalonador.sobrecarga_total
        })


    print("\nAnálise de Desempenho:")

    melhor_tempo_simulado = float('inf')
    melhor_algoritmo_tempo_simulado = ""
    menor_sobrecarga = float('inf')
    melhor_algoritmo_sobrecarga = ""
    menor_turnaround = float('inf')
    melhor_algoritmo_turnaround = ""
    menor_deadlines_perdidos = float('inf')
    melhor_algoritmo_deadlines = ""

    for res in resultados_coletados:
        if res["Tempo Simulado Total (s)"] < melhor_tempo_simulado:
            melhor_tempo_simulado = res["Tempo Simulado Total (s)"]
            melhor_algoritmo_tempo_simulado = res["Algoritmo"]
        
        if res["Sobrecarga Total (s)"] < menor_sobrecarga:
            menor_sobrecarga = res["Sobrecarga Total (s)"]
            melhor_algoritmo_sobrecarga = res["Algoritmo"]

        if res["Turnaround Médio (s)"] < menor_turnaround and res["Turnaround Médio (s)"] > 0:
            menor_turnaround = res["Turnaround Médio (s)"]
            melhor_algoritmo_turnaround = res["Algoritmo"]

        if res["Deadlines Perdidos (%)"] < menor_deadlines_perdidos:
            menor_deadlines_perdidos = res["Deadlines Perdidos (%)"]
            melhor_algoritmo_deadlines = res["Algoritmo"]


    print(f"\nAlgoritmo com o menor tempo total de simulação: {melhor_algoritmo_tempo_simulado} ({melhor_tempo_simulado:.2f}s)")
    print(f"Algoritmo com a menor sobrecarga total (simulada): {melhor_algoritmo_sobrecarga} ({menor_sobrecarga:.2f}s)")
    if melhor_algoritmo_turnaround:
        print(f"Algoritmo com o menor Turnaround Médio: {melhor_algoritmo_turnaround} ({menor_turnaround:.2f}s)")
    else:
        print("Não foi possível determinar o melhor Turnaround Médio (todos foram 0 ou não aplicáveis).")
    print(f"Algoritmo com menor porcentagem de Deadlines Perdidos: {melhor_algoritmo_deadlines} ({menor_deadlines_perdidos:.0f}%)")
    
    print("\nNota: 'Tempo Total Real (s)' refere-se ao tempo que o script Python levou para rodar a simulação do algoritmo.")
    print("'Tempo Simulado Total (s)' refere-se ao tempo decorrido dentro da simulação do escalonador.")


# ==============================================================================
# FUNÇÃO PRINCIPAL DE COMPARAÇÃO
# ==============================================================================

def executar_comparacao_algoritmos():
    """
    Executa comparação entre diferentes algoritmos de escalonamento.
    Permite usar tarefas completas ou simples.
    """
    print("=" * 80)
    print("COMPARAÇÃO DE ALGORITMOS DE ESCALONAMENTO CAV")
    print("=" * 80)
    
    # Escolha entre tarefas completas (com deadline e tipo) ou simples
    # Descomente a linha desejada:
    tarefas_base = criar_tarefas_cav_completas() # Para algoritmos que usam todos os atributos
    # tarefas_base = criar_tarefas_simples()      # Para algoritmos mais simples

    # Lista de escalonadores para comparar
    # Adicione seus novos algoritmos aqui, instanciando-os.
    escalonadores = [
        ("FIFO", EscalonadorFIFO()),
        ("Round Robin", EscalonadorRoundRobin()), 
        ("Prioridade Estática", EscalonadorPrioridade())
        # Adicione seu novo algoritmo aqui:
        # ("Meu Novo Algoritmo", MeuNovoAlgoritmo(parametros_se_tiver))
    ]
    
    escalonadores_com_tempos = []

    for nome, escalonador_instancia in escalonadores:
        print(f"\n{'='*20} TESTANDO {nome} {'='*20}")
        
        # Copia tarefas para cada escalonador (reset de estado)
        # É crucial que as tarefas sejam resetadas e adicionadas a cada escalonador individualmente
        tarefas_copia = copy.deepcopy(tarefas_base)
        
        for tarefa in tarefas_copia:
            tarefa.tempo_restante = tarefa.duracao
            tarefa.tempo_inicio = None
            tarefa.tempo_final = None
            tarefa.tempo_espera = 0.0
            tarefa.tempo_resposta = None
            tarefa.iniciado = False
            tarefa.timestamp_chegada = 0.0 

        escalonador_instancia.tarefas = []
        for tarefa in tarefas_copia:
            escalonador_instancia.adicionar_tarefa(tarefa)

        tempo_inicio_real_algoritmo = time.time()
        escalonador_instancia.escalonar()
        tempo_fim_real_algoritmo = time.time()
        tempo_exec_real_algoritmo = tempo_fim_real_algoritmo - tempo_inicio_real_algoritmo
        
        escalonadores_com_tempos.append((nome, escalonador_instancia, tempo_exec_real_algoritmo))
    
    # Analisar desempenho de todos os escalonadores
    analisar_desempenho(escalonadores_com_tempos)

if __name__ == "__main__":
    contexto_exemplo = ContextoConducao(
        clima=CondicaoClimatica.CHUVA,
        tipo_via=TipoVia.RODOVIA,
        trafego=DensidadeTrafego.ALTA,
        velocidade_atual=95.0,
        modo_autonomo=True
    )
    
    # Executa a comparação entre os algoritmos
    executar_comparacao_algoritmos()