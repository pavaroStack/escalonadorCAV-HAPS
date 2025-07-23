import random
import time
import copy
import os
import json
import heapq
import math
from collections import deque
from abc import ABC, abstractmethod
from enum import Enum
from typing import List, Dict, Optional, Any, Tuple
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

        self.preempcoes_consecutivas: int = 0

    def __str__(self):
        return f"Tarefa {self.nome} (Prioridade {self.prioridade}): {self.duracao} segundos"

    def executar(self, quantum: int, tempo_atual_simulado: float) -> float:
        """
        Executa a tarefa por um quantum de tempo usando tempo simulado
        """
        if not self.iniciado:
            self.tempo_inicio = tempo_atual_simulado
            self.tempo_resposta = self.tempo_inicio - self.timestamp_chegada
            self.iniciado = True

        tempo_execucao = min(quantum, self.tempo_restante)
        self.tempo_restante -= tempo_execucao
        
        if self.tempo_restante <= 0:
            self.tempo_final = tempo_atual_simulado + tempo_execucao
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

        # historico aprendizado -> [{ [ hash_tarefa: {DATA}, hash_tarefa: {DATA}, ... ]} ]
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
                'tipo_via': self.contexto.tipo_via.value,
                'trafego': self.contexto.trafego.value,
                'velocidade_atual': self.contexto.velocidade_atual,
                'modo_autonomo': self.contexto.modo_autonomo
            }
            
            dados_persistencia = {
                'versao_esquema': '1.0',
                'timestamp_salvamento': datetime.now().isoformat(),
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
        return f"{tarefa.nome}_{tarefa.tipo_processo.name}_{contexto_hash}"
    
    def _atualizar_aprendizado_ia(self, tarefa: TarefaCAV, sucesso: bool, tempo_execucao: float):
        """
        Atualiza aprendizado com base na execução

        Argumentos:
            Tarefa: Tarefa executada
            Sucesso: se a tarefa foi completada dentro do deadline
            tempo_execucao: Tempo de execução da tarefa
        """
        hash_tarefa = self._gerar_hash_tarefa(tarefa)
        timestamp_atual = datetime.now().isoformat()

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
        
        if len(dados_tarefa['contextos_execucao']) > 10:
            dados_tarefa['contextos_execucao'].pop(0)

        #calcula novo fator de aprendizado baseado na taxa sucesso, quanto melhor a taxa de sucesso, menos ela precisa aprender
        taxa_sucesso = dados_tarefa['execucoes_sucesso'] / dados_tarefa['execucoes_totais']

        if taxa_sucesso >= 0.9:
            #excelente desempenho
            dados_tarefa['fator_aprendizado'] = min(1.5, 1.0 + (taxa_sucesso - 0.9) * 2)
        elif taxa_sucesso >= 0.7:
            dados_tarefa['fator_aprendizado'] = 1.0 + (taxa_sucesso - 0.7)
        elif taxa_sucesso >= 0.5:
            dados_tarefa['fator_aprendizado'] = 1.0
        else:
            dados_tarefa['fator_aprendizado'] = max(0.7, 0.5 + taxa_sucesso)
        
        print(f"[IA-LEARN] {hash_tarefa}: Taxa sucesso {taxa_sucesso:.2f} -> Fator {dados_tarefa['fator_aprendizado']:.3f}")

    def prever_tempo_exec(self, tarefa: TarefaCAV) -> float:

        hash_tarefa = self._gerar_hash_tarefa(tarefa)
        #? se a tarefa não estiver no histórico de execucção e nem no modelo preditivo então salva o peso dela como 0.0
        #! se a tarefa não estiver no historico mas estiver no modelo retorna o peso dela, caso seja maior que 0.1
        if not tarefa.historico_exec:
            if hash_tarefa not in self.modelo_pred:
                self.modelo_pred[hash_tarefa] = (tarefa.duracao, 0.0)
            return max(0.1, tarefa.duracao)
        
        #? se o modelo de tarefa existe no historico mas não existe no modelo preditivo, inicializa ela
        if hash_tarefa not in self.modelo_pred:
            self.modelo_pred[hash_tarefa] = (tarefa.duracao, 0.0)
        
        #intercept e slope do modelo de regressão linear
        intercept, slope = self.modelo_pred[hash_tarefa]

        #quantas vezes a tarefa ja foi executada no escalonamento
        x = len(tarefa.historico_exec)

        tempo_restante_previsto = intercept + slope * x
        return max(0.1, tempo_restante_previsto)
    
    def atualizar_modelo(self, tarefa: TarefaCAV, tempo_real_executado: float):

        hash_tarefa = self._gerar_hash_tarefa(tarefa)

        if not tarefa.historico_exec:
            return
        
        #obtém o numero de execuções (quantums) 
        x_antigo = len(tarefa.historico_exec) - 1 # A execução mais recente ainda não foi utilizado para o modelo
        if x_antigo < 0:
            x_antigo = 0

        #parâmetros atuais do modelo
        intercept, slope = self.modelo_pred.get(hash_tarefa, (tarefa.duracao, 0.0)) 

        # Predição do tempo para a próxima execução (antes desta atualização) seria baseada em x_antigo
        # aqui é aplicada a regressão linear
        tempo_previsto = intercept + slope * x_antigo

        erro = tempo_real_executado - tempo_previsto

        novo_intercept = intercept + (self.alpha * erro)
        novo_slope = slope + (self.alpha * erro * x_antigo)

        self.modelo_pred[hash_tarefa] = (novo_intercept, novo_slope)

        print(f"[REGRESSÃO LINEAR] Modelo atualizado para {hash_tarefa}: intercept={novo_intercept:.2f}, slope={novo_slope:.2f} (erro={erro:.2f})")

    # DENTRO DA CLASSE EscalonadorHAPS
    def calcular_quantum(self, tarefa: TarefaCAV) -> int:
        quantum_base = self.quantum_base_por_tipo[tarefa.tipo_processo]

        hash_tarefa = self._gerar_hash_tarefa(tarefa)
        if hash_tarefa in self.historico_desempenho:
            dados_tarefa = self.historico_desempenho[hash_tarefa]
            tempo_medio = dados_tarefa.get("tempo_medio_execucao", quantum_base)

            # MUDANÇA: Lógica de ajuste simplificada e mais estável
            # Ajusta o quantum proporcionalmente ao tempo médio de execução histórico.
            fator_ajuste = tempo_medio / quantum_base
            quantum_ajustado = quantum_base * fator_ajuste
            
            # MUDANÇA: Limites corrigidos e mais razoáveis para evitar comportamento extremo
            # Permite que o quantum diminua até a metade do base (mínimo 1)
            quantum_min = max(1, quantum_base // 2) # <-- CORREÇÃO DE BUG E LÓGICA
            # Não deixa o quantum explodir, no máximo o dobro do base ou 10
            quantum_max = min(quantum_base * 2, 10)  # <-- LIMITE MAIS SEGURO

            quantum_final = int(round(
                max(quantum_min, min(quantum_ajustado, quantum_max))
            ))
            
            # print(f"[QUANTUM] {tarefa.nome}: base={quantum_base}, médio={tempo_medio:.2f}, ajustado={quantum_final}")
            return quantum_final
            
        return quantum_base

    # DENTRO DA CLASSE EscalonadorHAPS

    def calcular_prioridade_dinamica(self, tarefa: TarefaCAV, tempo_atual_simulado: float) -> float:
        """
        Calcula a prioridade usando um modelo de UTILIDADE PONDERADA.
        Esta abordagem é mais estável e estratégica que a multiplicação de fatores.
        """
        
        # --- Pesos (Weights): Define a "personalidade" do escalonador ---
        # Estes são os únicos valores que você talvez queira ajustar no futuro.
        # A soma deles não precisa ser 1. O que importa é a proporção entre eles.
        w_urgencia = 5.0    # MUITO IMPORTANTE: Foco total em não perder deadlines.
        w_importancia = 2.0 # IMPORTANTE: Tarefas de segurança têm grande vantagem.
        w_eficiencia = 1.5  # Bônus para tarefas que estão quase terminando (melhora o throughput).
        w_espera = 1.0      # Garante que ninguém seja esquecido (anti-starvation).

        # --- Cálculo das Pontuações (Scores) para cada fator ---

        # 1. Pontuação de IMPORTÂNCIA (Static Score)
        # Baseado no tipo de processo. Varia de 20 (Diagnóstico) a 100 (Segurança).
        score_importancia = tarefa.tipo_processo.value if tarefa.tipo_processo else 20

        # 2. Pontuação de URGÊNCIA (Deadline Score)
        score_urgencia = 0.0
        if tarefa.deadline is not None:
            tempo_restante_deadline = tarefa.deadline - tempo_atual_simulado
            if tempo_restante_deadline <= 0:
                # EMERGÊNCIA MÁXIMA: O deadline já passou ou está passando agora.
                score_urgencia = 5000 
            else:
                # A pontuação aumenta exponencialmente quanto mais perto o deadline estiver.
                score_urgencia = 1000 / tempo_restante_deadline

        # 3. Pontuação de EFICIÊNCIA (Shortest Remaining Time Score)
        # Dá um bônus para tarefas que estão perto de serem concluídas.
        score_eficiencia = 100 / max(1.0, tarefa.tempo_restante)

        # 4. Pontuação de ESPERA (Aging Score)
        # A pontuação aumenta linearmente com o tempo que a tarefa está esperando.
        tempo_espera = tempo_atual_simulado - tarefa.timestamp_chegada
        score_espera = tempo_espera
        
        # --- Cálculo Final da Prioridade ---
        # Soma ponderada de todas as pontuações.
        prioridade_final = (
            (w_importancia * score_importancia) +
            (w_urgencia * score_urgencia) +
            (w_eficiencia * score_eficiencia) +
            (w_espera * score_espera)
        )

        # print(f"Tarefa: {tarefa.nome} | Prio Final: {prioridade_final:.2f} | Urg: {score_urgencia:.2f} | Imp: {score_importancia:.2f} | Efi: {score_eficiencia:.2f} | Esp: {score_espera:.2f}")

        return prioridade_final
    
    def adicionar_tarefa_fila(self, tarefa: TarefaCAV, tempo_atual_simulado: float):
        """Adiciona tarefa à fila apropriada com prioridade dinâmica calculada no tempo simulado."""
        prioridade_dinamica = self.calcular_prioridade_dinamica(tarefa, tempo_atual_simulado)
        processo = ProcessoComPrioridade(tarefa, prioridade_dinamica)

        # Garante que o tipo de processo exista antes de acessar o dicionário de filas
        if tarefa.tipo_processo in self.filas_por_tipo:
            fila = self.filas_por_tipo[tarefa.tipo_processo]
            heapq.heappush(fila, processo)
            print(f"[HAPS] Adicionada: {tarefa.nome} ({tarefa.tipo_processo.name}) - Prioridade Dinâmica: {prioridade_dinamica:.2f}")
        else:
            print(f"[AVISO] Tarefa '{tarefa.nome}' sem tipo de processo válido. Não pode ser adicionada à fila HAPS.")
    
    def selecionar_proxima_tarefa(self) -> Optional[TarefaCAV]:
        """
        Seleciona próxima tarefa seguindo hierarquia de criticidade
        
        Returns:
            Próxima tarefa a ser executada ou None se não houver tarefas
        """
        # Ordem hierárquica de seleção
        filas_ordenadas = [
            (self.fila_seguranca, "SEGURANÇA CRÍTICA"),
            (self.fila_tempo_real, "TEMPO REAL"),
            (self.fila_navegacao, "NAVEGAÇÃO"),
            (self.fila_conforto, "CONFORTO"),
            (self.fila_diagnostico, "DIAGNÓSTICO")
        ]
        
        for fila, nome_fila in filas_ordenadas:
            if fila:
                processo = heapq.heappop(fila)
                print(f"[HAPS] Selecionada da fila {nome_fila}: {processo.tarefa.nome} (Prio: {processo.prioridade_dinamica:.2f})")
                return processo.tarefa
        
        return None
    
    def todas_filas_vazias(self) -> bool:
        """Verifica se todas as filas estão vazias"""
        return all(len(fila) == 0 for fila in self.filas_por_tipo.values())
    
    def reavaliar_prioridades_filas(self, tempo_atual_simulado: float) -> float:
        """Retorna tempo simulado gasto na reavaliação"""
        print("\n[REAVALIAÇÃO] Atualizando prioridades...")
        tempo_sobrecarga = 0.05  # Tempo simulado fixo para reavaliação
        
        for tipo, fila in self.filas_por_tipo.items():
            tarefas_temp = [heapq.heappop(fila).tarefa for _ in range(len(fila))]
            
            for tarefa in tarefas_temp:
                prioridade = self.calcular_prioridade_dinamica(tarefa, tempo_atual_simulado)
                heapq.heappush(fila, ProcessoComPrioridade(tarefa, prioridade))
        
        self.registrar_sobrecarga(tempo_sobrecarga)
        return tempo_sobrecarga
    
    # DENTRO DA CLASSE EscalonadorHAPS
    def escalonar(self):
        print(f"\n=== INICIANDO ESCALONAMENTO HAPS-AI ===")
        tempo_atual_simulado = 0.0
        ciclo = 1
        
        for tarefa in self.tarefas:
            if tarefa.tipo_processo is None:
                print(f"[AVISO] Tarefa '{tarefa.nome}' pulada por não ter um TipoProcessoCAV definido.")
                continue
            tarefa.timestamp_chegada = tempo_atual_simulado
            self.adicionar_tarefa_fila(tarefa, tempo_atual_simulado)
        
        while not self.todas_filas_vazias():
            print(f"\n--- CICLO {ciclo} (Tempo Simulado: {tempo_atual_simulado:.2f}s) ---")
            
            # ================================================================= #
            # ======== REMOÇÃO DA REAVALIAÇÃO PERIÓDICA DE PRIORIDADES ========#
            # Esta era a principal causa da baixa performance na simulação.    #
            # ================================================================= #
            # if ciclo > 1 and ciclo % self.intervalo_reavaliacao == 0:
            #     tempo_reavaliacao = self.reavaliar_prioridades_filas(tempo_atual_simulado)
            #     tempo_atual_simulado += tempo_reavaliacao
            
            tarefa_atual = self.selecionar_proxima_tarefa()
            if tarefa_atual is None:
                break
                
            quantum = self.calcular_quantum(tarefa_atual)
            print(f"[EXEC] Executando {tarefa_atual.nome} por {quantum} unidades (restante: {tarefa_atual.tempo_restante})")
            
            tempo_executado = tarefa_atual.executar(quantum, tempo_atual_simulado)
            tempo_atual_simulado += tempo_executado
            
            self.atualizar_modelo(tarefa_atual, tempo_executado)
            
            tempo_sobrecarga = 0.01  # Custo de troca de contexto mantido baixo
            self.registrar_sobrecarga(tempo_sobrecarga)
            tempo_atual_simulado += tempo_sobrecarga
            
            if tarefa_atual.tempo_restante <= 0:
                tempo_turnaround = tarefa_atual.tempo_final - tarefa_atual.timestamp_chegada
                deadline_perdido = (tarefa_atual.deadline is not None and 
                                    tempo_turnaround > tarefa_atual.deadline)
                
                tarefa_atual.preempcoes_consecutivas = 0
                print(f"[COMP] {tarefa_atual.nome} COMPLETADA! Turnaround: {tempo_turnaround:.2f}s")
                
                self._atualizar_aprendizado_ia(tarefa_atual, not deadline_perdido, tempo_turnaround)
            else:
                tarefa_atual.preempcoes_consecutivas += 1
                print(f"[PREE] {tarefa_atual.nome} preemptada, recolocando na fila")
                self.adicionar_tarefa_fila(tarefa_atual, tempo_atual_simulado)
            
            ciclo += 1
        
        self.tempo_simulacao_final = tempo_atual_simulado
        print(f"\n=== ESCALONAMENTO CONCLUÍDO ===")
        print(f"Tempo simulado total: {self.tempo_simulacao_final:.2f}s")
        
        self._salvar_historico_ia()
        self._coletar_metricas_execucao()
        self.calcular_metricas()
        self.exibir_sobrecarga()
    
    def _coletar_metricas_execucao(self):
        """Coleta métricas detalhadas da execução para logging"""
        qtd_tarefas_completadas = [t for t in self.tarefas if t.tempo_final is not None]
        qtd_deadlines_perdidos = sum(1 for t in qtd_tarefas_completadas 
                               if t.tempo_final and (t.tempo_final - t.timestamp_chegada) > t.deadline)
        
        if qtd_tarefas_completadas:
            tempo_medio_espera = sum(t.tempo_espera for t in qtd_tarefas_completadas) / len(qtd_tarefas_completadas)
            tempo_medio_resposta = sum(t.tempo_resposta for t in qtd_tarefas_completadas 
                                     if t.tempo_resposta is not None) / len(qtd_tarefas_completadas)
        else:
            tempo_medio_espera = 0
            tempo_medio_resposta = 0
        
        # Métricas por tipo de processo
        metricas_por_tipo = {}
        for tipo in TipoProcessoCAV:
            tarefas_tipo = [t for t in qtd_tarefas_completadas if t.tipo_processo == tipo]
            deadlines_tipo = sum(1 for t in tarefas_tipo 
                               if (t.tempo_final - t.timestamp_chegada) > t.deadline)
            
            metricas_por_tipo[tipo.name] = {
                'completadas': len(tarefas_tipo),
                'qtd_deadlines_perdidos': deadlines_tipo,
                'taxa_sucesso': (len(tarefas_tipo) - deadlines_tipo) / max(1, len(tarefas_tipo))
            }
        
        total_tarefas = len(self.tarefas)

        taxa_sucesso_geral = ((len(qtd_tarefas_completadas) - qtd_deadlines_perdidos) / len(self.tarefas)) if (total_tarefas > 0) else 0
        
        self.metricas_execucao = {
            'tarefas_totais': len(self.tarefas),
            'qtd_tarefas_completadas': len(qtd_tarefas_completadas),
            'qtd_deadlines_perdidos': qtd_deadlines_perdidos,
            'tempo_medio_espera': tempo_medio_espera,
            'tempo_medio_resposta': tempo_medio_resposta,
            'sobrecarga_total': self.sobrecarga_total,
            'trocas_contexto': self.trocas_contexto,
            'taxa_sucesso_geral': taxa_sucesso_geral,
            'metricas_por_tipo': metricas_por_tipo,
            'conhecimento_ia': len(self.historico_desempenho),
            'modelo_pred': {
                hash_tarefa: params for hash_tarefa, params in self.modelo_pred.items()
            }
        }

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