#!/usr/bin/env python3
"""
HAPS-AI CAV Scheduler - Escalonador Híbrido e Adaptativo com IA para Veículos Autônomos
Sistema de escalonamento de processos críticos para Connected Autonomous Vehicles (CAVs)
"""

import random
import time
import heapq
import json
import os
import math
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import csv
import copy
from collections import deque
from abc import ABC, abstractmethod
from enum import Enum
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
nome_arquivo = "dados"
# ===== ESTRUTURAS DE DADOS BASE =====

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

class TarefaCAV:
    """Representa uma tarefa/processo em um sistema CAV"""
    
    def __init__(self, nome: str, duracao: int, prioridade: int, deadline: int, tipo_processo: TipoProcessoCAV):
        self.nome = nome
        self.duracao = duracao
        self.prioridade = prioridade
        self.deadline = deadline
        self.tipo_processo = tipo_processo
        
        # Atributos de controle de execução
        self.tempo_restante = duracao
        self.tempo_inicio: Optional[float] = None
        self.tempo_final: Optional[float] = None
        self.tempo_espera: float = 0.0
        self.tempo_resposta: Optional[float] = None
        self.iniciado = False
        self.timestamp_chegada = time.time()
        
        # Histórico de execução (para o ARTS-AD)
        self.historico_exec: List[float] = []  # Tempos reais de execução em cada quantum
        
        # Contador de preempções consecutivas para prevenção de starvation
        self.preemptions_consecutivas: int = 0
    
    def executar(self, quantum: int) -> float:
        """
        Executa a tarefa por um quantum de tempo
        
        Args:
            quantum: Tempo de execução em unidades
            
        Returns:
            Tempo real de execução consumido
        """
        if not self.iniciado:
            self.tempo_inicio = time.time()
            self.tempo_resposta = self.tempo_inicio - self.timestamp_chegada
            self.iniciado = True

        # Simula execução
        tempo_execucao = min(quantum, self.tempo_restante)
        self.tempo_restante -= tempo_execucao
        
        if self.tempo_restante <= 0:
            self.tempo_final = time.time()
            self.tempo_espera = self.tempo_final - self.timestamp_chegada - self.duracao
        
        # Registra tempo real de execução
        self.historico_exec.append(tempo_execucao)
        return tempo_execucao
    
    def __repr__(self):
        return f"TarefaCAV({self.nome}, {self.tipo_processo.name}, dur={self.duracao}, deadline={self.deadline})"

class ProcessoComPrioridade:
    """Wrapper para tarefa com prioridade dinâmica para uso com heapq"""
    
    def __init__(self, tarefa: TarefaCAV, prioridade_dinamica: float):
        self.tarefa = tarefa
        self.prioridade_dinamica = prioridade_dinamica
    
    def __lt__(self, other):
        """
        Comparação para heapq: prioridade maior primeiro, em caso de empate, deadline menor primeiro (EDF)
        Nota: heapq é min-heap, então invertemos a prioridade para simular max-heap
        """
        if abs(self.prioridade_dinamica - other.prioridade_dinamica) < 0.001:
            # Em caso de empate de prioridade, usa EDF (deadline menor primeiro)
            return self.tarefa.deadline < other.tarefa.deadline
        # Prioridade maior primeiro (invertemos para min-heap)
        return self.prioridade_dinamica > other.prioridade_dinamica
    
    def __repr__(self):
        return f"ProcessoComPrioridade({self.tarefa.nome}, prio={self.prioridade_dinamica:.2f})"

# ===== ESCALONADORES =====

class EscalonadorCAV(ABC):
    """Classe base para escalonadores CAV"""

    def __init__(self):
        super().__init__()
        self.tarefas: List[TarefaCAV] = []
        self.sobrecarga_total: float = 0.0
        self.trocas_contexto: int = 0
    
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
            return
        
        qtd_tarefas_completadas = [t for t in self.tarefas if t.tempo_final is not None]
        qtd_deadlines_perdidos = sum(1 for t in qtd_tarefas_completadas 
                               if t.tempo_final and (t.tempo_final - t.timestamp_chegada) > t.deadline)
        
        if qtd_tarefas_completadas:
            tempo_medio_espera = sum(t.tempo_espera for t in qtd_tarefas_completadas) / len(qtd_tarefas_completadas)
            tempo_medio_resposta = sum(t.tempo_resposta for t in qtd_tarefas_completadas 
                                     if t.tempo_resposta is not None) / len(qtd_tarefas_completadas)
            # NOVO: Calcular turnaround médio (tempo_total - chegada)
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
        print(f"Tempo médio de turnaround: {tempo_medio_turnaround:.2f}s")  # NOVA MÉTRICA
        print(f"Deadlines perdidos: {qtd_deadlines_perdidos}")
        print(f"Taxa de sucesso: {((len(qtd_tarefas_completadas) - qtd_deadlines_perdidos) / len(self.tarefas) * 100):.1f}%" if self.tarefas else f"taxa de sucesso {0:.1f}%")
        
        # Armazenar para comparação
        self.metricas = {
            'turnaround_medio': tempo_medio_turnaround,
            'espera_medio': tempo_medio_espera,
            'resposta_medio': tempo_medio_resposta,
            'qtd_deadlines_perdidos': qtd_deadlines_perdidos,
            'taxa_sucesso': ((len(qtd_tarefas_completadas) - qtd_deadlines_perdidos) / len(self.tarefas) * 100) if self.tarefas else 0
        }

# ===== MODIFICAÇÕES NA FUNÇÃO executar_comparacao_algoritmos =====

class EscalonadorHAPS(EscalonadorCAV):
    """
        Escalonador Híbrido e Adaptativo com Prioridade e Sobrevivência para CAVs
        
        Características principais:
        - Escalonamento preemptivo baseado em quantum
        - Prioridade dinâmica recalculada a cada inserção
        - Seleção hierárquica por criticidade
        - Prevenção de starvation através de aging
        - Aprendizado baseado em histórico de desempenho PERSISTENTE
        - Modelo preditivo ARTS-AD granular por contexto
        - Quantum adaptativo baseado em aprendizado
        - Prevenção proativa de starvation
    """
    def __init__(self, contexto: ContextoConducao, arquivo_historico: str = "haps_ai_historico.json"):
        super().__init__()
        self.contexto = contexto
        self.arquivo_historico = arquivo_historico
        self.historico_desempenho: Dict[str, Dict[str, Any]] = {}
        
        # Parâmetros do modelo preditivo ARTS-AD
        self.alpha = 0.1  # Fator de aprendizado
        self.modelo_pred: Dict[str, Tuple[float, float]] = {}  # hash_tarefa -> (intercept, slope)
        
        # Carrega histórico persistente de aprendizado
        self._carregar_historico_ia()
        
        # Filas hierárquicas para cada tipo de processo
        self.fila_seguranca: List[ProcessoComPrioridade] = []
        self.fila_tempo_real: List[ProcessoComPrioridade] = []
        self.fila_navegacao: List[ProcessoComPrioridade] = []
        self.fila_conforto: List[ProcessoComPrioridade] = []
        self.fila_diagnostico: List[ProcessoComPrioridade] = []
        
        # Mapeamento de tipos para filas
        self.filas_por_tipo = {
            TipoProcessoCAV.SEGURANCA_CRITICA: self.fila_seguranca,
            TipoProcessoCAV.TEMPO_REAL: self.fila_tempo_real,
            TipoProcessoCAV.NAVEGACAO: self.fila_navegacao,
            TipoProcessoCAV.CONFORTO: self.fila_conforto,
            TipoProcessoCAV.DIAGNOSTICO: self.fila_diagnostico
        }
        
        # Quantum base por tipo de processo
        self.quantum_base_por_tipo = {
            TipoProcessoCAV.SEGURANCA_CRITICA: 2,
            TipoProcessoCAV.TEMPO_REAL: 3,
            TipoProcessoCAV.NAVEGACAO: 4,
            TipoProcessoCAV.CONFORTO: 5,
            TipoProcessoCAV.DIAGNOSTICO: 6
        }
        
        # Métricas de execução para logging
        self.metricas_execucao = {}
        
        # Contador para reavaliação periódica
        self.contador_ciclos = 0
        self.intervalo_reavaliacao = 3  # Reavaliar prioridades a cada 3 ciclos
    
    def _carregar_historico_ia(self):
        """
        Carrega histórico persistente de aprendizado da IA
        
        Estrutura do histórico:
        {
            "tarefa_hash": {
                "fator_aprendizado": float,
                "execucoes_totais": int,
                "execucoes_sucesso": int,
                "tempo_medio_execucao": float,
                "contextos_execucao": [list],
                "ultima_atualizacao": timestamp
            }
        }
        """
        try:
            if os.path.exists(self.arquivo_historico):
                with open(self.arquivo_historico, 'r', encoding='utf-8') as f:
                    dados_historico = json.load(f)
                    self.historico_desempenho = dados_historico.get('historico_aprendizado', {})
                    
                    # Carregar modelo_pred se existir
                    modelo_pred_salvo = dados_historico.get('modelo_pred', {})
                    for hash_tarefa, params in modelo_pred_salvo.items():
                        # Garantir que os parâmetros sejam tuplas
                        if isinstance(params, list):
                            self.modelo_pred[hash_tarefa] = tuple(params)
                        else:
                            self.modelo_pred[hash_tarefa] = params
                    
                    print(f"[IA] Histórico carregado: {len(self.historico_desempenho)} tarefas aprendidas")
                    print(f"[IA] Modelo preditivo carregado: {len(self.modelo_pred)} entradas")
                    self._exibir_estatisticas_ia()
            else:
                print(f"[IA] Arquivo de histórico não encontrado, iniciando aprendizado fresh")
                self.historico_desempenho = {}
                self.modelo_pred = {}
                
        except (json.JSONDecodeError, IOError) as e:
            print(f"[ERRO] Falha ao carregar histórico IA: {e}")
            print("[IA] Iniciando com histórico vazio por segurança")
            self.historico_desempenho = {}
            self.modelo_pred = {}
    
    def _salvar_historico_ia(self):
        """Persiste o histórico de aprendizado da IA"""
        try:
            # Converter contexto para dicionário serializável
            contexto_dict = {
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
    
    def _exibir_estatisticas_ia(self):
        """Exibe estatísticas do aprendizado da IA"""
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
        Gera hash mais robusto para identificação de tarefas incluindo contexto
        
        Inclui nome, tipo e contexto básico para melhor rastreamento
        """
        contexto_hash = f"{self.contexto.clima.value}_{self.contexto.trafego.value}_{self.contexto.modo_autonomo}"
        return f"{tarefa.nome}_{tarefa.tipo_processo.name}_{contexto_hash}"
    
    def _atualizar_aprendizado_ia(self, tarefa: TarefaCAV, sucesso: bool, tempo_execucao: float):
        """
        Atualiza o aprendizado da IA baseado no resultado da execução
        
        Args:
            tarefa: Tarefa executada
            sucesso: Se a tarefa foi completada dentro do deadline
            tempo_execucao: Tempo real de execução
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
        dados_tarefa['ultima_atualizacao'] = timestamp_atual
        
        if sucesso:
            dados_tarefa['execucoes_sucesso'] += 1
        
        # Atualiza tempo médio usando média móvel
        alpha = 0.3  # Fator de suavização
        if dados_tarefa['tempo_medio_execucao'] == 0:
            dados_tarefa['tempo_medio_execucao'] = tempo_execucao
        else:
            dados_tarefa['tempo_medio_execucao'] = (
                alpha * tempo_execucao + 
                (1 - alpha) * dados_tarefa['tempo_medio_execucao']
            )
        
        # Registra contexto de execução (mantém apenas os 10 mais recentes)
        contexto_execucao = {
            'clima': self.contexto.clima.value,
            'trafego': self.contexto.trafego.value,
            'velocidade': self.contexto.velocidade_atual,
            'sucesso': sucesso,
            'tempo_execucao': tempo_execucao,
            'timestamp': timestamp_atual
        }
        
        dados_tarefa['contextos_execucao'].append(contexto_execucao)
        if len(dados_tarefa['contextos_execucao']) > 10:
            dados_tarefa['contextos_execucao'].pop(0)
        
        # Calcula novo fator de aprendizado baseado na taxa de sucesso
        taxa_sucesso = dados_tarefa['execucoes_sucesso'] / dados_tarefa['execucoes_totais']
        
        if taxa_sucesso >= 0.9:
            # Excelente desempenho - aumenta prioridade
            dados_tarefa['fator_aprendizado'] = min(1.5, 1.0 + (taxa_sucesso - 0.9) * 2)
        elif taxa_sucesso >= 0.7:
            # Bom desempenho - ligeiro aumento
            dados_tarefa['fator_aprendizado'] = 1.0 + (taxa_sucesso - 0.7) * 0.5
        elif taxa_sucesso >= 0.5:
            # Desempenho neutro
            dados_tarefa['fator_aprendizado'] = 1.0
        else:
            # Desempenho ruim - reduz prioridade temporariamente
            dados_tarefa['fator_aprendizado'] = max(0.7, 0.5 + taxa_sucesso)
        
        print(f"[IA-LEARN] {hash_tarefa}: Taxa sucesso {taxa_sucesso:.2f} -> Fator {dados_tarefa['fator_aprendizado']:.3f}")
    
    # ======== MODELO PREDITIVO ARTS-AD (CORRIGIDO E GRANULAR) ========
    
    def prever_tempo_exec(self, tarefa: TarefaCAV) -> float:
        """
        Prediz o tempo restante de execução da tarefa usando modelo ARTS-AD granular
        
        Args:
            tarefa: Tarefa a ser analisada
            
        Returns:
            Tempo restante previsto em unidades
        """
        hash_tarefa = self._gerar_hash_tarefa(tarefa)
        
        # Se não há histórico, inicializa com a duração original
        if not tarefa.historico_exec:
            # Inicializa o modelo para essa tarefa se não existir
            if hash_tarefa not in self.modelo_pred:
                # Intercepto = duração original, slope = 0 (comportamento esperado)
                self.modelo_pred[hash_tarefa] = (tarefa.duracao, 0.0)
            return max(0.1, tarefa.duracao)
        
        # Se o modelo para essa tarefa não existe, cria um inicial
        if hash_tarefa not in self.modelo_pred:
            # Inicializa com intercepto = duração restante média? Ou duração original? 
            # Usamos a duração original como baseline
            self.modelo_pred[hash_tarefa] = (tarefa.duracao, 0.0)
        
        intercept, slope = self.modelo_pred[hash_tarefa]
        
        # X é o número de vezes que a tarefa foi executada (quantuns)
        x = len(tarefa.historico_exec)
        
        tempo_restante_previsto = intercept + slope * x
        return max(0.1, tempo_restante_previsto)
    
    def atualizar_modelo(self, tarefa: TarefaCAV, tempo_real_executado: float):
        """
        Atualiza o modelo preditivo granular com dados de execução reais
        
        Args:
            tarefa: Tarefa executada
            tempo_real_executado: Tempo real de execução no último quantum
        """
        hash_tarefa = self._gerar_hash_tarefa(tarefa)
        
        # Se não temos histórico, não podemos atualizar
        if not tarefa.historico_exec:
            return
        
        # Obtém o número de execuções (quantuns) antes desta última execução
        x_antigo = len(tarefa.historico_exec) - 1  # O índice da última execução ainda não foi usado para o modelo
        if x_antigo < 0:
            x_antigo = 0
        
        # Obtém os parâmetros atuais do modelo
        intercept, slope = self.modelo_pred.get(hash_tarefa, (tarefa.duracao, 0.0))
        
        # Predição do tempo para a próxima execução (antes desta atualização) seria baseada em x_antigo
        tempo_previsto = intercept + slope * x_antigo
        
        # Erro: diferença entre o tempo real executado e o previsto
        erro = tempo_real_executado - tempo_previsto
        
        # Atualiza os parâmetros
        novo_intercept = intercept + self.alpha * erro
        novo_slope = slope + self.alpha * erro * x_antigo
        
        # Atualiza o modelo
        self.modelo_pred[hash_tarefa] = (novo_intercept, novo_slope)
        
        print(f"[ARTS-AD] Modelo atualizado para {hash_tarefa}: intercept={novo_intercept:.2f}, slope={novo_slope:.2f} (erro={erro:.2f})")
    
    # ======== QUANTUM ADAPTATIVO BASEADO EM APRENDIZADO ========
    
    def calcular_quantum(self, tarefa: TarefaCAV) -> int:
        """Calcula quantum adaptativo baseado no tipo de processo e aprendizado"""
        # Quantum base pelo tipo
        quantum_base = self.quantum_base_por_tipo[tarefa.tipo_processo]
        
        # Se temos histórico de aprendizado para essa tarefa, ajustamos o quantum
        hash_tarefa = self._gerar_hash_tarefa(tarefa)
        if hash_tarefa in self.historico_desempenho:
            dados_tarefa = self.historico_desempenho[hash_tarefa]
            tempo_medio_execucao = dados_tarefa.get('tempo_medio_execucao', quantum_base)
            
            # Fator de ajuste: se o tempo médio de execução for menor que o quantum_base, reduzimos o quantum
            # Se for maior, aumentamos, mas com limites
            fator_ajuste = tempo_medio_execucao / quantum_base
            
            # Aplicamos o fator, mas com limites
            quantum_ajustado = quantum_base * fator_ajuste
            
            # Limites: mínimo 1, máximo 10 (ou o quantum_base * 2, o que for menor)
            quantum_min = max(1, quantum_base // 2)  # Pelo menos 1 e metade do base
            quantum_max = min(quantum_base * 2, 10)  # Máximo de 10 ou o dobro do base
            
            quantum_final = max(quantum_min, min(quantum_ajustado, quantum_max))
            quantum_final = int(round(quantum_final))
            
            print(f"[QUANTUM] {tarefa.nome}: base={quantum_base}, médio={tempo_medio_execucao:.2f}, ajustado={quantum_final}")
            return quantum_final
        else:
            return quantum_base
    
    # ======== CÁLCULO DE PRIORIDADE DINÂMICA (COM PREVENÇÃO DE STARVATION) ========
    
    def calcular_prioridade_dinamica(self, tarefa: TarefaCAV) -> float:
        """
        Calcula a prioridade dinâmica baseada em múltiplos fatores adaptativos
        
        A prioridade é calculada como:
        P = P_base × W_tipo × F_contexto × F_urgência × F_aprendizado × F_aging × F_preditivo × F_anti_starvation
        
        Args:
            tarefa: Tarefa CAV para calcular prioridade
            
        Returns:
            Prioridade dinâmica calculada
        """
        # Prioridade base da tarefa
        prioridade_base = tarefa.prioridade
        
        # Peso do tipo de processo
        peso_tipo = tarefa.tipo_processo.value / 100.0  # Normalizado
        
        # Fator de contexto baseado no ambiente de condução
        fator_contexto = self.contexto.get_fator_ajuste(tarefa.tipo_processo)
        
        # Fator de urgência baseado na proximidade do deadline
        tempo_atual = time.time()
        tempo_decorrido = tempo_atual - tarefa.timestamp_chegada
        tempo_restante_deadline = max(1, tarefa.deadline - tempo_decorrido)
        fator_urgencia = 1.0 / max(1, tempo_restante_deadline / 10.0)  # Normalizado
        
        # Fator de aprendizado baseado no histórico de desempenho
        hash_tarefa = self._gerar_hash_tarefa(tarefa)
        fator_aprendizado = self.historico_desempenho.get(hash_tarefa, {}).get('fator_aprendizado', 1.0)
        
        # Fator de aging para prevenir starvation
        tempo_espera = tempo_atual - tarefa.timestamp_chegada
        fator_aging = 1.0 + (tempo_espera * 0.1)
        
        # Fator preditivo baseado no modelo ARTS-AD
        tempo_restante_previsto = self.prever_tempo_exec(tarefa)
        risco_deadline = max(1, tempo_restante_previsto / tempo_restante_deadline)
        fator_preditivo = 1.0 + math.log(risco_deadline + 1)  # Transformação log para suavizar
        
        # Fator anti-starvation baseado em preempções consecutivas
        if tarefa.preemptions_consecutivas > 3:
            fator_anti_starvation = 1.0 + (tarefa.preemptions_consecutivas - 3) * 0.3
        else:
            fator_anti_starvation = 1.0
        
        # Combinação multiplicativa de todos os fatores
        prioridade_dinamica = (
            prioridade_base * peso_tipo * fator_contexto * 
            fator_urgencia * fator_aprendizado * fator_aging * 
            fator_preditivo * fator_anti_starvation
        )
        
        # Debugging
        print(f"[PRIO] {tarefa.nome}: "
              f"base={prioridade_base}, tipo={peso_tipo:.2f}, "
              f"ctx={fator_contexto:.2f}, urg={fator_urgencia:.2f}, "
              f"apren={fator_aprendizado:.2f}, aging={fator_aging:.2f}, "
              f"pred={fator_preditivo:.2f}, anti-starv={fator_anti_starvation:.2f} "
              f"-> total={prioridade_dinamica:.2f}")
        
        return prioridade_dinamica
    
    def adicionar_tarefa_fila(self, tarefa: TarefaCAV):
        """Adiciona tarefa à fila apropriada com prioridade dinâmica"""
        prioridade_dinamica = self.calcular_prioridade_dinamica(tarefa)
        processo = ProcessoComPrioridade(tarefa, prioridade_dinamica)
        
        fila = self.filas_por_tipo[tarefa.tipo_processo]
        heapq.heappush(fila, processo)
        
        print(f"[HAPS] Adicionada: {tarefa.nome} ({tarefa.tipo_processo.name}) - Prioridade: {prioridade_dinamica:.2f}")
    
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
    
    def reavaliar_prioridades_filas(self):
        """
        Reavalia e atualiza prioridades de todas as tarefas nas filas de espera
        
        Processo:
        1. Esvazia cada fila temporariamente
        2. Para cada tarefa, recalcula a prioridade dinâmica
        3. Reinsere tarefas na fila com nova prioridade
        """
        print("\n[REAVALIAÇÃO] Atualizando prioridades nas filas...")
        inicio = time.time()
        
        for tipo, fila in self.filas_por_tipo.items():
            # Extrai todas as tarefas da fila
            tarefas_temp = []
            while fila:
                processo = heapq.heappop(fila)
                tarefas_temp.append(processo.tarefa)
            
            # Recalcula prioridade para cada tarefa
            processos_atualizados = []
            for tarefa in tarefas_temp:
                prioridade_atualizada = self.calcular_prioridade_dinamica(tarefa)
                processos_atualizados.append(ProcessoComPrioridade(tarefa, prioridade_atualizada))
            
            # Reinsere tarefas na fila
            for processo in processos_atualizados:
                heapq.heappush(fila, processo)
        
        duracao = time.time() - inicio
        self.registrar_sobrecarga(duracao)
        print(f"[REAVALIAÇÃO] Prioridades atualizadas em {duracao:.4f}s")
    
    def escalonar(self):
        """
        Executa o algoritmo de escalonamento HAPS-AI
        
        Características:
        - Escalonamento preemptivo baseado em quantum
        - Prioridade dinâmica recalculada a cada inserção
        - Seleção hierárquica por criticidade
        - Prevenção de starvation através de aging
        - Modelo preditivo ARTS-AD granular
        - Quantum adaptativo baseado em aprendizado
        - Reavaliação periódica de prioridades
        """
        print(f"\n=== INICIANDO ESCALONAMENTO HAPS-AI ===")
        print(f"Contexto: {self.contexto.clima.value}, {self.contexto.tipo_via.value}, "
              f"tráfego {self.contexto.trafego.value}, {self.contexto.velocidade_atual}km/h, "
              f"autônomo: {self.contexto.modo_autonomo}")
        
        # Inicializa filas com todas as tarefas
        for tarefa in self.tarefas:
            self.adicionar_tarefa_fila(tarefa)
        
        tempo_inicio_sistema = time.time()
        ciclo = 1
        
        # Loop principal de escalonamento
        while not self.todas_filas_vazias():
            print(f"\n--- CICLO {ciclo} ---")
            
            # Reavalia prioridades periodicamente
            if ciclo > 1 and ciclo % self.intervalo_reavaliacao == 0:
                self.reavaliar_prioridades_filas()
            
            # Seleciona próxima tarefa
            tarefa_atual = self.selecionar_proxima_tarefa()
            if tarefa_atual is None:
                break
            
            # Calcula quantum adaptativo
            quantum = self.calcular_quantum(tarefa_atual)
            
            print(f"[EXEC] Executando {tarefa_atual.nome} por {quantum} unidades "
                  f"(restante: {tarefa_atual.tempo_restante})")
            
            # Simula execução e obtém tempo real de execução
            tempo_real_executado = tarefa_atual.executar(quantum)
            time.sleep(0.1)  # Simulação com aceleração
            
            # Atualiza modelo preditivo com dados reais
            self.atualizar_modelo(tarefa_atual, tempo_real_executado)
            
            # Registra sobrecarga de troca de contexto
            self.registrar_sobrecarga(0.2)
            
            if tarefa_atual.tempo_restante <= 0:
                tempo_execucao_total = time.time() - tarefa_atual.timestamp_chegada
                deadline_perdido = tempo_execucao_total > tarefa_atual.deadline
                
                # Tarefa completada: resetar contador de preempções
                tarefa_atual.preemptions_consecutivas = 0
                
                print(f"[COMP] {tarefa_atual.nome} COMPLETADA!")
                if deadline_perdido:
                    print(f"[WARN] Deadline perdido: {tempo_execucao_total:.2f}s > {tarefa_atual.deadline}s")
                
                # Atualiza aprendizado da IA
                self._atualizar_aprendizado_ia(tarefa_atual, not deadline_perdido, tempo_execucao_total)
            else:
                # Tarefa preemptada: incrementar contador
                tarefa_atual.preemptions_consecutivas += 1
                print(f"[PREE] {tarefa_atual.nome} preemptada, recolocando na fila (preempções consecutivas: {tarefa_atual.preemptions_consecutivas})")
                # Recalcula prioridade e reinsere na fila
                self.adicionar_tarefa_fila(tarefa_atual)
            
            ciclo += 1
        
        tempo_total_sistema = time.time() - tempo_inicio_sistema
        print(f"\n=== ESCALONAMENTO CONCLUÍDO ===")
        print(f"Tempo total de sistema: {tempo_total_sistema:.2f}s")
        print(f"Ciclos executados: {ciclo - 1}")
        
        # Salva o estado de aprendizado da IA
        self._salvar_historico_ia()
        
        # Coleta métricas para logging
        self._coletar_metricas_execucao()
        
        # Exibe métricas finais
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


# ===== ESCALONADORES DE COMPARAÇÃO =====

class EscalonadorFIFO(EscalonadorCAV):
    """Escalonador First-In-First-Out simples para comparação"""
    
    def __init__(self):
        super().__init__()
        self.fila = deque()
    
    def escalonar(self):
        print("\n=== ESCALONAMENTO FIFO ===")
        self.fila.extend(self.tarefas)
        
        while self.fila:
            tarefa = self.fila.popleft()
            print(f"[FIFO] Executando {tarefa.nome}")
            
            while tarefa.tempo_restante > 0:
                tarefa.executar(1)
                time.sleep(0.05)
            
            self.registrar_sobrecarga(0.1)
        
        self.calcular_metricas()
        self.exibir_sobrecarga()


class EscalonadorRoundRobin(EscalonadorCAV):
    """Escalonador Round Robin com quantum fixo para comparação"""
    
    def __init__(self, quantum: int = 3):
        super().__init__()
        self.quantum = quantum
        self.fila = deque()
    
    def escalonar(self):
        print(f"\n=== ESCALONAMENTO ROUND ROBIN (Q={self.quantum}) ===")
        self.fila.extend(self.tarefas)
        
        while self.fila:
            tarefa = self.fila.popleft()
            print(f"[RR] Executando {tarefa.nome} (restante: {tarefa.tempo_restante})")
            
            completada = tarefa.executar(self.quantum)
            time.sleep(0.1)
            
            if not completada:
                self.fila.append(tarefa)
            
            self.registrar_sobrecarga(0.15)
        
        self.calcular_metricas()
        self.exibir_sobrecarga()


class EscalonadorPrioridade(EscalonadorCAV):
    """Escalonador por prioridade estática para comparação"""
    
    def __init__(self):
        super().__init__()
        self.fila = []
    
    def escalonar(self):
        print("\n=== ESCALONAMENTO POR PRIORIDADE ===")
        
        for tarefa in self.tarefas:
            # Prioridade combinada: tipo + prioridade base
            prioridade_total = tarefa.tipo_processo.value + tarefa.prioridade
            heapq.heappush(self.fila, (-prioridade_total, tarefa))  # Negativo para max-heap
        
        while self.fila:
            _, tarefa = heapq.heappop(self.fila)
            print(f"[PRIO] Executando {tarefa.nome} ({tarefa.tipo_processo.name})")
            
            while tarefa.tempo_restante > 0:
                tarefa.executar(2)
                time.sleep(0.05)
            
            self.registrar_sobrecarga(0.1)
        
        self.calcular_metricas()
        self.exibir_sobrecarga()


# ===== FUNÇÕES AUXILIARES =====

def criar_tarefas_cav() -> List[TarefaCAV]:
    """Cria conjunto diversificado de tarefas CAV para teste com deadlines variados"""
    # Tarefas críticas de segurança (deadlines apertados)
    tarefas = [
        TarefaCAV("Detecção_Obstáculo", random.randint(5, 10), 95, random.randint(8, 15), TipoProcessoCAV.SEGURANCA_CRITICA),
        TarefaCAV("Sistema_Frenagem", random.randint(4, 7), 98, random.randint(6, 12), TipoProcessoCAV.SEGURANCA_CRITICA),
        TarefaCAV("Controle_Estabilidade", random.randint(6, 9), 90, random.randint(10, 15), TipoProcessoCAV.SEGURANCA_CRITICA),
        
        # Tarefas de tempo real
        TarefaCAV("Controle_Motor", random.randint(8, 15), 85, random.randint(15, 25), TipoProcessoCAV.TEMPO_REAL),
        TarefaCAV("Sensor_LiDAR", random.randint(6, 12), 80, random.randint(12, 20), TipoProcessoCAV.TEMPO_REAL),
        TarefaCAV("Processamento_Camera", random.randint(10, 18), 75, random.randint(20, 30), TipoProcessoCAV.TEMPO_REAL),
        
        # Tarefas de navegação (algumas com deadlines apertados)
        TarefaCAV("GPS_Localização", random.randint(7, 14), 70, random.randint(20, 35), TipoProcessoCAV.NAVEGACAO),
        TarefaCAV("Planejamento_Rota", random.randint(12, 20), 65, random.randint(30, 45), TipoProcessoCAV.NAVEGACAO),
        TarefaCAV("Mapeamento_HD", random.randint(15, 25), 60, random.randint(40, 60), TipoProcessoCAV.NAVEGACAO),
        
        # Tarefas de conforto (algumas com deadlines apertados)
        TarefaCAV("Controle_Climatização", random.randint(5, 10), 50, random.randint(40, 60), TipoProcessoCAV.CONFORTO),
        TarefaCAV("Sistema_Audio", random.randint(4, 8), 45, random.randint(50, 80), TipoProcessoCAV.CONFORTO),
        TarefaCAV("Interface_Usuario", random.randint(8, 15), 40, random.randint(45, 70), TipoProcessoCAV.CONFORTO),
        
        # Tarefas de diagnóstico (algumas com deadlines apertados)
        TarefaCAV("Monitoramento_Sistema", random.randint(10, 18), 30, random.randint(80, 120), TipoProcessoCAV.DIAGNOSTICO),
        TarefaCAV("Log_Eventos", random.randint(4, 8), 25, random.randint(90, 150), TipoProcessoCAV.DIAGNOSTICO),
        TarefaCAV("Telemetria", random.randint(7, 14), 20, random.randint(100, 180), TipoProcessoCAV.DIAGNOSTICO),
        
        # Tarefas adicionais para estresse do sistema
        TarefaCAV("Emergência_Colisão", 3, 100, 5, TipoProcessoCAV.SEGURANCA_CRITICA),
        TarefaCAV("Diagnóstico_Rápido", 4, 35, 8, TipoProcessoCAV.DIAGNOSTICO),
        TarefaCAV("Atualização_Mapa", 6, 55, 10, TipoProcessoCAV.NAVEGACAO),
    ]
    
    return tarefas

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
    tarefas_base = criar_tarefas_cav()
    
    # Lista de escalonadores para comparar
    escalonadores = [
        ("HAPS-AI", EscalonadorHAPS(contexto_teste)),
        ("FIFO", EscalonadorFIFO()),
        ("Round Robin", EscalonadorRoundRobin(4)),
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
    tarefas_teste = criar_tarefas_cav()
    
    # Instancia e executa o escalonador HAPS-AI
    #escalonador_haps = EscalonadorHAPS(contexto_adverso)
    #for tarefa in tarefas_teste:
    #    escalonador_haps.adicionar_tarefa_fila(tarefa)
    
    #escalonador_haps.escalonar()

    resultado = rodar_varias_simulacoes(10)
