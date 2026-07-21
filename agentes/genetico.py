"""Agente genético: rede pequena com pesos evoluídos, não treinados.

Diferença para o heurístico é que aqui ninguém escreve a regra de decisão. O
agente é uma rede de 6 entradas, 4 neurônios ocultos e 3 ações, e os pesos saem
de um algoritmo genético que só olha o placar do episódio. Sem gradiente e sem
recompensa passo a passo.

Entradas, todas normalizadas para ficar perto de [-1, 1]: altura da bola, altura
do centro da própria raquete, distância horizontal até ela, velocidade vertical
da bola, velocidade horizontal já orientada (positiva quando a bola vem) e um
termo constante. As saídas são pontuações para parado, sobe e desce; ganha a
maior.

O treino está em treinar_genetico.py e salva os pesos em pesos_genetico.npz.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .base import Agente
from .ram import (
    BOLA_Y_MAXIMO,
    BOLA_Y_MINIMO,
    EstadoPong,
    decodificar,
)

# Conjunto mínimo de ações do Pong.
NOOP = 0
FIRE = 1
CIMA = 2
BAIXO = 3

# A rede escolhe entre parar, subir e descer; o saque é tratado à parte.
ACOES = (NOOP, CIMA, BAIXO)

N_ENTRADAS = 6
N_OCULTOS = 4
N_SAIDAS = len(ACOES)

# Tamanho do vetor de genes: duas camadas com peso e viés.
N_GENES = (
    N_ENTRADAS * N_OCULTOS + N_OCULTOS + N_OCULTOS * N_SAIDAS + N_SAIDAS
)

ARQUIVO_PESOS = Path(__file__).with_name("pesos_genetico.npz")

# Escalas usadas para normalizar as entradas.
ALTURA_TELA = BOLA_Y_MAXIMO - BOLA_Y_MINIMO
CENTRO_TELA = (BOLA_Y_MINIMO + BOLA_Y_MAXIMO) / 2
LARGURA_CAMPO = 128.0  # distância entre as duas raquetes, medida na RAM
VELOCIDADE_TIPICA = 8.0


def genes_aleatorios(rng: np.random.Generator, escala: float = 0.5) -> np.ndarray:
    """Sorteia um indivíduo inicial."""
    return rng.normal(0.0, escala, size=N_GENES)


def _desempacotar(genes: np.ndarray):
    """Fatia o vetor de genes nas matrizes da rede."""
    i = 0
    w1 = genes[i : i + N_ENTRADAS * N_OCULTOS].reshape(N_ENTRADAS, N_OCULTOS)
    i += N_ENTRADAS * N_OCULTOS
    b1 = genes[i : i + N_OCULTOS]
    i += N_OCULTOS
    w2 = genes[i : i + N_OCULTOS * N_SAIDAS].reshape(N_OCULTOS, N_SAIDAS)
    i += N_OCULTOS * N_SAIDAS
    b2 = genes[i : i + N_SAIDAS]
    return w1, b1, w2, b2


class AgenteGenetico(Agente):
    """Rede minúscula com pesos vindos do algoritmo genético."""

    nome = "genetico"

    def __init__(
        self,
        lado: str,
        genes: np.ndarray | None = None,
        arquivo: Path | str | None = None,
    ):
        """Sem genes, carrega os pesos do arquivo salvo pelo treino."""
        if lado not in ("esquerda", "direita"):
            raise ValueError("lado deve ser 'esquerda' ou 'direita'")
        self.lado = lado
        self.nome = f"genetico_{lado}"

        if genes is None:
            genes = carregar_genes(arquivo)
        genes = np.asarray(genes, dtype=np.float64)
        if genes.shape != (N_GENES,):
            raise ValueError(
                f"esperado vetor de {N_GENES} genes, veio {genes.shape}"
            )
        self.genes = genes
        self._rede = _desempacotar(genes)

        self._anterior: EstadoPong | None = None
        self._ultima_acao = NOOP

    def reiniciar(self) -> None:
        self._anterior = None
        self._ultima_acao = NOOP

    def agir(self, observacao: np.ndarray) -> int:
        estado = decodificar(observacao)

        if not estado.bola_em_jogo:
            # Mesmo detalhe do heurístico: o botão de saque é sensível à borda,
            # então alternamos FIRE e parado. Isso fica fora do que evolui, senão
            # a partida trava e nenhum indivíduo consegue pontuar.
            self._anterior = None
            acao = NOOP if self._ultima_acao == FIRE else FIRE
            self._ultima_acao = acao
            return acao

        entradas = self._caracteristicas(estado)
        w1, b1, w2, b2 = self._rede
        oculta = np.tanh(entradas @ w1 + b1)
        pontuacoes = oculta @ w2 + b2
        acao = ACOES[int(np.argmax(pontuacoes))]

        self._anterior = estado
        self._ultima_acao = acao
        return acao

    def _caracteristicas(self, estado: EstadoPong) -> np.ndarray:
        if self._anterior is None or not self._anterior.bola_em_jogo:
            dy = 0.0
            dx = 0.0
        else:
            dy = estado.bola_y - self._anterior.bola_y
            dx = estado.bola_x - self._anterior.bola_x

        # Fica positivo quando a bola se aproxima, valendo para os dois lados.
        aproximando = -dx if self.lado == "esquerda" else dx
        distancia = abs(estado.x_raquete(self.lado) - estado.bola_x)

        return np.array(
            [
                (estado.bola_y - CENTRO_TELA) / ALTURA_TELA,
                (estado.centro_raquete(self.lado) - CENTRO_TELA) / ALTURA_TELA,
                distancia / LARGURA_CAMPO,
                dy / VELOCIDADE_TIPICA,
                aproximando / VELOCIDADE_TIPICA,
                1.0,
            ],
            dtype=np.float64,
        )


def carregar_genes(arquivo: Path | str | None = None) -> np.ndarray:
    caminho = Path(arquivo) if arquivo else ARQUIVO_PESOS
    if not caminho.exists():
        raise FileNotFoundError(
            f"pesos do agente genético não encontrados em {caminho}. "
            "Rode: python treinar_genetico.py"
        )
    with np.load(caminho) as dados:
        return dados["genes"]


def salvar_genes(genes: np.ndarray, arquivo: Path | str | None = None, **extras) -> Path:
    """Grava os pesos campeões junto com os metadados do treino."""
    caminho = Path(arquivo) if arquivo else ARQUIVO_PESOS
    np.savez(caminho, genes=np.asarray(genes, dtype=np.float64), **extras)
    return caminho
