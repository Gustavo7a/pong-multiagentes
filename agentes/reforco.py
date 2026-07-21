"""Agente por aprendizado por reforco, com Q-learning tabular.

A ideia aqui e aprender direto durante as partidas: a cada transicao, o agente
atualiza uma tabela Q com base na recompensa observada e no melhor valor do
proximo estado. Como o Pong cabe mal em uma tabela gigante, o estado eh
discretizado em poucos bins extraidos da RAM.

O agente aprende online enquanto joga. Se a instancia for reutilizada em varios
episodios, o conhecimento acumulado continua valendo.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .base import Agente
from .ram import BOLA_Y_MAXIMO, BOLA_Y_MINIMO, EstadoPong, decodificar

# Conjunto minimo de acoes do Pong usado pelos agentes do projeto.
NOOP = 0
FIRE = 1
CIMA = 2
BAIXO = 3

CENTRO_TELA = (BOLA_Y_MINIMO + BOLA_Y_MAXIMO) / 2
ALTURA_TELA = BOLA_Y_MAXIMO - BOLA_Y_MINIMO
ARQUIVO_POLITICA = Path(__file__).with_name("politica_reforco.npz")


class AgenteReforco(Agente):
    """Aprende uma politica por Q-learning tabular."""

    nome = "reforco"

    def __init__(
        self,
        lado: str,
        n_acoes: int,
        seed: int | None = None,
        arquivo: Path | str | None = None,
        carregar: bool = True,
        taxa_aprendizado: float = 0.12,
        desconto: float = 0.95,
        epsilon: float = 0.15,
        epsilon_min: float = 0.02,
        decaimento_epsilon: float = 0.997,
    ):
        if lado not in ("esquerda", "direita"):
            raise ValueError("lado deve ser 'esquerda' ou 'direita'")
        self.lado = lado
        self.nome = f"reforco_{lado}"
        self.n_acoes = n_acoes
        self.taxa_aprendizado = taxa_aprendizado
        self.desconto = desconto
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.decaimento_epsilon = decaimento_epsilon
        self._rng = np.random.default_rng(seed)

        if carregar:
            try:
                self._q = carregar_politica(arquivo)
            except FileNotFoundError:
                self._q = {}
        else:
            self._q = {}
        self._estado_anterior: EstadoPong | None = None
        self._chave_anterior: tuple[int, ...] | None = None
        self._acao_anterior: int | None = None
        self._recompensa_pendente = 0.0
        self._tem_recompensa_pendente = False
        self._ultima_acao = NOOP

    def reiniciar(self) -> None:
        self._estado_anterior = None
        self._chave_anterior = None
        self._acao_anterior = None
        self._recompensa_pendente = 0.0
        self._tem_recompensa_pendente = False
        self._ultima_acao = NOOP

    def agir(self, observacao: np.ndarray) -> int:
        estado = decodificar(observacao)

        if (
            self._tem_recompensa_pendente
            and self._estado_anterior is not None
            and self._chave_anterior is not None
            and self._acao_anterior is not None
        ):
            self._aprender(
                self._chave_anterior,
                self._acao_anterior,
                self._recompensa_pendente,
                self._estado_anterior,
                estado,
            )
            self._tem_recompensa_pendente = False

        if not estado.bola_em_jogo:
            # Fora de jogo, a acao serve so para o saque. Nao vale a pena
            # aprender sobre esse estado, entao nao o mantemos como transicao.
            self._estado_anterior = None
            self._chave_anterior = None
            self._acao_anterior = None
            acao = NOOP if self._ultima_acao == FIRE else FIRE
            self._ultima_acao = acao
            return acao

        chave = self._codificar_estado(estado, self._estado_anterior)
        if self._rng.random() < self.epsilon:
            acao = int(self._rng.integers(self.n_acoes))
        else:
            acao = self._escolher_melhor_acao(chave)

        self._estado_anterior = estado
        self._chave_anterior = chave
        self._acao_anterior = acao
        self._ultima_acao = acao
        return acao

    def observar(self, recompensa: float, terminou: bool) -> None:
        if (
            self._estado_anterior is None
            or self._chave_anterior is None
            or self._acao_anterior is None
        ):
            return

        if terminou:
            self._aprender(
                self._chave_anterior,
                self._acao_anterior,
                recompensa,
                None,
                None,
            )
            self._estado_anterior = None
            self._chave_anterior = None
            self._acao_anterior = None
            self._tem_recompensa_pendente = False
            self._ultima_acao = NOOP
            self.epsilon = max(self.epsilon_min, self.epsilon * self.decaimento_epsilon)
            return

        self._recompensa_pendente = recompensa
        self._tem_recompensa_pendente = True

    def _aprender(
        self,
        chave: tuple[int, ...],
        acao: int,
        recompensa: float,
        contexto: EstadoPong | None,
        proximo_estado: EstadoPong | None,
    ) -> None:
        valores = self._q.setdefault(chave, np.zeros(self.n_acoes, dtype=np.float64))

        if proximo_estado is None:
            alvo = recompensa
        else:
            proxima_chave = self._codificar_estado(proximo_estado, contexto)
            proximos_valores = self._q.setdefault(
                proxima_chave, np.zeros(self.n_acoes, dtype=np.float64)
            )
            alvo = recompensa + self.desconto * float(np.max(proximos_valores))

        valores[acao] += self.taxa_aprendizado * (alvo - valores[acao])

    def _escolher_melhor_acao(self, chave: tuple[int, ...]) -> int:
        valores = self._q.setdefault(chave, np.zeros(self.n_acoes, dtype=np.float64))
        melhor_valor = float(np.max(valores))
        melhores = np.flatnonzero(valores == melhor_valor)
        return int(self._rng.choice(melhores))

    def _codificar_estado(
        self, estado: EstadoPong, anterior: EstadoPong | None
    ) -> tuple[int, ...]:
        bola_x_bin = self._bin(estado.bola_x, 16, 0.0, 255.0)
        bola_y_bin = self._bin(estado.bola_y, 12, BOLA_Y_MINIMO, BOLA_Y_MAXIMO)
        raquete_y_bin = self._bin(
            estado.centro_raquete(self.lado), 12, BOLA_Y_MINIMO, BOLA_Y_MAXIMO
        )
        distancia_x_bin = self._bin(
            abs(estado.x_raquete(self.lado) - estado.bola_x), 8, 0.0, 255.0
        )

        if anterior is None:
            delta_x = 0
            delta_y = 0
        else:
            delta_x = self._sinal(estado.bola_x - anterior.bola_x)
            delta_y = self._sinal(estado.bola_y - anterior.bola_y)

        if self.lado == "esquerda":
            vindo = int(delta_x < 0)
        else:
            vindo = int(delta_x > 0)

        # Normaliza a posicao da raquete para perto do centro da tela, o que
        # ajuda a tabela a generalizar um pouco melhor.
        raquete_relativa = self._bin(
            estado.centro_raquete(self.lado) - CENTRO_TELA,
            12,
            -ALTURA_TELA / 2,
            ALTURA_TELA / 2,
        )

        return (
            bola_x_bin,
            bola_y_bin,
            raquete_y_bin,
            raquete_relativa,
            distancia_x_bin,
            delta_x,
            delta_y,
            vindo,
        )

    @staticmethod
    def _bin(valor: float, bins: int, minimo: float, maximo: float) -> int:
        if maximo <= minimo:
            return 0
        valor = min(max(valor, minimo), maximo)
        frac = (valor - minimo) / (maximo - minimo)
        return min(bins - 1, int(frac * bins))

    @staticmethod
    def _sinal(valor: float) -> int:
        if valor > 0:
            return 1
        if valor < 0:
            return -1
        return 0


def carregar_politica(arquivo: Path | str | None = None) -> dict[tuple[int, ...], np.ndarray]:
    caminho = Path(arquivo) if arquivo else ARQUIVO_POLITICA
    if not caminho.exists():
        raise FileNotFoundError(
            f"politica do agente de reforco nao encontrada em {caminho}. "
            "Rode: python treinar_reforco.py"
        )

    with np.load(caminho, allow_pickle=True) as dados:
        estados = dados["estados"]
        valores = dados["valores"]

    politica: dict[tuple[int, ...], np.ndarray] = {}
    if len(estados) != len(valores):
        raise ValueError("politica corrompida: chaves e valores com tamanhos diferentes")

    for estado, valor in zip(estados, valores):
        chave = tuple(int(item) for item in np.asarray(estado).tolist())
        politica[chave] = np.asarray(valor, dtype=np.float64)
    return politica


def salvar_politica(
    politica: dict[tuple[int, ...], np.ndarray],
    arquivo: Path | str | None = None,
    **extras,
) -> Path:
    caminho = Path(arquivo) if arquivo else ARQUIVO_POLITICA
    estados = np.array(list(politica.keys()), dtype=object)
    valores = np.array(
        [np.asarray(valor, dtype=np.float64) for valor in politica.values()],
        dtype=object,
    )
    np.savez(caminho, estados=estados, valores=valores, **extras)
    return caminho