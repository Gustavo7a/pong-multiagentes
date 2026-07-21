"""Agente heurístico: regras fixas sobre o estado lido da RAM.

Alinha a raquete com a altura atual da bola e para de corrigir quando já está
perto o bastante, para não tremer em torno do alvo.

O sentido do eixo vertical não é assumido. Nos primeiros passos o agente testa
se a ação CIMA aumenta ou diminui o byte da sua raquete, então o mesmo código
funciona nos dois lados.

Sobre a variante prever=True: ela antecipa onde a bola vai cruzar a linha da
raquete, extrapolando a velocidade e simulando as ricochetes. Parece melhor,
mas perde por +44 de saldo para a versão simples. Extrapolar de dois frames erra
a estimativa e a raquete sai cedo para o lugar errado, enquanto a versão simples
chega a tempo do mesmo jeito. Ficou só para reproduzir a comparação:

    python torneio.py --tipos heuristico heuristico_preditivo
"""

from __future__ import annotations

import numpy as np

from .base import Agente
from .ram import BOLA_Y_MAXIMO, BOLA_Y_MINIMO, EstadoPong, decodificar

# Conjunto mínimo de ações do Pong.
NOOP = 0
FIRE = 1
CIMA = 2
BAIXO = 3

CENTRO = (BOLA_Y_MINIMO + BOLA_Y_MAXIMO) / 2


class AgenteHeuristico(Agente):
    """Rastreia a bola por regras fixas."""

    nome = "heuristico"

    def __init__(
        self,
        lado: str,
        zona_morta: float = 3.0,
        prever: bool = False,
    ):
        """zona_morta é a folga em pixels que o agente aceita sem corrigir."""
        if lado not in ("esquerda", "direita"):
            raise ValueError("lado deve ser 'esquerda' ou 'direita'")
        self.lado = lado
        self.zona_morta = zona_morta
        self.prever = prever
        self.nome = f"heuristico{'_preditivo' if prever else ''}_{lado}"

        self._anterior: EstadoPong | None = None
        # +1 se a ação CIMA aumenta o byte da raquete, -1 se diminui.
        self._sentido: int | None = None
        self._votos_sentido = 0
        self._ultima_acao = NOOP

    def reiniciar(self) -> None:
        self._anterior = None
        self._ultima_acao = NOOP
        # A calibração do eixo vale para o episódio seguinte também.

    def agir(self, observacao: np.ndarray) -> int:
        estado = decodificar(observacao)
        self._calibrar_sentido(estado)

        if not estado.bola_em_jogo:
            # O jogo espera o saque, e o botão é sensível à borda: segurar FIRE
            # não dispara de novo. Alternar FIRE e NOOP simula apertar e soltar.
            # Sem isso a partida trava e o ALE cobra a penalidade de saque.
            self._anterior = None
            acao = NOOP if self._ultima_acao == FIRE else FIRE
            self._ultima_acao = acao
            return acao

        if self._sentido is None:
            # Ainda calibrando: move e observa o efeito no byte da raquete.
            acao = CIMA
        else:
            acao = self._mover_para(estado, self._alvo(estado))

        self._anterior = estado
        self._ultima_acao = acao
        return acao

    def _alvo(self, estado: EstadoPong) -> float:
        """Altura que a raquete deveria ocupar neste passo."""
        if not estado.bola_em_jogo:
            return CENTRO

        if not self.prever or self._anterior is None:
            return estado.bola_y

        dx = estado.bola_x - self._anterior.bola_x
        dy = estado.bola_y - self._anterior.bola_y

        if dx == 0:
            return estado.bola_y

        vindo = dx < 0 if self.lado == "esquerda" else dx > 0
        if not vindo:
            return CENTRO

        return self._prever_impacto(estado, dx, dy)

    def _prever_impacto(self, estado: EstadoPong, dx: float, dy: float) -> float:
        distancia = abs(estado.x_raquete(self.lado) - estado.bola_x)
        y_livre = estado.bola_y + dy * (distancia / abs(dx))
        return self._refletir(y_livre)

    @staticmethod
    def _refletir(y: float) -> float:
        """Dobra a altura prevista de volta para dentro da tela.

        Uma trajetória com ricochetes é a mesma coisa que uma reta num espaço
        espelhado várias vezes, então basta dobrar o valor para dentro dos
        limites. O resultado é a onda triangular dessa reta.
        """
        altura = BOLA_Y_MAXIMO - BOLA_Y_MINIMO
        posicao = (y - BOLA_Y_MINIMO) % (2 * altura)
        if posicao > altura:
            posicao = 2 * altura - posicao
        return BOLA_Y_MINIMO + posicao

    def _mover_para(self, estado: EstadoPong, alvo: float) -> int:
        erro = alvo - estado.centro_raquete(self.lado)
        if abs(erro) <= self.zona_morta:
            return NOOP
        # Erro positivo quer dizer alvo mais embaixo, ou seja, byte maior.
        precisa_aumentar = erro > 0
        sobe_com_cima = self._sentido == 1
        return CIMA if precisa_aumentar == sobe_com_cima else BAIXO

    def _calibrar_sentido(self, estado: EstadoPong) -> None:
        """Vota, passo a passo, se CIMA aumenta ou diminui o byte da raquete.

        Cada passo em que a raquete se moveu depois de um comando vale um voto.
        Três votos na mesma direção fecham a questão.
        """
        if self._sentido is not None or self._anterior is None:
            return
        if self._ultima_acao not in (CIMA, BAIXO):
            return

        y_agora = (
            estado.raquete_esquerda_y
            if self.lado == "esquerda"
            else estado.raquete_direita_y
        )
        y_antes = (
            self._anterior.raquete_esquerda_y
            if self.lado == "esquerda"
            else self._anterior.raquete_direita_y
        )
        delta = y_agora - y_antes
        if delta == 0:
            return

        efeito = delta if self._ultima_acao == CIMA else -delta
        self._votos_sentido += 1 if efeito > 0 else -1
        if abs(self._votos_sentido) >= 3:
            self._sentido = 1 if self._votos_sentido > 0 else -1
