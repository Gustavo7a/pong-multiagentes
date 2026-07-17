"""Fábrica e utilitários do ambiente Pong multiagente (PettingZoo / Atari).

Contexto II do estudo dirigido: ambiente multiagente Atari do PettingZoo,
configurado para fornecer como observação apenas a RAM do console
(``obs_type="ram"``), ou seja, um vetor com 128 bytes (inteiros de 0 a 255).

O Pong do PettingZoo é competitivo entre dois agentes:
    - "first_0"  -> raquete da esquerda
    - "second_0" -> raquete da direita

Cada agente recebe, a cada passo, o mesmo vetor de 128 bytes da RAM e escolhe
uma ação discreta. A recompensa é +1 quando marca ponto e -1 quando sofre ponto
(soma zero entre os dois jogadores).
"""

from __future__ import annotations

from pettingzoo.atari import pong_v3

# Nomes dos dois agentes no ambiente Pong do PettingZoo.
AGENTE_ESQUERDA = "first_0"
AGENTE_DIREITA = "second_0"


def criar_ambiente(
    render_mode: str | None = None,
    max_cycles: int = 125_000,
    full_action_space: bool = False,
):
    """Cria o ambiente Pong no modo *parallel* com observação de RAM.

    Parâmetros
    ----------
    render_mode:
        ``"human"`` abre janela, ``"rgb_array"`` devolve frames, ``None`` roda
        sem renderização (mais rápido para treino/avaliação).
    max_cycles:
        Número máximo de passos antes de encerrar o episódio.
    full_action_space:
        Se ``False`` usa o conjunto mínimo de ações do Pong; se ``True`` usa as
        18 ações do Atari. Mantido ``False`` por padrão para simplificar os
        agentes.

    Retorna
    -------
    Ambiente ``ParallelEnv`` do PettingZoo já configurado.
    """
    return pong_v3.parallel_env(
        obs_type="ram",
        render_mode=render_mode,
        max_cycles=max_cycles,
        full_action_space=full_action_space,
    )


def descrever_ambiente() -> None:
    """Imprime espaços de observação/ação de cada agente (útil para depurar)."""
    env = criar_ambiente()
    env.reset(seed=0)
    print("Agentes:", env.agents)
    for agente in env.agents:
        print(f"  {agente}:")
        print(f"    observação: {env.observation_space(agente)}")
        print(f"    ação:       {env.action_space(agente)}")
    env.close()


if __name__ == "__main__":
    descrever_ambiente()
