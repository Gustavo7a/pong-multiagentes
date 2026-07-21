"""Torneio todo-contra-todo entre os agentes implementados.

Comparar cada agente só contra o aleatório diz pouco, porque qualquer um que
rastreie a bola ganha de lavada. Aqui todos se enfrentam par a par e o aleatório
fica só como piso de referência.

Cada confronto roda nos dois lados, com o mesmo número de partidas em cada
raquete, porque quem saca e quem recebe não jogam a mesma partida.

Uso:
    python torneio.py                                    # aleatório, heurístico, genético
    python torneio.py --episodios 5
    python torneio.py --sem-baseline                     # tira o aleatório
    python torneio.py --tipos heuristico heuristico_preditivo   # variantes
"""

from __future__ import annotations

import argparse
from itertools import combinations

from agentes import BASELINES, TIPOS, TODOS, criar_agente
from ambiente_pong import (
    AGENTE_DIREITA,
    AGENTE_ESQUERDA,
    MAX_CICLOS,
    criar_ambiente,
)
from jogar import jogar_episodio

# Pontuação no estilo tabela de campeonato.
PONTOS_VITORIA = 3
PONTOS_EMPATE = 1


class Placar:
    """Acumula o desempenho de um agente ao longo do torneio."""

    def __init__(self, tipo: str):
        self.tipo = tipo
        self.vitorias = 0
        self.empates = 0
        self.derrotas = 0
        self.saldo = 0.0

    @property
    def pontos(self) -> int:
        return self.vitorias * PONTOS_VITORIA + self.empates * PONTOS_EMPATE

    @property
    def partidas(self) -> int:
        return self.vitorias + self.empates + self.derrotas

    def registrar(self, saldo: float) -> None:
        self.saldo += saldo
        if saldo > 0:
            self.vitorias += 1
        elif saldo < 0:
            self.derrotas += 1
        else:
            self.empates += 1


def disputar(
    env,
    tipo_esquerda: str,
    tipo_direita: str,
    n_acoes: int,
    episodios: int,
    seed: int,
) -> list[float]:
    """Joga as partidas e devolve o saldo do agente da esquerda em cada uma."""
    saldos = []
    for ep in range(episodios):
        semente = seed + ep
        # Agentes novos a cada partida, para nenhum herdar estado da anterior.
        agentes = {
            AGENTE_ESQUERDA: criar_agente(
                tipo_esquerda, "esquerda", n_acoes, seed=semente
            ),
            AGENTE_DIREITA: criar_agente(
                tipo_direita, "direita", n_acoes, seed=semente + 500
            ),
        }
        total = jogar_episodio(env, agentes, seed=semente)
        # Como o jogo é soma zero, o saldo do oponente é o simétrico deste.
        saldos.append(total.get(AGENTE_ESQUERDA, 0.0))
    return saldos


def main() -> None:
    parser = argparse.ArgumentParser(description="Torneio todo-contra-todo")
    parser.add_argument(
        "--tipos", nargs="+", choices=TODOS, default=list(TIPOS),
        help="quais agentes entram no torneio (padrão: os do estudo)",
    )
    parser.add_argument(
        "--episodios", type=int, default=3, help="partidas por lado em cada confronto"
    )
    parser.add_argument("--max-ciclos", type=int, default=MAX_CICLOS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--sem-baseline", action="store_true", help="exclui o aleatório"
    )
    args = parser.parse_args()

    tipos = [t for t in args.tipos if not (args.sem_baseline and t in BASELINES)]
    if len(tipos) < 2:
        parser.error("é preciso pelo menos dois agentes no torneio")

    env = criar_ambiente(max_cycles=args.max_ciclos)
    env.reset(seed=args.seed)
    n_acoes = env.action_space(AGENTE_ESQUERDA).n

    placares = {tipo: Placar(tipo) for tipo in tipos}
    confrontos: dict[tuple[str, str], float] = {}

    print(f"Torneio: {', '.join(tipos)}")
    print(f"{args.episodios} partida(s) por lado, limite de {args.max_ciclos} ciclos\n")

    for a, b in combinations(tipos, 2):
        seed = args.seed
        # O mesmo confronto com os lados trocados, para anular vantagem de lado.
        saldos_a = disputar(env, a, b, n_acoes, args.episodios, seed)
        saldos_b = disputar(env, b, a, n_acoes, args.episodios, seed + 100)

        for saldo in saldos_a:
            placares[a].registrar(saldo)
            placares[b].registrar(-saldo)
        for saldo in saldos_b:
            placares[b].registrar(saldo)
            placares[a].registrar(-saldo)

        como_esquerda = sum(saldos_a)
        como_direita = -sum(saldos_b)
        saldo_a = como_esquerda + como_direita
        confrontos[(a, b)] = saldo_a
        confrontos[(b, a)] = -saldo_a

        print(
            f"{a:<20} x {b:<20} "
            f"saldo de {a}: {saldo_a:+4.0f}  "
            f"(jogando à esquerda {como_esquerda:+.0f}, "
            f"à direita {como_direita:+.0f})"
        )

    env.close()

    print("\nClassificação (V=3, E=1):\n")
    cabecalho = f"  {'agente':<22}{'P':>4}{'V':>4}{'E':>4}{'D':>4}{'saldo':>8}"
    print(cabecalho)
    print("  " + "-" * (len(cabecalho) - 2))
    for placar in sorted(
        placares.values(), key=lambda p: (p.pontos, p.saldo), reverse=True
    ):
        print(
            f"  {placar.tipo:<22}{placar.pontos:>4}{placar.vitorias:>4}"
            f"{placar.empates:>4}{placar.derrotas:>4}{placar.saldo:>+8.0f}"
        )

    print("\nSaldo por confronto (linha contra coluna):\n")
    largura = max(len(t) for t in tipos) + 2
    print(" " * (largura + 2) + "".join(f"{t[:10]:>12}" for t in tipos))
    for a in tipos:
        celulas = "".join(
            f"{'.':>12}" if a == b else f"{confrontos[(a, b)]:>+12.0f}"
            for b in tipos
        )
        print(f"  {a:<{largura}}{celulas}")


if __name__ == "__main__":
    main()
