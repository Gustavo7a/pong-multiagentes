"""Treina o agente de reforco por Q-learning online.

O agente aprende enquanto joga partidas completas contra um oponente fixo. A
cada episodio, as recompensas observadas atualizam a tabela Q e o valor de
epsilon cai aos poucos, para equilibrar exploracao e exploracao no inicio e
explotacao no fim.

Uso:
    python treinar_reforco.py
    python treinar_reforco.py --episodios 500 --adversario heuristico
    python treinar_reforco.py --lado direita --saida agentes/politica_reforco_direita.npz
    python treinar_reforco.py --inicial agentes/politica_reforco.npz
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from agentes.fabrica import criar_agente
from agentes.reforco import ARQUIVO_POLITICA, AgenteReforco, salvar_politica
from ambiente_pong import AGENTE_DIREITA, AGENTE_ESQUERDA, MAX_CICLOS, criar_ambiente
from jogar import jogar_episodio


def main() -> None:
    parser = argparse.ArgumentParser(description="Treino do agente de reforco")
    parser.add_argument("--episodios", type=int, default=200)
    parser.add_argument("--lado", choices=("esquerda", "direita"), default="esquerda")
    parser.add_argument("--adversario", default="aleatorio")
    parser.add_argument("--max-ciclos", type=int, default=MAX_CICLOS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--saida", default=str(ARQUIVO_POLITICA))
    parser.add_argument(
        "--inicial",
        default=None,
        help="continua de uma politica salva anteriormente",
    )
    parser.add_argument("--taxa-aprendizado", type=float, default=0.12)
    parser.add_argument("--desconto", type=float, default=0.95)
    parser.add_argument("--epsilon", type=float, default=0.20)
    parser.add_argument("--epsilon-min", type=float, default=0.02)
    parser.add_argument("--decaimento-epsilon", type=float, default=0.997)
    args = parser.parse_args()

    if args.inicial and not Path(args.inicial).exists():
        parser.error(f"arquivo inicial nao encontrado: {args.inicial}")

    env = criar_ambiente(max_cycles=args.max_ciclos)
    env.reset(seed=args.seed)
    n_acoes = env.action_space(AGENTE_ESQUERDA).n

    agente = AgenteReforco(
        lado=args.lado,
        n_acoes=n_acoes,
        seed=args.seed,
        arquivo=args.inicial,
        carregar=bool(args.inicial),
        taxa_aprendizado=args.taxa_aprendizado,
        desconto=args.desconto,
        epsilon=args.epsilon,
        epsilon_min=args.epsilon_min,
        decaimento_epsilon=args.decaimento_epsilon,
    )

    lado_oponente = "direita" if args.lado == "esquerda" else "esquerda"
    nome_proprio = AGENTE_ESQUERDA if args.lado == "esquerda" else AGENTE_DIREITA
    nome_oponente = AGENTE_DIREITA if args.lado == "esquerda" else AGENTE_ESQUERDA

    print(
        f"Treino por reforco: lado {args.lado} contra {args.adversario} | "
        f"episodios: {args.episodios} | max ciclos: {args.max_ciclos}"
    )
    if args.inicial:
        print(f"Continuando de: {args.inicial}")
    print()

    historico = []
    inicio = time.time()

    for episodio in range(args.episodios):
        semente = args.seed + episodio
        oponente = criar_agente(args.adversario, lado_oponente, n_acoes, seed=semente)
        agentes = {
            nome_proprio: agente,
            nome_oponente: oponente,
        }

        total = jogar_episodio(env, agentes, seed=semente)
        saldo = float(total.get(nome_proprio, 0.0))
        historico.append(saldo)

        janela = historico[-20:]
        media = float(np.mean(janela)) if janela else 0.0
        print(
            f"Episodio {episodio + 1:>4}/{args.episodios}: "
            f"saldo {saldo:+6.0f}   media 20 {media:+6.2f}   "
            f"epsilon {agente.epsilon:5.3f}   "
            f"({time.time() - inicio:5.0f}s)"
        )

    env.close()

    caminho = salvar_politica(
        agente._q,
        args.saida,
        historico=np.array(historico),
        lado=args.lado,
        adversario=args.adversario,
        episodios=args.episodios,
        epsilon_final=agente.epsilon,
        taxa_aprendizado=args.taxa_aprendizado,
        desconto=args.desconto,
    )
    print(f"\nEstados aprendidos: {len(agente._q)}")
    print(f"Politica salva em {caminho}")


if __name__ == "__main__":
    main()