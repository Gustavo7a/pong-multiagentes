"""Descobre na marra quais bytes da RAM guardam a bola e as raquetes.

Em vez de confiar em tabela de terceiros, o script mexe no jogo e observa o que
acontece com a memória. São cinco testes:

1. Controle: move uma raquete de cada vez e vê qual byte acompanha o comando.
2. Variação: joga aleatório e lista os bytes que mudam mais, onde a bola aparece.
3. Faixas: mostra os valores mínimo e máximo dos bytes que o projeto usa.
4. Eixo: segura CIMA e BAIXO até travar, para achar os limites da raquete.
5. Rebatidas: nos instantes em que a bola inverte o sentido, descobre a coluna
   de cada raquete e o deslocamento entre os dois referenciais verticais.

Uso:
    python scripts/inspecionar_ram.py
    python scripts/inspecionar_ram.py --passos 600
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agentes import ram as mapa_ram  # noqa: E402
from ambiente_pong import AGENTE_DIREITA, AGENTE_ESQUERDA, criar_ambiente  # noqa: E402

# Conjunto mínimo de ações do Pong (full_action_space=False).
NOOP, FIRE, CIMA, BAIXO = 0, 1, 2, 3


def coletar(env, acao_esquerda: int, acao_direita: int, passos: int) -> np.ndarray:
    """Roda com ações fixas e devolve a matriz (passos, 128) da RAM."""
    observacoes, _ = env.reset(seed=0)
    historico = []
    for _ in range(passos):
        if not env.agents:
            break
        acoes = {AGENTE_ESQUERDA: acao_esquerda, AGENTE_DIREITA: acao_direita}
        acoes = {nome: acoes[nome] for nome in env.agents}
        observacoes, _, _, _, _ = env.step(acoes)
        if not env.agents:
            break
        historico.append(np.asarray(observacoes[AGENTE_ESQUERDA], dtype=np.int64))
    return np.array(historico)


def tendencia(historico: np.ndarray) -> np.ndarray:
    """Inclinação média de cada byte no tempo. Positiva quer dizer subindo."""
    passos = np.arange(len(historico), dtype=np.float64)
    passos = passos - passos.mean()
    centrado = historico - historico.mean(axis=0)
    return (passos[:, None] * centrado).sum(axis=0) / (passos**2).sum()


def testar_controle(env, passos: int) -> None:
    print("=" * 70)
    print("1. TESTE DE CONTROLE: que byte responde ao comando de cada raquete")
    print("=" * 70)

    # Um lado se move e o outro fica parado, então só o byte do lado ativo reage.
    # Inverter a ação depois separa movimento de verdade dos bytes que mudariam
    # de qualquer jeito, como os da bola.
    for lado, acoes_cima, acoes_baixo in (
        ("ESQUERDA (second_0)", (CIMA, NOOP), (BAIXO, NOOP)),
        ("DIREITA  (first_0)", (NOOP, CIMA), (NOOP, BAIXO)),
    ):
        com_cima = tendencia(coletar(env, *acoes_cima, passos))
        com_baixo = tendencia(coletar(env, *acoes_baixo, passos))

        # O byte da raquete inverte o sinal quando o comando inverte.
        inverteu = com_cima * com_baixo < 0
        forca = np.minimum(np.abs(com_cima), np.abs(com_baixo)) * inverteu

        print(f"\nRaquete {lado}, bytes que seguem o comando:")
        for byte in np.argsort(forca)[::-1][:3]:
            if forca[byte] == 0:
                continue
            sentido = "aumenta" if com_cima[byte] > 0 else "diminui"
            print(
                f"  byte {byte:>3}: inclinação com CIMA {com_cima[byte]:+.3f}, "
                f"com BAIXO {com_baixo[byte]:+.3f}  (ação CIMA {sentido} o byte)"
            )

    print("\nEsperado por agentes/ram.py:")
    print(f"  raquete esquerda = byte {mapa_ram.BYTE_RAQUETE_ESQUERDA}")
    print(f"  raquete direita  = byte {mapa_ram.BYTE_RAQUETE_DIREITA}")


def testar_variacao(env, passos: int) -> np.ndarray:
    print()
    print("=" * 70)
    print("2. TESTE DE VARIAÇÃO: bytes que mudam a cada passo, bola inclusa")
    print("=" * 70)

    observacoes, _ = env.reset(seed=1)
    rng = np.random.default_rng(0)
    n_acoes = env.action_space(AGENTE_ESQUERDA).n
    historico = []
    for _ in range(passos):
        if not env.agents:
            break
        acoes = {nome: int(rng.integers(n_acoes)) for nome in env.agents}
        observacoes, _, _, _, _ = env.step(acoes)
        if not env.agents:
            break
        historico.append(np.asarray(observacoes[AGENTE_ESQUERDA], dtype=np.int64))

    historico = np.array(historico)
    mudancas = (np.diff(historico, axis=0) != 0).mean(axis=0)

    print("\nTop 10 bytes por frequência de mudança:")
    for byte in np.argsort(mudancas)[::-1][:10]:
        coluna = historico[:, byte]
        print(
            f"  byte {byte:>3}: muda em {mudancas[byte]:6.1%} dos passos  "
            f"faixa [{coluna.min():>3}, {coluna.max():>3}]"
        )

    print("\nEsperado por agentes/ram.py:")
    print(f"  bola x = byte {mapa_ram.BYTE_BOLA_X}")
    print(f"  bola y = byte {mapa_ram.BYTE_BOLA_Y}")
    return historico


def relatar_faixas(historico: np.ndarray) -> None:
    print()
    print("=" * 70)
    print("3. FAIXAS DOS BYTES USADOS: base para calibrar as constantes")
    print("=" * 70)

    rotulos = {
        mapa_ram.BYTE_BOLA_X: "bola x",
        mapa_ram.BYTE_BOLA_Y: "bola y",
        mapa_ram.BYTE_RAQUETE_ESQUERDA: "raquete esquerda",
        mapa_ram.BYTE_RAQUETE_DIREITA: "raquete direita",
        mapa_ram.BYTE_PLACAR_ESQUERDA: "placar esquerda",
        mapa_ram.BYTE_PLACAR_DIREITA: "placar direita",
    }
    print()
    for byte, rotulo in rotulos.items():
        coluna = historico[:, byte]
        ativos = coluna[coluna > 0]  # zeros são os momentos fora de jogo
        if len(ativos) == 0:
            print(f"  byte {byte:>3} ({rotulo}): sempre 0, endereço suspeito")
            continue
        print(
            f"  byte {byte:>3} ({rotulo:<17}): min={ativos.min():>3} "
            f"max={ativos.max():>3} distintos={len(np.unique(ativos)):>3}"
        )

    print(
        f"\nConstantes atuais: "
        f"raquete y [{mapa_ram.RAQUETE_Y_MINIMO}, {mapa_ram.RAQUETE_Y_MAXIMO}] "
        f"bola y [{mapa_ram.BOLA_Y_MINIMO}, {mapa_ram.BOLA_Y_MAXIMO}] "
        f"DESLOCAMENTO_RAQUETE={mapa_ram.DESLOCAMENTO_RAQUETE}"
    )
    print("Se as faixas acima divergirem, ajuste em agentes/ram.py.")


def calibrar_eixo(env, passos: int) -> None:
    """Mede os extremos do eixo vertical e confere as colunas das raquetes."""
    print()
    print("=" * 70)
    print("4. CALIBRAÇÃO DO EIXO: extremos das raquetes e colunas fixas")
    print("=" * 70)

    no_topo = coletar(env, CIMA, CIMA, passos)
    no_fundo = coletar(env, BAIXO, BAIXO, passos)

    for rotulo, byte in (
        ("raquete esquerda", mapa_ram.BYTE_RAQUETE_ESQUERDA),
        ("raquete direita", mapa_ram.BYTE_RAQUETE_DIREITA),
    ):
        print(
            f"  byte {byte:>3} ({rotulo:<17}): "
            f"segurando CIMA -> {no_topo[:, byte].min():>3}..{no_topo[:, byte].max():>3}   "
            f"segurando BAIXO -> {no_fundo[:, byte].min():>3}..{no_fundo[:, byte].max():>3}"
        )

    print("\n  Colunas (x) das raquetes, que devem ser constantes:")
    for rotulo, byte in (
        ("raquete esquerda x", mapa_ram.BYTE_RAQUETE_ESQUERDA_X),
        ("raquete direita x", mapa_ram.BYTE_RAQUETE_DIREITA_X),
    ):
        coluna = no_topo[:, byte]
        print(
            f"  byte {byte:>3} ({rotulo:<19}): valores distintos = "
            f"{sorted(np.unique(coluna))[:6]}"
        )


def analisar_rebatidas(env, passos: int) -> None:
    """Descobre a linha de cada raquete e qual byte defende qual lado.

    Toda vez que a bola inverte o sentido horizontal é porque bateu em alguma
    raquete. Nesses instantes o bola_x entrega a coluna daquela raquete, e o byte
    que estiver verticalmente mais perto da bola é quem defende aquele lado. A
    distância média entre os dois vira o DESLOCAMENTO_RAQUETE.
    """
    print()
    print("=" * 70)
    print("5. REBATIDAS: coluna da raquete, lado de cada byte e deslocamento")
    print("=" * 70)

    observacoes, _ = env.reset(seed=7)
    rng = np.random.default_rng(1)
    n_acoes = env.action_space(AGENTE_ESQUERDA).n
    historico = []
    for _ in range(passos):
        if not env.agents:
            break
        acoes = {nome: int(rng.integers(n_acoes)) for nome in env.agents}
        observacoes, _, _, _, _ = env.step(acoes)
        if not env.agents:
            break
        historico.append(np.asarray(observacoes[AGENTE_ESQUERDA], dtype=np.int64))
    historico = np.array(historico)

    bola_x = historico[:, mapa_ram.BYTE_BOLA_X].astype(float)
    bola_y = historico[:, mapa_ram.BYTE_BOLA_Y].astype(float)
    em_jogo = (bola_x > 0) & (bola_y > 0)

    dx = np.diff(bola_x)
    valido = em_jogo[:-1] & em_jogo[1:] & (dx != 0)
    # Procura troca de sinal entre dois deslocamentos válidos seguidos.
    inverteu = np.zeros(len(dx), dtype=bool)
    indices = np.flatnonzero(valido)
    for anterior, atual in zip(indices, indices[1:]):
        inverteu[atual] = dx[anterior] * dx[atual] < 0

    momentos = np.flatnonzero(inverteu)
    if len(momentos) == 0:
        print("  Nenhuma rebatida detectada. Tente aumentar --passos.")
        return

    xs = bola_x[momentos]
    corte = (xs.min() + xs.max()) / 2
    print(f"\n  {len(momentos)} rebatidas detectadas. Colunas (bola x):")
    print(f"    lado de x baixo: {xs[xs <= corte].mean():.1f} "
          f"(min {xs.min():.0f})")
    print(f"    lado de x alto:  {xs[xs > corte].mean():.1f} "
          f"(max {xs.max():.0f})")

    for rotulo, selecao in (("x baixo", xs <= corte), ("x alto", xs > corte)):
        instantes = momentos[selecao]
        if len(instantes) == 0:
            continue
        print(f"\n  Quem estava alinhado com a bola no lado de {rotulo}:")
        for byte in (mapa_ram.BYTE_RAQUETE_ESQUERDA, mapa_ram.BYTE_RAQUETE_DIREITA):
            diferenca = bola_y[instantes] - historico[instantes, byte]
            print(
                f"    byte {byte:>3}: |bola_y - raquete| médio = "
                f"{np.abs(diferenca).mean():6.1f}   "
                f"diferença média = {diferenca.mean():+6.1f}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspeciona a RAM do Pong")
    parser.add_argument("--passos", type=int, default=400)
    args = parser.parse_args()

    env = criar_ambiente()
    try:
        testar_controle(env, args.passos)
        historico = testar_variacao(env, args.passos)
        relatar_faixas(historico)
        calibrar_eixo(env, args.passos)
        analisar_rebatidas(env, max(args.passos, 3000))
    finally:
        env.close()


if __name__ == "__main__":
    main()
