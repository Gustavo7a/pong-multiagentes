# Pong Multiagente — Estudo Dirigido de IA (2026.1)

Ambiente multiagente **Pong** da biblioteca [PettingZoo](https://pettingzoo.farama.org/)
(categoria Atari), configurado para fornecer como observação apenas a **RAM do
console** (`obs_type="ram"`): a cada passo, cada agente recebe um vetor com os
**128 bytes** da memória do Atari (inteiros de 0 a 255).

Contexto II do estudo dirigido: dois tipos de agentes competem no mesmo
ambiente. Este repositório entrega a **base** (ambiente + baseline aleatório +
loop de partida). Os agentes inteligentes (heurístico, aprendizado por reforço,
genético) serão adicionados em `agentes/`.

## Estrutura

```
ambiente_pong.py      # fábrica do ambiente PettingZoo (Pong, RAM)
jogar.py              # roda partidas entre 2 agentes + métricas
agentes/
  base.py             # interface Agente (agir / reiniciar / observar)
  aleatorio.py        # AgenteAleatorio — baseline de referência
requirements.txt
```

## Ambiente

- **Jogo:** Pong (`pettingzoo.atari.pong_v3`), dois jogadores competitivos.
- **Agentes:** `first_0` (raquete esquerda) e `second_0` (raquete direita).
- **Observação:** vetor de 128 bytes da RAM (`obs_type="ram"`).
- **Ação:** discreta (movimentos da raquete).
- **Recompensa:** `+1` ao marcar ponto, `-1` ao sofrer (soma zero).
- **Término:** fim da partida ou `max_cycles`.

## Instalação

Requer Python 3.9–3.13.

> **Windows:** o Atari do PettingZoo depende de `multi-agent-ale-py`, que **não
> publica wheel para Windows** (só Linux e macOS ARM). No Windows use **WSL2
> (Ubuntu)** — as instruções abaixo são para Linux/WSL, onde a wheel instala
> pronta e o render sai pelo WSLg. No WSL, instale antes:
> `sudo apt install -y python3-venv python3-pip`.

```bash
python3 -m venv ~/pong-venv
source ~/pong-venv/bin/activate

pip install -r requirements.txt
```

Os ambientes Atari precisam das ROMs. O pacote `autorom[accept-rom-license]`
as instala automaticamente; se necessário, force com:

```bash
AutoROM --accept-license
```

## Execução

Inspecionar espaços de observação/ação:

```bash
python ambiente_pong.py
```

Rodar uma partida (aleatório x aleatório):

```bash
python jogar.py --episodios 5          # sem janela (rápido)
python jogar.py --episodios 3 --render # com janela
```

## Próximos passos

Adicionar em `agentes/`, todos herdando de `Agente` (`base.py`):

- [ ] agente heurístico (busca / estado / objetivo)
- [ ] agente de aprendizado por reforço
- [ ] agente genético

Cada novo agente só precisa implementar `agir(observacao) -> int` e pode ser
plugado no `jogar.py` no lugar de um `AgenteAleatorio`.
