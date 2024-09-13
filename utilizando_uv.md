# Introdução

TL;DR: [uv](https://docs.astral.sh/uv/) é um instalador e resolvedor de pacotes Python extremamente rápido , escrito em Rust e projetado como um substituto imediato para fluxos de trabalho pipe pip-tools.

uv representa um marco em nossa busca por um "Cargo for Python" : um gerenciador de projetos e pacotes Python abrangente que seja rápido, confiável e fácil de usar.

Como parte deste lançamento, também estamos assumindo a administração do Rye , uma ferramenta experimental de empacotamento Python de Armin Ronacher . Manteremos o Rye enquanto expandimos o uv para um projeto sucessor unificado, para
cumprir nossa visão compartilhada para o empacotamento Python

## Instalando 

Para instalar o `uv` em seu ambiente de desenvolvimento, execute:

```bash
pip install uv
```
Endereco do projeto no [pip](https://pypi.org/project/uv/).

## Usando o uv

Para instalar suas dependências, execute.

```bash
uv pip install -r requirements.txt
```

Tambem é possível instalar dependências a partir de um arquivo [pyproject.toml](pyproject.toml):	


```bash
uv pip install -r pyproject.toml
```

## Ambiente python

O `uv` tambem pode ser usado para criar ambientes virtuais de Python. Sendo mais rapido e mais confiável.

```bash	
uv venv --python 3.12.0

source .venv/bin/activate
```



