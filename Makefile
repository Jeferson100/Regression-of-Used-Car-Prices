install:
	# Comando para atualizar pip e instalar dependências do arquivo requirements.txt
	# pip install --user -r requirements.txt
	#pip cache purge && \
	pip install --upgrade pip && \
	pip install -r requirements.txt

format:
	black *.ipynb

lint:
	pylint --disable=R,C *.ipynb

pyright:
	pyright *.ipynb

test:
	python -m pytest -vv --cov=*.ipynb

refactor: format lint

all: install lint format pyright test
