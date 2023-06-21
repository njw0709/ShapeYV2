install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt

format:
	black *.py

lint:
	pylint --disable=R,C ./shapeymodular/

upload:
	if [ -d "dist" ]; then rm -r dist; fi
	mkdir dist
	python setup.py sdist
	twine upload --verbose --repository-url https://upload.pypi.org/legacy/ dist/*

test:
	python -m pytest -vs tests

all: install format lint upload