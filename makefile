.PHONY: build
install:
	python3 -m venv venv && \
	. venv/bin/activate && \
	pip3 install -U pip setuptools wheel && \
	pip3 install -r requirements-dev.txt
lint:
	. venv/bin/activate && \
	pylint --rcfile=.pylintrc ccc/*.py
test:
	. venv/bin/activate && \
	pytest
coverage:
	. venv/bin/activate && \
	pytest --cov-report term-missing -v --cov=ccc/
build:
	pip3 install --upgrade setuptools wheel
	python3 setup.py sdist bdist_wheel

clean: 
	rm -rf *.egg-info build/
