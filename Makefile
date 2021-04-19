# define the name of the virtual environment directory
VENV := env

# default target, when make executed without arguments
.PHONY: all
all: venv

$(VENV)/bin/activate: requirements.txt
	python3 -m venv $(VENV)
	./$(VENV)/bin/pip install -r requirements.txt

model/model.sav: venv
	@echo "generating model"
	@cd model && ../$(VENV)/bin/python3 train.py

# venv is a shortcut target
.PHONY: venv
venv: $(VENV)/bin/activate

.PHONY: run
run: venv model/model.sav
	./$(VENV)/bin/python3 server.py

.PHONY: clean
clean:
	rm -rf $(VENV)
	rm model/model.sav
	find . -type f -name '*.pyc' -delete
