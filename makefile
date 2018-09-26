MPIRUN=mpirun
PYTHON=python
NP=5

run:
	$(MPIRUN) -np $(NP) $(PYTHON) h5manager_main.py