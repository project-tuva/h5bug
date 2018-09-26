MPIRUN=mpirun
PYTHON=python3.5
NP=5

run:
	$(MPIRUN) -np $(NP) $(PYTHON) h5manager_main.py