# TENAX Project

This repository provides the experimental source code of the TENAX project.

## File Description

The folder **Datasets** stores automatically generated datasets, and code files related to data processing.


The folder **PUF** stores the code of different PUF classes.


The folder **Save** stores experiment logs generated during the runs.


**TRL.py** defines the TRL layer structure used in TENAX.


**AttackWithPca_main.py** is the source code that uses TENAX to attack XOR APUFs. The dataset will be automatically generated according to the random seed.  

**main.py** is the source code for attacking XOR APUFs using TENAX(TT-Only) or LR.  

**AttackIPUF_main.py** and **SplitIPUF.py** are the source codes for attacking IPUF with TENAX and SplitIPUF, respectively. The dataset will be automatically generated according to the random seed.

**AttackLSPUF_main.py** and **BreakingLSPUF.py** are the source codes for attacking LSPUFs with TENAX(TT-Only) and BreakingLSPUF, respectively. 

## Environment Requirements

- Python 3.6  
- TensorFlow 2.10 GPU version  
- scikit-learn  
- pandas  

## Quick Start

### 1. Attack XOR APUF

- Run a simple attack using TENAX:

```bash
python AttackWithPca_main.py
```

- Run a simple attack using TENAX(TT-Only):

```bash
python main.py
```

- To use LR instead, modify

```python
method = "Ett"
```

to

```python
method = "LR"
```

and run:

```bash
python main.py
```





### 2. Attack iPUF

run:

```bash
python AttackIPUF_main.py
```

to run a simple attack on a 64-bit (1,5)-iPUF using TENAX.

Alternatively, run:

```bash
python SplitIPUF.py
```

to run a simple attack on a 64-bit (1,5)-iPUF using Split iPUF.



### 3. Attack LSPUF

- Run a simple attack using TENAX(TT-Only):

```bash
python AttackLSPUF_main.py
```

- Run a simple attack using BreakingLSPUF:

```bash
python BreakingLSPUF.py
```
