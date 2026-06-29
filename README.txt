OGC 2026 Optimization Challenge
================================

Contents
--------
  alg_tester/    Algorithm testing tool.
                 See alg_tester/README.txt for details.
  baseline/      Baseline algorithm template.
                 See baseline/README.txt for details.
  ogc2026_env.yml  Conda environment definition.

Environment setup (once)
------------------------
  conda env create -f ogc2026_env.yml

  We recommend Miniforge as the conda distribution:
    https://github.com/conda-forge/miniforge

Quick start
-----------
  Step 1  Set up the conda environment (see above).
  Step 2  Open baseline/ and edit myalgorithm.py.
  Step 3  Test your algorithm with the Algorithm Tester:
            conda activate ogc2026
            cd alg_tester
            python alg_tester_app.py
