OGC 2026 Baseline Algorithm
============================

Files
-----
  myalgorithm.py      -- YOUR algorithm goes here.  Fill in the algorithm()
                         function; do not change the function signature.
  baseline_greedy.py  -- Reference greedy implementation.  You may use it as
                         a starting point or call it from myalgorithm.py.
  utils.py            -- Feasibility checker and scoring utilities.
                         Do NOT modify this file.

Algorithm interface
-------------------
  myalgorithm.py must define:

      def algorithm(prob_info: dict, timelimit: float) -> dict:
          ...
          return solution

  prob_info  -- dict loaded from the problem instance JSON file
  timelimit  -- wall-clock seconds allowed for computation
  solution   -- dict matching the submission format defined in the problem
                statement

  You may import other modules or define helper functions freely inside
  myalgorithm.py, as long as the algorithm() signature is unchanged.

Requirements
------------
  Install the ogc2026 conda environment (once):
    conda env create -f ogc2026_env.yml

  We recommend Miniforge as the conda distribution:
    https://github.com/conda-forge/miniforge

Testing with Algorithm Tester
------------------------------
  1. Launch alg_tester (conda activate ogc2026 first):
       python alg_tester_app.py

  2. Step 1: select a problem instance JSON file.
  3. Step 2: select THIS folder (the one containing myalgorithm.py).
  4. Step 3: set a time limit and click [Run].
     The Solution tab shows the feasibility check result and objective value.

Feasibility check (standalone)
--------------------------------
  from utils import check_feasibility
  result = check_feasibility(prob_info, solution)
  print(result)   # {"stage": "PASS", "objective": <value>} on success
