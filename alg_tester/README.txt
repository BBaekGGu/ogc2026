OGC 2026 Algorithm Tester
=========================

Requirements
------------
  Install the ogc2026 conda environment (once):
    conda env create -f ogc2026_env.yml

  We recommend Miniforge as the conda distribution:
    https://github.com/conda-forge/miniforge

Run
---
  conda activate ogc2026
  python alg_tester_app.py

How to use
----------
  Step 1  Click "..." next to Instance  and select a problem JSON file.
  Step 2  Click "..." next to Algorithm and select the folder that contains
          your myalgorithm.py.
  Step 3  Set the time limit, then click [Run].
          Results and the feasibility check appear in the Solution tab.

Algorithm interface
-------------------
  Your myalgorithm.py must define:

      def algorithm(prob_info: dict, timelimit: float) -> dict:
          ...
          return solution

  prob_info  -- dict loaded from the problem instance JSON
  timelimit  -- seconds allowed for computation
  solution   -- dict matching the submission format defined in the problem
                statement

Notes
-----
  * The algorithm runs as a subprocess, so stdout/stderr output from your
    code appears in the log panel inside the app.
  * The feasibility checker (utils.py) is the same one used for official
    scoring; a "PASS" result here means the solution is valid.
