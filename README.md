# PDC
Efficient phase diagram construction based on uncertainty sampling

## USAGE
- The following command outputs the next point in 'next_point.csv' without parameter constraint.
  - `python PDC_sampler.py data.csv --estimation [Estimation method] --sampling [Sampling method]`
  - Estimation method: LP or LS
  - Sampling method: LC, MS, EA, or RS
- The following command outputs the next point in 'next_point.csv' with parameter constraint.
  - `python PDC_sampler.py data.csv --estimation [Estimation method] --sampling [Sampling method] --parameter_constraint --prev_point '20, 100'`
- The following command outputs the multiple next points in 'next_point.csv' without parameter constraint.
  - `python PDC_sampler.py data.csv --estimation [Estimation method] --sampling [Sampling method] --multi_method [Multiple method] --multi_num [Number of suggestion] --NE_k [k]`
  - Estimation method: LP or LS
  - Sampling method: LC, MS, EA, or RS
  - Multiple method: OU or NE (OU: Only US ranking, NE: Neighobr Exclusion)
  - Number of suggestions: integer
  - k: k value in neighbor exclusion

## Reference
Kei Terayama, Ryo Tamura, Yoshitaro Nose, Hidenori Hiramatsu, Hideo Hosono, Yasushi Okuno, Koji Tsuda, [Efficient Construction Method for Phase Diagrams Using Uncertainty Sampling](https://doi.org/10.1103/PhysRevMaterials.3.033802), *Physical Review Materials*, Vol. 3, No. 3, 033802, 2019. [DOI: 10.1103/PhysRevMaterials.3.033802]
