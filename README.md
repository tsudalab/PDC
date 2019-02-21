# PDC
Efficient phase diagram construction based on uncertainty sampling

## USAGE
- Output the next point in 'next_point.csv' without parameter constraint.
  - `python PDC_sampler.py data.csv --estimation [Estimation method] --sampling [Sampling method]`
  - Estimation method: LP or LS
  - Sampling method: LC, MS, EA, or RS
- Output the next point in 'next_point.csv' with parameter constraint.
  - `python PDC_sampler.py data.csv --estimation [Estimation method] --sampling [Sampling method] --parameter_constraint --prev_point '20, 100'`

## Reference
Kei Terayama, Ryo Tamura, Yoshitaro Nose, Hidenori Hiramatsu, Hideo Hosono, Yasushi Okuno, Koji Tsuda, Efficient Construction Method for Phase Diagrams Using Uncertainty Sampling, arXiv:1812.02306, 2018. [https://arxiv.org/abs/1812.02306]
