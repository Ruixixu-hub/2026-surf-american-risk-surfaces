# Basic / Original Classical Benchmark: CN + PSOR

CN+PSOR is the original American-option benchmark. It discretizes the
Black--Scholes obstacle problem with Crank--Nicolson and solves each time-step
LCP using projected successive over-relaxation.

The canonical implementation and evidence remain in the original project:

- [`experiments/05_obstacle_complementarity_diagnostics.py`](../../../experiments/05_obstacle_complementarity_diagnostics.py)
- [`src/american_risk_surfaces/solvers/cn_psor.py`](../../../src/american_risk_surfaces/solvers/cn_psor.py)
- [`results/01_solver_validation/`](../../../results/01_solver_validation/)

This method is retained as the basic/original classical benchmark rather than
duplicated in the extension portfolio.
