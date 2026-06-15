# Ticket 00 Setup Check

Date: 2026-06-15

## Scope

This is the Ticket 0 setup-only report for `reports/03_solver/solver_implementation_tickets.md`.
No solver source code, Ticket 1 files, payoff utilities, Black-Scholes utilities, CN/PSOR code,
boundary code, Greek code, dataset code, stress-map code, or neural-network code were created.

Source context checked:

- `reports/03_solver/solver_implementation_tickets.md`, Ticket 0.
- `reports/00_planning/planning_report.md`, workflow stages and decision gates.
- `reports/03_solver/solver_validation_plan.md`, sections 14 and 19.
- `docs/student_handout_full_report.pdf`, reproducibility guide.

## 1. Current Repo Structure

The repository is currently documentation-first. After this Ticket 0 report folder is created, the
top-level structure is:

```text
.
|-- README.md
|-- docs/
|   |-- README.md
|   |-- Student_Methodology_FreeBoundary_Risk_Surfaces.pdf
|   `-- student_handout_full_report.pdf
`-- reports/
    |-- 00_planning/
    |   `-- planning_report.md
    |-- 01_literature/
    |   `-- literature_map.md
    |-- 02_math/
    |   `-- formulation_note.md
    `-- 03_solver/
        |-- solver_implementation_tickets.md
        |-- solver_validation_plan.md
        `-- tickets/
            `-- ticket_00_setup_check.md
```

No Python source files were found.

## 2. Required Directory Check

| Directory | Exists now? | Notes |
| --- | --- | --- |
| `src/` | No | Future source package location. |
| `tests/` | No | Future test location. |
| `experiments/` | No | Future validation script location. |
| `results/` | No | Future generated artifact location. |

The solver validation plan lists future files under `src/`, `experiments/`, and `results/`, but it
also says not to create those files during the planning stage. This report follows that rule.

## 3. Recommended Python Package and Import Convention

Recommended convention:

- Use a `src` layout with one explicit package namespace, preferably
  `src/american_risk_surfaces/`.
- Put future modules under package subdirectories, for example
  `american_risk_surfaces.solvers`, `american_risk_surfaces.diagnostics`,
  `american_risk_surfaces.boundaries`, and `american_risk_surfaces.greeks`.
- Use absolute imports in tests and experiments, for example:

```python
from american_risk_surfaces.solvers.black_scholes import ...
```

Rationale:

- This is compatible with the student handout's `PYTHONPATH=src` convention.
- It avoids importing from bare top-level names such as `solvers`, which are more likely to collide
  with unrelated packages.
- It keeps future Ticket 1 and later implementation import paths clear before code exists.

If the team chooses to keep the roadmap's shorter future paths such as `src/solvers/...`, that can
work with `PYTHONPATH=src`, but it is less robust than a single project package namespace.

## 4. Dependency Status

No project dependency or packaging file was found. Specifically, no `pyproject.toml`,
`requirements*.txt`, `setup.py`, `setup.cfg`, `tox.ini`, `noxfile.py`, `pytest.ini`, `uv.lock`,
`Pipfile`, `poetry.lock`, or `environment*.yml` file was present.

Observed local runtime status:

| Runtime | Available packages checked |
| --- | --- |
| System `python3` 3.13.5 | `numpy`, `scipy`, `pandas`, and `matplotlib` are importable; `pytest`, `reportlab`, `pypdf`, and `pdfplumber` are not importable. |
| Bundled Codex Python | `numpy`, `pandas`, `reportlab`, `pypdf`, and `pdfplumber` are importable; `scipy`, `matplotlib`, and `pytest` are not importable. |

Observed PDF tooling:

- Bundled `pdfinfo` is available.
- Bundled `pdftoppm` is available.
- Bundled `pdftotext` is not available.

Conclusion: the repository currently has no declared reproducible dependency environment. Local
machines may have enough scientific packages to start prototyping, but that is not a project-level
setup.

## 5. Recommended Test Command

Because `pytest` is not installed in either checked Python runtime and the handout already gives a
standard-library test command, the recommended immediate test command is:

```bash
PYTHONPYCACHEPREFIX=/private/tmp/codex_pycache PYTHONPATH=src \
python3 -m unittest discover -s tests
```

Recommended compile check once `src/`, `tests/`, and `experiments/` exist:

```bash
PYTHONPYCACHEPREFIX=/private/tmp/codex_pycache PYTHONPATH=src \
python3 -m compileall -q src tests experiments
```

Current result: there is no `tests/` directory and no Python source. Existing test discovery is
therefore not meaningful yet. `python3 -m pytest --version` failed because `pytest` is not installed.
`python3 -m compileall .` completed successfully but found no project Python source to compile.

If the team wants `pytest`, approve a setup change first and declare it in project metadata. Then the
future command can become:

```bash
PYTHONPATH=src python3 -m pytest tests -q
```

## 6. Setup Changes Needed Before Ticket 1

Yes, at least a small setup decision is needed before Ticket 1 implementation begins.

Recommended but not created in this Ticket 0 pass:

- Add a minimal `pyproject.toml` or `requirements.txt`.
- Decide whether the project uses `unittest` only or adds `pytest`.
- Create `src/` and `tests/` as part of the approved setup or Ticket 1 work.
- Prefer the package namespace `american_risk_surfaces` under `src/`.
- Declare numerical dependencies needed by future solver work, likely at least `numpy`; add `scipy`,
  `pandas`, and `matplotlib` only if the implementation and validation scripts actually use them.

No setup files were created because the Ticket 0 instructions require proposing setup files in this
report unless explicitly approved.

## 7. Next Recommended Action

Human review should approve one of these setup paths before Ticket 1:

1. Minimal standard-library testing path: create `src/american_risk_surfaces/`, `tests/`, and use
   `unittest` with the handout command.
2. Pytest path: create `pyproject.toml`, declare runtime and test dependencies, and use
   `python3 -m pytest tests -q`.

Recommended choice: start with the minimal standard-library testing path unless the team already
wants `pytest` conventions. After that approval, Ticket 1 can add only payoff and European
closed-form utility work with focused tests.

## Inspection Commands Run

```bash
sed -n '1,220p' reports/03_solver/solver_implementation_tickets.md
pwd
ls -la
rg --files -g 'pyproject.toml' -g 'requirements*.txt' -g 'setup.py' -g 'setup.cfg' -g 'tox.ini' -g 'noxfile.py' -g 'pytest.ini' -g '.python-version' -g 'uv.lock' -g 'Pipfile' -g 'poetry.lock' -g 'environment*.yml'
find . -maxdepth 2 -type d
rg --files
sed -n '1,220p' README.md
find . -maxdepth 1 -type d -name src -o -name tests -o -name experiments -o -name results
rg -n "^(##|###)|Gate|Stage|stage|gate|reproduc|environment|dependency|test|pytest|src|experiments|results" reports/00_planning/planning_report.md
rg -n "^(##|###)|14|19|reproduc|artifact|test|pytest|dependency|environment|src|experiments|results" reports/03_solver/solver_validation_plan.md
which python3
python3 --version
which pytest
find . -name '*.py' -type f
python3 -m pytest --version
python3 -m compileall .
/Users/xrx/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -m pytest --version
/Users/xrx/.cache/codex-runtimes/codex-primary-runtime/dependencies/bin/pdfinfo docs/student_handout_full_report.pdf
python3 -c "import importlib.util; ..."
/Users/xrx/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -c "import importlib.util; ..."
sed -n '622,642p' reports/03_solver/solver_validation_plan.md
sed -n '767,780p' reports/03_solver/solver_validation_plan.md
sed -n '247,317p' reports/00_planning/planning_report.md
sed -n '399,444p' reports/00_planning/planning_report.md
/Users/xrx/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -c "from pypdf import PdfReader; ..."
sed -n '1,220p' docs/README.md
git status --short
git branch --show-current
mkdir -p reports/03_solver/tickets
```
