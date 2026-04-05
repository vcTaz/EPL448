# Contributing

Thank you for your interest in contributing to the CERN Dielectron Invariant
Mass Prediction project! Contributions of all kinds are welcome — bug fixes,
documentation improvements, new experiments, or better visualisations.

---

## Getting Started

1. **Fork** the repository and create your branch from `main`:

   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Set up the environment**:

   ```bash
   python -m venv .venv
   source .venv/bin/activate   # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Download the dataset** — see [data/README.md](data/README.md) for
   instructions. Place `dielectron.csv` in the `data/` directory.

---

## Workflow

- Keep commits small and focused. Write a clear, imperative subject line
  (e.g. `Add delta_phi wrapping unit test`).
- Open a pull request against `main` with a concise description of your
  changes and why they are needed.
- Reference related issues using `Closes #<issue-number>` in the PR body.

---

## Code Style

- Follow [PEP 8](https://peps.python.org/pep-0008/) for Python files in
  `src/`.
- Write NumPy-style docstrings for all public functions and classes.
- Keep notebooks cells focused; add Markdown cells to explain each step.
- Run notebooks top-to-bottom with a fresh kernel before committing them
  (`Kernel → Restart & Run All`).

---

## Testing

There is no automated test suite yet. Before submitting a PR, please manually
verify that:

- All `src/` modules import without errors.
- The main notebook runs end-to-end without exceptions.
- Any new helper function works correctly for edge-case inputs.

---

## Reporting Issues

Use the [GitHub Issues](../../issues) tracker:

- **Bug report** — unexpected errors or incorrect results.
- **Feature request** — ideas for new models, features, or visualisations.

Please include enough detail (Python version, OS, error traceback) to
reproduce the problem.

---

## Code of Conduct

By participating you agree to abide by the
[Code of Conduct](CODE_OF_CONDUCT.md).
