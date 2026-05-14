# Contributing to Machine-Learning

Thank you for considering contributing to this project! Contributions of all kinds are welcome — bug fixes, new concept notebooks, new ML projects, documentation improvements, and dataset additions.

---

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [How to Contribute](#how-to-contribute)
  - [Reporting Bugs](#reporting-bugs)
  - [Suggesting Enhancements](#suggesting-enhancements)
  - [Adding a Concept Notebook](#adding-a-concept-notebook)
  - [Adding a New ML Project](#adding-a-new-ml-project)
- [Development Workflow](#development-workflow)
- [Style Guidelines](#style-guidelines)
- [Pull Request Checklist](#pull-request-checklist)

---

## Code of Conduct

This project adheres to the [Contributor Covenant](https://www.contributor-covenant.org/version/2/1/code_of_conduct/) Code of Conduct. By participating, you are expected to uphold this standard. Please report unacceptable behaviour to the project maintainer via GitHub.

---

## Getting Started

1. **Fork** the repository on GitHub.
2. **Clone** your fork locally:
   ```bash
   git clone https://github.com/<your-username>/Machine-Learning.git
   cd Machine-Learning
   ```
3. **Create a virtual environment** and install dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate        # Windows: .venv\Scripts\activate
   pip install numpy pandas matplotlib seaborn scikit-learn torch jupyter
   ```
4. **Create a feature branch** from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```

---

## How to Contribute

### Reporting Bugs

- Search [existing issues](https://github.com/HarshaVardhanMannem/Machine-Learning/issues) before opening a new one.
- Use the **Bug Report** issue template.
- Include: environment details (OS, Python version, library versions), steps to reproduce, expected vs. actual behaviour, and any relevant screenshots or log output.

### Suggesting Enhancements

- Open a [Feature Request](https://github.com/HarshaVardhanMannem/Machine-Learning/issues) issue.
- Describe the motivation, the proposed solution, and any alternatives you considered.

### Adding a Concept Notebook

Concept notebooks live in `ML concepts/`. A good notebook:

- Focuses on **one clearly scoped ML concept** (e.g. a single algorithm, metric, or preprocessing technique).
- Opens with a **problem statement** explaining *why* the concept matters.
- Uses a **real or well-known public dataset**. Place data files in `ML concepts/data/`.
- Contains **exploratory analysis** and **visualisations** before modelling.
- Documents **key engineering decisions** (e.g. scaler fitted only on training data).
- Ends with a **findings summary** and takeaways.
- Uses **Markdown cells** (not just code comments) to narrate the story.
- Is fully runnable top-to-bottom with a fresh kernel (`Kernel → Restart & Run All`).

Update the root `README.md` to include your notebook in the *ML Concepts Experiments* and *Core ML Concepts Covered* sections.

### Adding a New ML Project

End-to-end projects live in `MLProjects/<ProjectName>/`. A production-grade project:

- Separates concerns: data loading, feature engineering, training, evaluation, and inference in distinct modules.
- Persists all artifacts (model, preprocessor, thresholds) to an `artifacts/` directory.
- Includes a `README.md` inside the project folder with setup, usage, configuration, and architecture documentation.
- Includes at minimum: training script, inference API / batch scorer, configuration file, and at least one evaluation metric.
- Contains no hard-coded paths; use `pathlib` or config files.
- Is fully reproducible given the same dataset and random seeds.

Update both `MLProjects/README.md` and the root `README.md` to document the new project.

---

## Development Workflow

```
main
 └── feature/<short-description>   ← your branch
```

1. Make atomic commits with clear messages (`feat:`, `fix:`, `docs:`, `refactor:`, `test:`).
2. Keep notebooks clean: clear all cell outputs before committing (`Cell → All Output → Clear`).
3. Push your branch and open a **Pull Request** against `main`.
4. Address review feedback in additional commits (do not force-push to an open PR).

---

## Style Guidelines

| Area | Convention |
|---|---|
| Python code | [PEP 8](https://peps.python.org/pep-0008/) |
| Imports | `stdlib` → `third-party` → `local`, sorted alphabetically within each group |
| Docstrings | [Google style](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings) |
| Notebook filenames | `Snake_Case_Description.ipynb` |
| Project folder names | `PascalCaseProjectName/` |
| Random seeds | Always set `random_state=42` for reproducibility |
| Data leakage | Scalers and encoders must be fit on training data only |

---

## Pull Request Checklist

Before requesting a review, confirm:

- [ ] Branch is up-to-date with `main`
- [ ] Notebook cells are cleared of output before committing
- [ ] Notebook runs top-to-bottom without errors on a fresh kernel
- [ ] Root `README.md` updated to reference any new notebook or project
- [ ] New project folder contains its own `README.md`
- [ ] No secrets, API keys, or personally identifiable information committed
- [ ] Commit messages follow the `type: description` convention

---

*Thank you for helping make this repository better!*
