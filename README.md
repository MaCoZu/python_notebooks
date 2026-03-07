# Python & Data Science Learning Repository

A comprehensive monorepo for learning Python and data science, organized for efficiency and best practices.

## Structure

- **pandas/** - DataFrame manipulation and analysis
- **numpy/** - Array computing and numerical operations
- **python/** - Core Python language features
- **sklearn/** - Machine learning with scikit-learn
- **visualization/** - Plotting libraries (matplotlib, seaborn, plotly)
- **statistics/** - Statistical methods and theory
- **sql/** - SQL and database interactions
- **utils/** - Reusable utilities package
- **tests/** - Unit tests for utils package
- **docs/** - Documentation and learning notes
- **data/** - Datasets (gitignored)
- **challenges/** - Coding challenges

## Quick Start

```bash
# Install dependencies
make install

# Install pre-commit hooks
make pre-commit-setup

# Run all checks
make all
```

## Development Workflow

```bash
# Format code
make format

# Run linting
make lint

# Type check
make type

# Run tests
make test
```

## Utils Package

Reusable utilities for DRY code:

```python
from utils.cleaning import clean_columns, handle_missing
from utils.plotting import quick_hist
from utils.stats import describe_numeric

df = clean_columns(df)
df = handle_missing(df, strategy="median")
quick_hist(df['age'])
```

## Adding Dependencies

```bash
# Core dependency
uv add package-name

# Optional group
uv add --optional-group ml tensorflow

# Install with extras
uv sync --extra ml
uv sync --all-extras
```

## Tools

- **uv** - Fast dependency management
- **ruff** - Linting and formatting
- **mypy** - Type checking
- **pytest** - Testing framework
- **pre-commit** - Git hooks
- **GitHub Actions** - CI/CD

## Data

Datasets are stored in `data/` (gitignored). Download from:
https://drive.google.com/drive/folders/1UzgxrOvtdJwKui7gbKhzohp5e_WQihSP?usp=sharing
