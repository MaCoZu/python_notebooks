# AI Coding Guidelines for Python Data Science Learning Repository

## Project Overview
This workspace contains Jupyter notebooks for learning Python, Pandas, data wrangling, and related data science topics. It's organized by subject areas (Pandas/, Python/, Misc/) with practical examples and code challenges.

## Key Patterns and Conventions

### Data Loading
- Load datasets from Google Drive using direct download links:
  ```python
  df = pd.read_csv("https://drive.google.com/uc?id=1oE-3rt17bFW7fOzDIjwFSEMPTIV3NvcO")
  ```
- Reference: [Pandas/02_Pandas_Selecting_Indexing_Filtering.ipynb](Pandas/02_Pandas_Selecting_Indexing_Filtering.ipynb)

### Pandas Column Selection
- Use `df.filter(regex="^pattern", axis=1)` for selecting columns by name patterns (e.g., all grades starting with "grade"):
  ```python
  df.filter(regex="^grade", axis=1)  # columns starting with "grade"
  df.filter(regex="t2$", axis=1)     # columns ending with "t2"
  df.filter(like="math", axis=1)     # columns containing "math"
  ```
- For multiple specific columns, use `df[["col1", "col2"]]`
- Reference: [Pandas/02_Pandas_Selecting_Indexing_Filtering.ipynb](Pandas/02_Pandas_Selecting_Indexing_Filtering.ipynb)

### Indexing and Slicing
- Use `df.loc[]` for label-based indexing, `df.iloc[]` for position-based
- Boolean indexing with conditions: `df[(df["col"] > value) & (df["other"] == val)]`
- Slicing with steps: `df.iloc[::2, ::2]` for every other row/column
- Reference: [Pandas/02_Pandas_Selecting_Indexing_Filtering.ipynb](Pandas/02_Pandas_Selecting_Indexing_Filtering.ipynb)

### Environment and Type Checking
- Python version: 3.14.2 (managed by pyenv)
- Use strict mypy for type checking on .py files (see mypy.ini)
- Environment setup via direnv (.envrc)

### Libraries
- Primary: pandas, numpy
- Alternative: polars (see Polars_cheatsheet.py)
- Additional: openpyxl for Excel, requests for APIs

## Workflows
- Develop in Jupyter notebooks within VS Code
- Export notebook code to .py files for mypy validation when needed
- Data sources available at: https://drive.google.com/drive/folders/1UzgxrOvtdJwKui7gbKhzohp5e_WQihSP

## Code Style
- Follow patterns established in existing notebooks
- Use descriptive variable names (e.g., df for DataFrames)
- Include comments for complex operations</content>
<parameter name="filePath">/home/mz/code/Python/.github/copilot-instructions.md