# Regression Models

This repository contains implementations of Linear Regression and Logistic Regression models using gradient descent.

## Repository Structure

```
examples.ipynb        # Examples of model usage
linearRegression.py   # Linear regression implementation
logisticRegression.py # Logistic regression implementation
regression.py         # Abstract base class for regressions
```

## Class Overview

### 1. `LinearRegression`
- Performs linear regression with gradient descent.
- Supports L1 (Lasso) and L2 (Ridge) regularization.

### 2. `LogisticRegression`
- Binary classification with gradient descent.
- Supports L1 and L2 regularization.

Both classes inherit from a common `Regression` base class (`regression.py`).

## Setup Instructions

### 1. Create a virtual environment:
```bash
python -m venv venv
```

### 2. Activate the virtual environment:

- **Windows:**
  ```bash
  venv\Scripts\activate
  ```

- **macOS/Linux:**
  ```bash
  source venv/bin/activate
  ```

### 3. Install dependencies:
```bash
pip install -r requirements.txt
```