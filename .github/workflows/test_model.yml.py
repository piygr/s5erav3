name: Test MNIST Model

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - name: Checkout repository
      uses: actions/checkout@v3

    - name: Set up Python
      uses: actions/setup-python@v3
      with:
        python-version: 3.8

    - name: Install dependencies
      run: pip install -r requirements.txt

    - name: Run training script
      run: python main.py

    - name: Run model test
      run: python tests/test_model.py