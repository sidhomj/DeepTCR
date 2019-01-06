#!/bin/bash

# Render the notebook to HTML
jupyter nbconvert \
  --ExecutePreprocessor.allow_errors=True \
  --ExecutePreprocessor.timeout=-1 \
  --output-dir =/results \
  --execute Tutorial.ipynb