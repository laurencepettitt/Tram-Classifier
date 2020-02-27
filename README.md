# Tram classifier

A sound-based classifier of several trams in operation in Prague.

## Installation
Make sure you have a **virtual** python environment, with a **python version newer than 3.7**.

Then just install the dependencies.
```bash
pip install -e .
```

## Usage
Just run the classifier on a file.
```bash
python tram_classifier/classify.py [/path/to/test_file.wav]
```
You will get a CSV output, of Tram Events found in the file.

### Retraining from scratch
If your would like to retrain the CNN classifier:
```bash
python tram_classifier/training/train.py
```
It will take a while to download the dataset (~1GB), extract features and fit the data to the model. Subsequent runs will use cached data for much of this. To reset a cache, you can delete the cache's folder, such as ``tram_classifier/cache`` or ``tram_classifier/dataset/cache``.
