# Generate names

A fun project for diving into generation method in [pytorch](http://pytorch.org).  
There are times when you want to name your startup, product or baby! This model might tell you amazing suggestions.  
You just have to provide it some example names to generate new names.

## How to run
Just install the requirements using pip:
```bash
pip install -r requirements.txt
```

And run the follwoing command to download and prepare the babynames dataset:
```bash
python prepare.py
```
Finally you can train your model and generate sample names with the following command:
```bash
python train.py namesbystate.txt
python generate.py
```