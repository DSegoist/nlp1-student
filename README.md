We are using python 3.14, here is how you reproduce our setup.

```bash
python3.14 -m venv ~/envs/nlp1-2026
source ~/envs/nlp1-2026/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
jupyter lab
```

Each `part{N}` folder contains a tutorial notebook `tutorial.ipynb` and project instructions in `project.ipynb`.
