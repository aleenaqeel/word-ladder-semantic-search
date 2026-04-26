# Word Ladder Search in Semantic Embedding Space

---

## About

Generalizes the classic word ladder problem using GloVe 100-dimensional 
word embeddings. A path is found from a source word to a target word by 
stepping through semantically similar words using various search algorithms.

---

## Algorithms

- BFS, DFS, Uniform Cost Search
- Greedy Best-First Search, A*

---

## Setup

Download `glove.100d.20000.txt` separately and place it in the root folder, 
then run:

```bash
pip install -r requirements.txt
streamlit run gui/app.py
```

---

## Submission
Includes source code, GUI and report.pdf
