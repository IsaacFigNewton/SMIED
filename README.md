# SMIED
#### SMIED is a pipeline for Semantic Metagraph-based Information Extraction and Decomposition.
---

## Quick Start
1. Install a SpaCy pipeline using one of the following commands:
    - `python -m spacy download en_core_web_sm`
    - `python -m spacy download en_core_web_md`
    - `python -m spacy download en_core_web_lg`
2. Run `pip install git+https://github.com/IsaacFigNewton/SMIED.git` to install SMIED from the repo's main branch.
3. Try running the full pipeline on a piece of text with the following snippet:
```python
    import spacy
    from smied import SemanticMetagraph
    
    nlp = spacy.load('en_core_web_sm')
    text = "The quick brown fox jumps over the lazy dog."
    doc = nlp(text)
    
    G = SemanticMetagraph(doc)
    
    G.plot()
```
---

## Testing
Note: If modifying parts of the package, you may want to install noske with `pip install -e git+https://github.com/IsaacFigNewton/SMIED.git` in lieu of step 2 above.

### Unittest Framework
Open and run `tests.py` in the SDE of your choice.

### Pytest Framework
1. SMIED should have installed the pytest package as one of its dependencies, but if it didn't, you can do so manually with `pip install pytest`
2. Run `python -m pytest` to run all the unit tests.
---