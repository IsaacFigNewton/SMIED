# SMIED
#### SMIED is a pipeline for Semantic Metagraph-based Information Extraction and Decomposition.
---

## Install Dependencies
For OpenCog's Atomspace
```
    <!-- Install general dependencies -->
    apt install libboost-all-dev cmake guile-3.0-dev pkg-config cxxtest
    
    <!-- Install, configure Octool -->
    sudo curl -L http://raw.github.com/opencog/ocpkg/master/ocpkg -o /usr/local/bin/octool
    sudo chmod +x /usr/local/bin/octool
    sudo octool -rdcpav -l default
    sudo octool -rdcv
    sudo octool -rdv

    <!-- Install cogutil -->
    git clone https://github.com/opencog/cogutil
    cd ./cogutil &&\
        mkdir build &&\
        cd build &&\
        cmake .. &&\
        make -j$(nproc) &&\
        sudo make install
    
    <!-- Install AtomSpace -->
    git clone https://github.com/opencog/atomspace
    cd ./atomspace &&\
        mkdir build &&\
        cd build &&\
        cmake .. &&\
        make -j$(nproc) &&\
        sudo make install

    <!-- Install OpenCog -->
    git clone https://github.com/opencog/opencog
    cd ./opencog &&\
        mkdir build &&\
        cd build &&\
        cmake .. &&\
        make -j$(nproc) &&\
        sudo make install
```
---

## Quick Start
1. Run `pip install git+https://github.com/IsaacFigNewton/SMIED.git` to install SMIED from the repo's main branch.
2. Try running the full pipeline on a piece of text with the following snippet:
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