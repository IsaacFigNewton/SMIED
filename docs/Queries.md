# Queries
The only query structure I've implemented so far consists of simple path homeomorphism queries.
In later work I plan to incorporate a reduced form of the [Graphbrain hypergraph querying language](https://graphbrain.net/index.html).


## Path Query Format
Path queries are encoded as lists of dictionaries containing attributes of the associated graph feature.
Since paths are essentially ordered sequences of nodes and edges, a query looks for a subgraph homeomorphism by matching each node-dict against a graph node and each edge-dict against the graph’s edge connecting them.
Queries may contain 0 or more of each substructure's associated attributes.
```python
{
    "name": "pattern name",
    "pattern": [
        { /* node 0 query */ },
        { /* edge 0 → 1 query */ },
        { /* node 1 query */ },
        { /* edge 1 → 2 query */ },
        { /* node 2 query */ },
        …  
    ]
}
```
---

### Node Query Substructure Format
Each node query is a JSON object whose keys are **attribute names** and whose values are either:
- A single literal (e.g. `"lower"`)
- A set of literals to be treated as "OR" (e.g. `["lower","title"]`)
- *In the case of* `subtype_features`, a list/array all of whose items **must** be present on the node

Supported attributes:

#### **case**
- `"lower"`, `"upper"`, `"title"`
- matches spaCy’s `Token.is_lower`, `is_upper`, `is_title`

#### **type**
- `"currency"`
- `"url"`
- `"email"`
- `"word"`
- `"num"`
- `"whitespace"`
- `"punct"`
  - corresponds to the first‐level token type

#### **subtype\_features**
   Array of zero or more of:
    - `"left"`  (left punctuation)
    - `"right"` (right punctuation)
    - `"bracket"`
    - `"quote"`

#### **Morphological features**
   SpaCy's small model is the current default for extracting tokens' morphological features.
   For every key emitted by `t.morph.to_dict()`, you may query on that feature.
   The common keys are (but not limited to):
    - `Person`         -  e.g. `["1","2","3"]`
    - `Number`         -  e.g. `["Sing","Plur"]`
    - `Tense`          -  e.g. `["Past","Pres"]`
    - `Mood`           -  e.g. `["Ind","Imp","Sub"]`
    - `VerbForm`       -  e.g. `["Fin","Part","Inf"]`
    - `PronType`       -  e.g. `["Prs","Dem","Int","Rel"]`
    - `Degree`         -  e.g. `["Pos","Cmp","Sup"]`
    - `Case`           -  e.g. `["Nom","Acc","Dat","Gen"]`
    - `Gender`         -  e.g. `["Masc","Fem","Neut"]`
   For a complete list of morphological features available, see [universaldependencies.org's page](https://universaldependencies.org/u/feat/index.html)
   **Usage:** to require singular nouns, for example:
   ```json
   { "Number": ["Sing"] }
   ```
---

### Edge Query Substructure Format
Each edge query is a JSON-like object whose keys are **edge‐attribute names** and whose values are either a literal or a set/array of literals. Supported attributes:

#### **type**
- dependency‐relation label(s) (e.g. `"nsubj"`, `"dobj"`, `"prep"`, `"amod"`)
- if you supply a list, the edge’s `dep_` must be one of them

#### **rel_pos**
- `"before"`  (head → child in the token sequence)
- `"after"` (child → head in the token sequence)
---

### Example Query
```python
{
    "name": "pattern name",
    "pattern": [
        {
            # node 0 must be a verb in lowercase form
            "case": "lower",
            "POS": {"VERB"}
        },
        {
            # edge 0→1 must be either an nsubj or a prep relation
            "type": {"nsubj", "prep"}
        },
        {
            # node 1 must be a proper noun or an adposition token
            "POS": {"PROPN", "ADP"}
        }
    ]
}
```