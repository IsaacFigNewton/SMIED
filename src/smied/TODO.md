# SMIED TODO: Multi-Frame, Multi-Sense Pathfinding Fixes

## Problem Analysis

### Current Issues:
1. **Wrong Word Sense Selection**: System selects inappropriate senses (e.g., "cat" as person vs. feline)
2. **Single Frame Exploration**: Only explores one semantic frame per predicate synset
3. **Early Success Prevention**: Once any path is found, better interpretations aren't explored

### Root Cause:
- FramenetSpacySRL identifies multiple frames but only "best frame" is used
- SemanticDecomposer stops exploring after first success with wrong senses
- No mechanism to rank paths by semantic coherence

## Proposed Fix: Replace Current Frame Strategy

### 1. Modify FramenetSpacySRL._get_frames_for_predicate()
**Replace single best-frame selection with multi-frame ranking**

```python
def _get_frames_for_predicate(self, pred_span: Span) -> List[Tuple[str, float]]:
    """Get ALL relevant frames with coherence scores (not just best)"""
    frames = set()
    head = pred_span.root
    pos_map = {'VERB': 'v', 'NOUN': 'n', 'ADJ': 'a', 'ADV': 'adv'}
    if head.pos_ in pos_map:
        fn_pos = pos_map[head.pos_]
        key = (head.lemma_.lower(), fn_pos)
        if key in self.lexical_unit_cache:
            frames.update(self.lexical_unit_cache[key])

    # Return ALL frames with coherence scores (don't select "best")
    frame_scores = []
    for frame_name in frames:
        score = self._score_frame_coherence(frame_name, pred_span)
        frame_scores.append((frame_name, score))
    
    # Return sorted by coherence (best first) but keep all
    return sorted(frame_scores, key=lambda x: x[1], reverse=True)
```

### 2. Modify SemanticDecomposer._find_framenet_subject_predicate_paths()
**Explore multiple frames instead of stopping at first success**

```python
def _find_framenet_subject_predicate_paths(self, subject_synsets, predicate_synset, ...):
    """Try ALL frame interpretations, not just first successful one"""
    framenet_paths = []
    
    # Get ALL frames for predicate (not just best)
    predicate_text = f"{predicate_synset.lemmas()[0].name()} {predicate_synset.definition()}"
    pred_doc = self.framenet_srl.process_text(predicate_text)
    
    # Get all relevant frames with scores
    frame_scores = self.framenet_srl._get_frames_for_predicate(pred_doc[0:1])  # span
    
    # Try EACH frame interpretation
    for frame_name, coherence_score in frame_scores:
        frame_paths = self._find_paths_via_frame(
            subject_synsets, predicate_synset, frame_name, coherence_score, ...
        )
        framenet_paths.extend(frame_paths)
    
    # Rank all paths by semantic coherence score
    return sorted(framenet_paths, key=lambda p: p.coherence_score, reverse=True)
```

### 3. Add Semantic Coherence Scoring
**Replace arbitrary "first success" with coherence-based ranking**

```python
def _score_frame_coherence(self, frame_name: str, subject_word: str, 
                          predicate_word: str, object_word: str) -> float:
    """Score semantic coherence of frame interpretation with triple"""
    frame = self.frame_cache[frame_name]
    score = 0.0
    
    # Check if frame elements align with expected roles
    core_elements = [fe for fe, data in frame.FE.items() if data.coreType == "Core"]
    
    # Agent/Experiencer frames expect animate subjects
    if any(fe in ["Agent", "Experiencer"] for fe in core_elements):
        if self._is_animate_word(subject_word):
            score += 0.3
    
    # Theme/Patient frames expect concrete objects  
    if any(fe in ["Theme", "Patient", "Goal"] for fe in core_elements):
        if self._is_concrete_word(object_word):
            score += 0.3
            
    # Lexical unit match
    if predicate_word in frame_name.lower() or any(
        predicate_word in lu_name.lower() for lu_name in frame.lexUnit
    ):
        score += 0.4
        
    return score

def _is_animate_word(self, word: str) -> bool:
    """Check if word typically refers to animate entities"""
    synsets = wn.synsets(word, pos=wn.NOUN)
    for synset in synsets[:2]:  # Check top 2 senses
        # Check if any hypernym indicates animacy
        for hypernym in synset.hypernyms():
            if 'person' in hypernym.name() or 'animal' in hypernym.name():
                return True
    return False
```

### 4. Modify SemanticDecomposer.find_connected_shortest_paths()
**Collect all interpretations before selecting best**

```python
def find_connected_shortest_paths(self, subject_word, predicate_word, object_word, ...):
    """Collect ALL possible interpretations, then select most coherent"""
    
    # Try each predicate synset with ALL its frame interpretations
    all_interpretation_paths = []
    
    for pred_synset in predicate_synsets:
        # Get paths for ALL frame interpretations (not just first success)
        subject_paths = self._find_subject_to_predicate_paths(
            subject_synsets, pred_synset, g, get_new_beams_fn,
            beam_width, max_depth, relax_beam, max_results_per_pair, len_tolerance
        )
        
        object_paths = self._find_predicate_to_object_paths(
            pred_synset, object_synsets, g, get_new_beams_fn,
            beam_width, max_depth, relax_beam, max_results_per_pair, len_tolerance
        )
        
        # Score ALL valid combinations by semantic coherence
        for subj_path in subject_paths:
            for obj_path in object_paths:
                coherence = self._score_path_coherence(
                    subj_path, obj_path, pred_synset, subject_word, predicate_word, object_word
                )
                combined_length = len(subj_path) + len(obj_path) - 1
                all_interpretation_paths.append((
                    subj_path, obj_path, pred_synset, coherence, combined_length
                ))
    
    # Select best interpretation by coherence, breaking ties with path length
    if all_interpretation_paths:
        best = max(all_interpretation_paths, key=lambda x: (x[3], -x[4]))  # max coherence, min length
        return best[0], best[1], best[2]
    
    return None, None, None
```

## Implementation Steps (Minimal Changes)

### Step 1: Modify Existing Methods (No New Classes)
- [ ] Update `FramenetSpacySRL._get_frames_for_predicate()` to return ALL frames with scores
- [ ] Add `FramenetSpacySRL._score_frame_coherence()` method
- [ ] Add `FramenetSpacySRL._is_animate_word()` and `_is_concrete_word()` helpers

### Step 2: Update SemanticDecomposer Strategy 
- [ ] Modify `_find_framenet_subject_predicate_paths()` to try all frames
- [ ] Update `find_connected_shortest_paths()` to collect all interpretations first
- [ ] Add `_score_path_coherence()` method for ranking complete paths

### Step 3: Remove Early Termination Logic
- [ ] Remove "only if no paths found" conditions in strategy cascade
- [ ] Let all strategies run and compete based on coherence scores
- [ ] Replace "first success wins" with "best coherence wins"

## Expected Outcomes

After implementation:

1. **"cat-chase-mouse"** should find: `cat.n.01 (feline) -> chase.v.01 (hunt) -> mouse.n.01 (rodent)`
2. **"bird-fly-sky"** should find: `bird.n.01 (animal) -> fly.v.01 (travel through air) -> sky.n.01 (atmosphere)`
3. **Wrong interpretations** (like "guy-chase romantically-timid person") will be outranked by coherent ones
4. **All frame strategies** will run and compete rather than early termination on first success