import spacy
from smied.FramenetSpacySRL import FrameNetSpaCySRL

if __name__ == "__main__":
    # Method 1: Use as standalone class
    print("METHOD 1: Standalone Usage")
    print("-" * 40)
    srl = FrameNetSpaCySRL(use_wordnet_expansion=True, min_confidence=0.2)

    # Test sentences
    test_sentences = [
        "John gave Mary a book in the library.",
        "The chef cooked dinner for the guests with great skill.",
        "She quickly ran to the store to buy milk.",
        "The student solved the problem by thinking creatively."
    ]

    for sentence in test_sentences:
        # Process the sentence
        doc = srl.process_text(sentence)

        # Print visualization
        print(srl.visualize_frames(doc))
        print()

    # Method 2: Use as SpaCy pipeline component
    print("\nMETHOD 2: SpaCy Pipeline Component")
    print("-" * 40)

    # Load SpaCy and add the component
    nlp = spacy.load("en_core_web_sm")
    nlp.add_pipe("framenet_srl", config={"min_confidence": 0.5})

    # Process text through the pipeline
    for sentence in test_sentences:
        doc = nlp("The company announced its new product at the conference.")

        # Access frame information
        for frame in doc._.frames:
            print(f"\nFrame: {frame.name}")
            print(f"\tPredicate: '{frame.target.text}'")
            print(f"\tConfidence: {frame.confidence:.2f}")
            print(f"\tFrame entities:")
            for element in frame.elements:
                print(f"\t\t{element.name}: '{element.span.text}' ({element.fe_type})")

    # Method 3: Quick one-liner for testing
    print("\nMETHOD 3: Quick Testing")
    print("-" * 40)
    quick_srl = FrameNetSpaCySRL()
    for sentence in test_sentences:
        doc = quick_srl.process_text(sentence)

        summary = quick_srl.get_frame_summary(doc)
        print(f"\nFound {summary['statistics']['total_frames']} frames "
            f"with {summary['statistics']['total_elements']} elements")
        
        for frame_data in summary['frames']:
            print(f"Frame: {frame_data['frame']}")
            print(f"\tPredicate: '{frame_data['predicate']}'")
            print(f"\tConfidence: {frame_data['confidence']:.2f}")
            print(f"\tFrame entities:")
            for element in frame_data['elements']:
                print(f"\t\t{element['role']}: '{element['text']}' ({element["type"]})")
