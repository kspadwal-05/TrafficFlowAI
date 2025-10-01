import difflib

def parse_file(file_path):
    """
    Reads the given file and returns a list of tuples: (row_id, sentence).
    Assumes each line is formatted as "ID: sentence".
    """
    # Initialize id variable
    id = 371 #start on row 371 of sheet
    sentences = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                line = "N/A"  # Skip blank lines
            # Split at the first colon to get the ID and the sentence.
            sentences.append((id, line))
            id += 1
    print(f"Done!")
    return sentences

def find_best_match(bad_sentence, good_sentences):
    """
    Finds the sentence from good_sentences that is most similar to bad_sentence.
    Returns a tuple (best_good, best_ratio). If the best_ratio is below the threshold,
    it still returns the best match.
    """
    best_match = None
    best_ratio = 0
    for good_id, good_sentence in good_sentences:
        ratio = difflib.SequenceMatcher(None, bad_sentence, good_sentence).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_match = (good_id, good_sentence)
    return best_match, best_ratio

def main():
    # Define file paths
    good_file_path = 'goodFile.txt'
    bad_file_path = 'badFile.txt'
    output_file_path = 'outputFile.txt'
    replace_file_path = 'outputReplaceOnly.txt'
    
    # Parse files
    print(f"Parsing good streets... ", end='')
    good_sentences = parse_file(good_file_path)
    print(f"Parsing bad streets... ", end='')
    bad_sentences = parse_file(bad_file_path)
    
    # Open output file for writing
    
    with open(replace_file_path, 'w', encoding='utf-8') as rep_f:
        print(f"Writing to copy & paste file... ", end='\r')
         # For each bad sentence, find the best matching good sentence
        for bad_id, bad_sentence in bad_sentences:
            (bad_id, good_sentence), ratio = find_best_match(bad_sentence, good_sentences)
            # Check if similarity is at least 60%
            if ratio >= 0.7:
                out_line = f"{good_sentence}\n"
            else:
                # Optionally, handle no good match (you can modify this behavior as needed)
                out_line = f"{bad_sentence}\n" #-- [No close match found]\n"
            rep_f.write(out_line)
    print(f"Writing to copy & paste file... Done!")
    print(f"Output results only written to {replace_file_path}")


if __name__ == "__main__":
    main()
