import itertools


SEQUENCE_DATASET_PATH = './data/coursera_bioinf_dataset_4.txt'

def read_dataset():

    
    print(f"Begin reading dataset from {SEQUENCE_DATASET_PATH}")

    # Read .txt
    with open(SEQUENCE_DATASET_PATH,'r') as file:
        text = file.read()

    print(f"Complete reading dataset from {SEQUENCE_DATASET_PATH}")

    return text


''' 
Pattern Count
    Input: 
        text - The dataset
        pattern - The pattern to count
    Return: 
        Int - Pattern Count
'''
def pattern_count(text, pattern):

    count = 0
    
    for i in range(len(text)-len(pattern) + 1):

        if text[i:i+len(pattern)] == pattern:
            
            print(f"Pattern '{pattern}' found in text")
            count += 1

    return count

'''
Frequent Words Count 
    Time Complexity:
        O(n² x k)
    Input:
        text - The dataset
        k - the length of k-mer
    Return: 
        Int - Set    
'''
def frequent_words(text, k):

    frequent_patterns = set()
    count_arr = []

    # Get a pattern thru the sliding window and perform pattern count on it
    for i in range(len(text) - k + 1):
        pattern = text[i:i+k]
        count_arr.append(pattern_count(text, pattern))    

    # Takes the max count
    max_count = max(count_arr)

    # Update Frequent Pattern set through evaluating if
    # the element in count_arr (corresponding to the sliding window text too)
    # reflects the most frequent pattern. If it is, add it into the set.
    for i in range(len(text) - k + 1):
        if count_arr[i] == max_count:
            frequent_patterns.add(text[i:i+k])
    
    return frequent_patterns



'''
Faster Frequent Words Count 
    Time Complexity:
        O(n x k)
    Input:
        text - The dataset
        k - the length of k-mer
    Return: 
        Int - Set    
'''
def faster_frequent_words(text, k):
    
    frequency_map = dict()
    n = len(text)

    for i in range(n-k+1):

        pattern = text[i:i+k]
        
        if pattern not in frequency_map:
            frequency_map[pattern] = 1
        else:
            frequency_map[pattern] += 1

        # Find patterns with max frequency
        max_count = max(frequency_map.values())
        frequent_patterns = {pattern for pattern, count in frequency_map.items() if count == max_count}
        
    return frequent_patterns



'''
Hamming Distance
    Input:
        p, q - two equal length strings
    Return:
        Int - number of mismatches
'''
def hamming_distance(p, q):
    
    count = 0

    # Compare using zip
    for char_p, char_q in zip(p, q):

        # Increase counter (hamming distance) if not equal
        if char_p != char_q:
            count +=1
    
    return count



'''
Distance Pattern to String
    Input: 
        pattern - k-mer pattern
        text - a DNA string
    Return:
        Int - minimum hamming distance of pattern across text
    
'''
def distance_pattern_to_string(pattern, text):
    
    k = len(pattern)
    min_distance = float('inf')

    for i in range (len(text) - k + 1):
        
        # The sliding window along the sequence (text)
        window = text[i:i+k]
        distance = hamming_distance(pattern, window)

        # Keep finding the minimum hamming distance as the window slides
        if distance < min_distance:
            min_distance = distance
    
    return min_distance



'''
Distance Pattern to Strings
    Input: 
        pattern - k-mer pattern
        dna - list of DNA string
    Return:
        Int - sum of minimum distances across all DNA strings
'''
def distance_pattern_to_strings(pattern, listOfSequences):
    
    total_distance = 0
    
    # Iterate each sequence in the list and find its distance
    for sequence in listOfSequences:
        
        # IMPORTANT
        # d(pattern, listOfSequence)
        # Sum of the minimum hamming distance between the pattern and each sequence
        total_distance += distance_pattern_to_string(pattern, sequence)
    
    return total_distance



'''
Median String Search
    Time Complexity:
        O(4^k tnk)
        k is -mer
        t is no. of sequence
        n is length of sequence
    Input:
        dna - list of DNA strings
        k   - length of k-mer
    Return: 
        k-mer string that is the median string

'''
def median_string_search(listOfSequences, k):
    
    alphabet = 'ACGT'
    median = None
    best_distance = float('inf')
    medianList = list()

    # itertools.product generates all possible combi of ACGT 
    for kmer_tuple in itertools.product(alphabet, repeat=k):

        # (AAA...AA, AAA...AC ... TTT...TT) 
        kmer = ''.join(kmer_tuple)
        distance = distance_pattern_to_strings(kmer, listOfSequences)

        if distance < best_distance:
            best_distance = distance
            median = kmer

            medianList = []
            medianList.append(kmer)

        elif distance == best_distance:
            medianList.append(kmer)

    return median, medianList
    


    


def main():

    ''' Sample Test '''
    # text = "GCGCG"
    # pattern = "GCG"
    # result = pattern_count(text, pattern)  # Returns 2

    ''' Read from dataset '''
    # pattern = "GCG"
    text = read_dataset()
    # text= 'ACGTTGCATGTCGCATGATGCATGAGAGCT'
    # result = pattern_count(text, pattern) # Returns 3
    # print(f"Pattern Count for '{pattern}': {result} ")

    ''' Frequent Patterns '''
    most_frequent_word = faster_frequent_words(text,4)
    print(f"Frequent Word: {most_frequent_word}\n")

    ''' Median String Search '''
    listOfSequences = [
        'CTCGATGAGTAGGAAAGTAGTTTCACTGGGCGAACCACCCCGGCGCTAATCCTAGTGCCC',
        'GCAATCCTACCCGAGGCCACATATCAGTAGGAACTAGAACCACCACGGGTGGCTAGTTTC',
        'GGTGTTGAACCACGGGGTTAGTTTCATCTATTGTAGGAATCGGCTTCAAATCCTACACAG'
    ]

    medianString, medianList = median_string_search(listOfSequences, 7)

    print(f"Median String {medianString}")
    print(f"Median String List {medianList}")

if __name__ == "__main__":
    main()
