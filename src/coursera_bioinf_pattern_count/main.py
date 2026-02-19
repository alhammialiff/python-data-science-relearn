
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


if __name__ == "__main__":
    main()
