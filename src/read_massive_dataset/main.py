import time
import pandas as pd
import random
import os
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

PATH = './data/massiveDataset.csv'
NEWPATH = './data/massiveDatasetMultipliedNonBatch.csv'
NEWBATCHPATH = './data/massiveDatasetMultipliedBatch.csv'
PARQUETPATH = './data/massiveDataset.parquet'
NEWPARQUETPATH = './data/massiveDatasetMultiplied.parquet'
MULTIPROCESSPATH = './data/massiveDatasetMultiprocess.csv'



def createDummyDataset():

    dfColumns = [
        "column1",
        "column2",
        "column3"
    ]
    table = []

    for i in range(0,10000000):
        row = []
        for j in range(0,3):
            row.append(random.randint(1,10))
        
        table.append(row)

    df = pd.DataFrame(table,columns=dfColumns)
    print("Massive dataset created with shape:{}".format(df.shape))
    df.to_csv(PATH)
    print("Massive dataset saved to path: {}".format(PATH))



def readCsv():

    start = time.time()
    df = pd.read_csv(PATH)
    print("(1) Time taken to read massive dataset: {}secs".format(time.time() - start))
    return df


def multiplyColumnBy10(df):

    start = time.time()
    df["column1"] *=10
    print("(2) Time taken to Column1 x 10: {}secs".format(time.time() - start))
    return df

    
"""
Process CSV in chunks to reduce memory usage and improve performance
"""
def batchProcessing(chunkSize=2000000):

    startTime = time.time()

    # Read and process in chunks
    firstChunk = True

    for chunk in pd.read_csv(PATH, chunksize=chunkSize):
        
        # Process the chunk
        chunk["column1"] *=10

        if firstChunk:
            # Write into new CSV
            chunk.to_csv(NEWBATCHPATH, mode='w',index=False)
        else:
            # Appand into existing CSV
            chunk.to_csv(NEWBATCHPATH, mode='a',index=False, header=False)

    print("Batch processing completed in: {}secs".format(time.time() - startTime))

    
"""
Process CSV in chunks to reduce memory usage and improve performance
and write into Parquet
"""
def batchProcessingParquet(chunkSize=2000000):
    
    startTime = time.time()

    chunks = []

    for chunk in pd.read_csv(PATH, chunksize=chunkSize):
        chunk["column1"] *= 10
        chunks.append(chunk)

    result = pd.concat(chunks, ignore_index=True)
    result.to_parquet(NEWPARQUETPATH, index=False)

    print("Batch processing (Parquet) completed in: {}secs".format(time.time() - startTime))


'''
(Multiprocessing) Process Chunk
'''
def processChunk(args: tuple) -> tuple:

    chunk, chunkIndex = args
    chunk["column1"] *= 10
    return (chunk, chunkIndex)

'''
(Multiprocessing) Process CSV with multiprocessing
'''
def multiprocessBatchProcessing(chunkSize=500000, maxWorkers= 4):

    startTime = time.time()

    # Read chunks into a list with their index
    chunksWithIndex = []
    
    # Assign index to each chunk for ordering later
    for i, chunk in enumerate(pd.read_csv(PATH, chunksize=chunkSize)):
        chunksWithIndex.append((chunk,i))

    # Process chunks in parallel
    with ProcessPoolExecutor(max_workers=maxWorkers) as executor:
        results = list(executor.map(processChunk, chunksWithIndex))

    # Sort by original index to maintain order 
    # lambda x: x[1] -> x==tuple, x[1] second value in tuple
    # Eg. (chunk_df,2) -> x[1] == 2
    # Sort by the second index of the tuple
    results.sort(key=lambda x: x[1])

    # Convert sorted results into array
    processedChunks = [chunk for chunk, _ in results]
    
    # Concatenate and write into CSV
    resultDf = pd.concat(processedChunks, ignore_index=True)
    resultDf.to_csv(MULTIPROCESSPATH, index=False)

    total_time = time.time() - startTime
    print("Multiprocess batch processing completed in: {}secs".format(total_time))
    

def main():
    
    datasetFileAlreadyExist = os.path.isfile(PATH)
    
    if not datasetFileAlreadyExist:
        createDummyDataset()
    
    '''
    (1. Read-Process-Write) Without batch processing and Parquet
    '''
    # (Read) Read CSV in pandas dataframe
    df = readCsv()

    # Create a copy of the dataframe for performance verification
    dfCopy = df.copy()

    # (Process) Apply x10 on all rows in Column1   
    updatedDf = multiplyColumnBy10(dfCopy)

    # (Write) Apply updated dataframe into a new CSV file
    start = time.time()
    updatedDf.to_csv(NEWPATH)  
    print("(3) Time taken to write new massive dataset: {}secs".format(time.time() - start))

    '''
    (2. Read-Process-Write) With batch processing only
    '''
    batchProcessing()

    '''
    (3. Read-Process-Write) With batch processing and Parquet
    '''
    batchProcessingParquet()

    '''
    (4. Read-Process-Write) With batch processing in multiprocessing approach
    '''
    multiprocessBatchProcessing(chunkSize=500000, maxWorkers=4)

    # (Debug to verify performance)
    print(df.head())
    print(updatedDf.head())

'''
Outputs
(1) Time taken to read massive dataset: 1.8767564296722412secs
(2) Time taken to Column1 x 10: 0.015341758728027344secs
(3) Time taken to write new massive dataset: 6.208568572998047secs
Batch processing completed in: 6.519100904464722secs
Batch processing (Parquet) completed in: 2.584967851638794secs
Multiprocess batch processing completed in: 9.313894987106323secs (!!!)

Remark - 

Multiprocessing takes longest because of overhead costs that outweigh the benefits:

Your bottleneck is I/O (reading/writing files), NOT computation:

Let's break down the time:

Reading CSV: ~1.9s
Processing (multiply): ~0.016s (extremely fast)
Writing CSV: ~6.2s
Why multiprocessing is slower:

Process creation overhead - Spawning 4 separate Python processes takes time (~1-2s)
Data serialization - Must pickle/unpickle data between processes (expensive for large DataFrames)
Memory copying - Each process gets its own copy of the data
All chunks loaded at once - Uses way more memory than sequential chunking
Final concatenation - Combining all processed chunks back together

Sequential chunking: 6.59s (just read → process → write)

Multiprocessing:
  - Load all chunks: ~2s
  - Spawn processes: ~1s  
  - Process in parallel: 0.016s × 4 cores = ~0.004s (tiny!)
  - Serialize/deserialize: ~2s
  - Concatenate results: ~1s
  - Write final CSV: ~6s
  Total: ~12s

'''



if __name__ == "__main__":

    main()