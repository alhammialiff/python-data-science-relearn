import time
import pandas as pd
import random
import os

PATH = './data/massiveDataset.csv'
NEWPATH = './data/massiveDatasetMultipliedNonBatch.csv'
NEWBATCHPATH = './data/massiveDatasetMultipliedBatch.csv'
PARQUETPATH = './data/massiveDataset.parquet'
NEWPARQUETPATH = './data/massiveDatasetMultiplied.parquet'


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

    # (Debug to verify performance)
    print(df.head())
    print(updatedDf.head())

'''
Outputs
(1) Time taken to read massive dataset: 1.8662135601043701secs
(2) Time taken to Column1 x 10: 0.013091564178466797secs
(3) Time taken to write new massive dataset: 6.172140121459961secs
Batch processing completed in: 6.488108158111572secs
Batch processing (Parquet) completed in: 2.5882418155670166secs

'''



if __name__ == "__main__":

    main()