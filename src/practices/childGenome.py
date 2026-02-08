import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

def parseChildGenomeDataIntoCsv():

    # Resolve path
    csvPath = Path(__file__).resolve().parents[2] / "data" / "Child 1 Genome.csv"

    df = pd.read_csv(
        csvPath,
    )

    return df



