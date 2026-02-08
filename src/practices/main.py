from childGenome import parseChildGenomeDataIntoCsv
import pandas as pd
import matplotlib.pyplot as plt

def main():
    print("Python Refresher Playground")

    childGenomeDf = parseChildGenomeDataIntoCsv()

    # Variants per chromosome (bar)
    churnGenomeByChromosome = childGenomeDf.groupby("chromosome").size()
    genotypeCount = childGenomeDf["genotype"].value_counts()

    fig, axes = plt.subplots(1,2,figsize=(12,5))

    # Variants per chromosome
    churnGenomeByChromosome.plot(kind = "bar", ax = axes[0], title = "Variants per chromosome")
    axes[0].set_xlabel("Chromosome")
    axes[0].set_ylabel("Count")
    
    # Genotype frequency (bar)
    genotypeCount.plot(kind="bar", ax = axes[1], title="Genotype frequency")
    axes[1].set_xlabel("Genotype")
    axes[1].set_ylabel("Count")



    # print("-- Row Counts of each column in DF --")
    # print(childGenomeDf.count())

    print("-- .head(): First five --")
    print(childGenomeDf.head())

    # print("-- Groupby Chromosome --")
    # print(churnGenomeByChromosome)

    print("-- iloc example of [2:4,1:3] slicing --")
    print(childGenomeDf.iloc[2:4,1:3])
    
    print("-- First 5 rows of genotype that are AA --")
    print(childGenomeDf[childGenomeDf["genotype"] == 'AA'].head())

    
    plt.tight_layout()
    plt.show()
    

if __name__ == "__main__":

    main()