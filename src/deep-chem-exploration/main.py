import deepchem as dc


def main():

    # We are loading a Delaney solubility data set. 
    # We use GraphConv to represent (or featurize) our dataset a certain way
    task, datasets, transformers = dc.molnet.load_delaney(featurizer='ECFP')
    # task2, datasets2, transformers2 = dc.molnet.load_delaney(featurizer='MolGraphConv')
    task2, datasets2, transformers2 = dc.molnet.load_delaney(featurizer=dc.feat.MolGraphConvFeaturizer(use_edges=True))
    
    # Destructuring dataset creates a training, validation, and test sets
    trainingDataset, validationDataset, testDataset = datasets
    trainingDataset2, validationDataset2, testDataset2 = datasets2

    # print(f"trainingDataset shape -:\n {trainingDataset.shape()}")

    # Create a model - we will use Graph Convolutional Network
    # model = dc.models.GraphConvModel(n_tasks=1, mode='regression', dropout=0.2, batch_normalize=False)
    # model = dc.models.AttentiveFPModel(n_tasks=1, mode='regression', dropout=0.2)
    multitaskRegressorModel = dc.models.MultitaskRegressor(
        n_tasks=1,
        n_features=1024,
        layer_sizes=[500, 200],
        dropouts=0.2,
        mode='regression'
    )

    # Create another model - we will use Graph Convolutional Network
    GCNModel = dc.models.GCNModel(n_tasks=1, mode='regression')
    
    # Training GNNModular
    GCNModel.fit(trainingDataset2, nb_epoch=100)

    # Training a model GraphConvModel
    multitaskRegressorModel.fit(trainingDataset, nb_epoch=100)

    

    # Get our r2 score of the model
    metric = dc.metrics.Metric(dc.metrics.pearson_r2_score)
    print("Training set score (MTRM):", multitaskRegressorModel.evaluate(trainingDataset, [metric], transformers))
    print("Test set score (MTRM):", multitaskRegressorModel.evaluate(testDataset, [metric], transformers))
    print("Training set score (GCNModel):", GCNModel.evaluate(trainingDataset2, [metric], transformers2))
    print("Test set score (GCNModel):", GCNModel.evaluate(testDataset2, [metric], transformers2))

    # Solubilities
    solubilities = multitaskRegressorModel.predict_on_batch(testDataset.X[:10])

    # Multi-iterate 
    print("Predicted Solubility | Actual Solubility | Molecule (SMILES)")
    for molecule, solubility, testSolubility in zip(testDataset.ids, solubilities, testDataset.y):
        print(solubility, testSolubility, molecule)

if __name__ == "__main__":
    main()