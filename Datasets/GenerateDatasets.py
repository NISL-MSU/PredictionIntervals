import numpy as np
import pandas as pd
from utils import create_synth_data


class DataLoader:
    """Class used to load or generate datasets used for equation learning"""

    def __init__(self, dataset: str = 'S0'):
        """
        Initialize DataLoader class
        :param dataset: Dataset name
        """

        # Read dataset
        if dataset == 'Boston':
            D = pd.read_csv('Datasets//boston_housing_data.csv')
            self.X = D.drop(['MDEV'], axis=1).to_numpy()
            self.Y = D['MDEV'].to_numpy()
        elif dataset == 'Wine':
            D = pd.read_csv('Datasets//winequality-red.csv')
            self.X = D.drop(['quality'], axis=1).to_numpy()
            self.Y = D['quality'].to_numpy()
        elif dataset == 'Concrete':
            D = pd.read_csv('Datasets//concrete_data.csv')
            self.X = D.drop(['concrete_compressive_strength'], axis=1).to_numpy()
            self.Y = D['concrete_compressive_strength'].to_numpy()
        elif dataset == 'Energy':
            D = pd.read_csv('Datasets//energy_data.csv').drop(['Y2'], axis=1)
            self.X = D.drop(['Y1'], axis=1).to_numpy()[:768, :-2]
            self.Y = D['Y1'].to_numpy()[:768]
        elif dataset == 'Kin8nm':
            D = pd.read_csv('Datasets//kin8nm_data.csv')
            self.X = D.drop(['y'], axis=1).to_numpy()
            self.Y = D['y'].to_numpy()
        elif dataset == 'Naval':
            D = pd.read_csv('Datasets//naval_data.csv').drop(['GTTurb'], axis=1)
            self.X = D.drop(['GTComp'], axis=1).to_numpy()
            self.Y = D['GTComp'].to_numpy()
        elif dataset == 'Power':
            D = pd.read_csv('Datasets//power_data.csv')
            self.X = D.drop(['PE'], axis=1).to_numpy()
            self.Y = D['PE'].to_numpy()
        elif dataset == 'Protein':
            D = pd.read_csv('Datasets//protein_data.csv')
            self.X = D.drop(['RMSD'], axis=1).to_numpy()
            self.Y = D['RMSD'].to_numpy()
        elif dataset == 'Yacht':
            D = pd.read_csv('Datasets//yacht_data.csv')
            self.X = D.drop(['Y'], axis=1).to_numpy()
            self.Y = D['Y'].to_numpy()
        elif dataset == 'Year':
            D = pd.read_csv('Datasets//yearMSD_data.csv')
            self.X = D.iloc[:, 1:].to_numpy()
            self.Y = D.iloc[:, 0].to_numpy()
        elif dataset == 'Synth':
            self.X, self.Y, _, _ = create_synth_data()
            self.X = np.reshape(self.X, (len(self.X), 1))


if __name__ == '__main__':
    dataLoader = DataLoader(dataset='Synth')
    X, Y = dataLoader.X, dataLoader.Y
