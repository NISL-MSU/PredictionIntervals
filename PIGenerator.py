import os
import sys
import time
import utils
import torch
import pickle
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from models.NNModel import NNModel
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
# Functions needed for QD+
from models.aggregation_functions import _split_normal_aggregator  # You can comment it if you only want to test DualAQD


class PIGenerator:

    def __init__(self, dataset='Boston', method='AQD'):

        self.dataset = dataset
        self.method = method

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
        elif dataset == 'Yacht':
            D = pd.read_csv('Datasets//yacht_data.csv')
            self.X = D.drop(['Y'], axis=1).to_numpy()
            self.Y = D['Y'].to_numpy()
        elif dataset == 'Synth':
            self.X, self.Y, _, _ = utils.create_synth_data()
            self.X = np.reshape(self.X, (len(self.X), 1))

        self.kfold = KFold(n_splits=10, shuffle=True, random_state=13)  # Initialize kfold object

        # Load model
        print("Loading model...")
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = self.reset_model()

    def reset_model(self):
        return NNModel(device=self.device, nfeatures=self.X.shape[1], method=self.method)

    def train(self, crossval='10x1', batch_size=16, epochs=500, alpha_=0.01, printProcess=True):
        """Train using cross validation
        @param crossval: Type of cross-validation. Options: '10x1' or '5x2'
        @param batch_size: Mini batch size. It is recommended a small number, like 16
        @param epochs: Number of training epochs
        @param alpha_: Hyperparameter(s) used by the selected PI generation method
        @param printProcess: If True, print the training process (loss and validation metrics after each epoch)"""
        # Create lists to store metrics
        cvmse, cvpicp, cvmpiw, cvdiffs = [], [], [], []
        ypred, y_u, y_l, iterator = None, None, None, None

        # If the folder does not exist, create it
        folder = "CVResults//" + self.dataset + "//" + self.method
        if not os.path.exists("CVResults//" + self.dataset):
            os.mkdir("CVResults//" + self.dataset)
        if not os.path.exists(folder):
            os.mkdir(folder)

        if crossval == "10x1":
            iterator = self.kfold.split(self.X)
            print("Using 10x1 cross-validation for this dataset")
        elif crossval == "5x2":
            # Choose seeds for each iteration is using 5x2 cross-validation
            seeds = [13, 51, 137, 24659, 347, 436, 123, 64, 958, 234]
            iterator = enumerate(seeds)
            print("Using 5x2 cross-validation for this dataset")
        else:
            sys.exit("Only '10x1' and '5x2' cross-validation are permited.")

        ntrain = 1
        # Iterate through each partition
        for first, second in iterator:
            if crossval == '10x1':
                # Gets the list of training and test images using kfold.split
                train = np.array(first)
                test = np.array(second)
            else:
                # Split the dataset in 2 parts with the current seed
                train, test = train_test_split(range(len(self.X)), test_size=0.50, random_state=second)
                train = np.array(train)
                test = np.array(test)

            print("\n******************************")
            print("Training fold: " + str(ntrain))
            print("******************************")
            # Normalize using the training set
            Xtrain, means, stds = utils.normalize(self.X[train])
            Ytrain, maxs, mins = utils.minMaxScale(self.Y[train])
            Xval = utils.applynormalize(self.X[test], means, stds)
            Yval = utils.applyMinMaxScale(self.Y[test], maxs, mins)

            # Define path where the model will be saved
            filepath = folder + "//weights-" + self.method + "-" + self.dataset + "-" + str(ntrain)

            # Train model(s). AQD and MCDropout use one model while QD and QD+ use an ensemble of 5 models
            mse, PICP, MPIW = None, None, None
            if self.method in ['DualAQD', 'MCDropout']:
                m = 1
            else:
                m = 5
                filepath = [filepath] * m  # Array that will contain the filepath of each model of the ensemble
            for mi in range(m):
                if self.method in ['DualAQD', 'MCDropout']:
                    f = filepath
                else:
                    filepath[mi] = filepath[mi] + "-Model" + str(mi)
                    f = filepath[mi]
                # Train the model using the current training-validation split
                self.model = self.reset_model()
                _, _, _, mse, PICP, MPIW = self.model.trainFold(Xtrain=Xtrain, Ytrain=Ytrain, Xval=Xval, Yval=Yval,
                                                                batch_size=batch_size, epochs=epochs, filepath=f,
                                                                printProcess=printProcess, alpha_=alpha_,
                                                                yscale=[maxs, mins])

            # Run the model over the validation set 'MC-samples' times and Calculate PIs and metrics
            if self.method not in ['DualAQD']:  # DualAQD already performs validation and aggregation in "trainFold"
                [mse, PICP, MPIW, ypred, y_u, y_l] = self.calculate_metrics(Xval, Yval, maxs, mins, filepath)
            print('PERFORMANCE AFTER AGGREGATION:')
            print("Val MSE: " + str(mse) + " Val PICP: " + str(PICP) + " Val MPIW: " + str(MPIW))

            # Add metrics to the list
            cvmse.append(mse)
            cvpicp.append(PICP)
            cvmpiw.append(MPIW)

            # Plot synthetic dataset using results from the last validation fold
            if self.dataset == "Synth":  # and ntrain == 10:
                if self.method in ['DualAQD']:
                    self.model.loadModel(filepath)
                    yout = self.model.evaluateFoldUncertainty(valxn=Xval, maxs=None, mins=None, batch_size=32,
                                                              MC_samples=50)
                    yout = np.array(yout)
                    # Obtain upper and lower bounds
                    y_u = np.mean(yout[:, 0], axis=1)
                    y_l = np.mean(yout[:, 1], axis=1)
                    ypred = np.mean(yout[:, 2], axis=1)
                    ypred = utils.reverseMinMaxScale(ypred, maxs, mins)
                    y_u = utils.reverseMinMaxScale(y_u, maxs, mins)
                    y_l = utils.reverseMinMaxScale(y_l, maxs, mins)
                Xvalp = utils.reversenormalize(Xval, means, stds)
                _, _, P1, P2 = utils.create_synth_data(plot=True)
                diffs = 0
                for iv, x in enumerate(test):
                    ubound, lbound = P1[x], P2[x]
                    diffs += np.abs(ubound - y_u[iv]) + np.abs(y_l[iv] - lbound)
                cvdiffs.append(diffs)
                plt.scatter(Xvalp[:, 0], ypred, label='Predicted Data', s=24)
                plt.scatter(Xvalp[:, 0], y_u, label='Predicted Upper Bounds', s=24)
                plt.scatter(Xvalp[:, 0], y_l, label='Predicted Lower Bounds', s=24, c='gold')
                plt.legend(bbox_to_anchor=(1.06, 0.6), fontsize=18)
                plt.title(self.method, fontsize=24)
                plt.xlabel('x', fontsize=22)
                plt.ylabel('y', fontsize=22)
                plt.xticks(fontsize=22)
                plt.yticks(fontsize=22)

            # Reset all weights
            self.model = self.reset_model()

            ntrain += 1

        # Save metrics of all folds
        np.save(folder + '//validation_MSE-' + self.method + "-" + self.dataset, cvmse)
        np.save(folder + '//validation_MPIW-' + self.method + "-" + self.dataset, cvmpiw)
        np.save(folder + '//validation_PICP-' + self.method + "-" + self.dataset, cvpicp)
        if self.dataset == "Synth":
            np.save(folder + '//validation_DIFFS-' + self.method + "-" + self.dataset, cvdiffs)
        # Save metrics in a txt file
        file_name = folder + "//regression_report-" + self.method + "-" + self.dataset + ".txt"
        with open(file_name, 'w') as x_file:
            x_file.write("Overall MSE %.6f (+/- %.6f)" % (float(np.mean(cvmse)), float(np.std(cvmse))))
            x_file.write('\n')
            x_file.write("Overall PICP %.6f (+/- %.6f)" % (float(np.mean(cvpicp)), float(np.std(cvpicp))))
            x_file.write('\n')
            x_file.write("Overall MPIW %.6f (+/- %.6f)" % (float(np.mean(cvmpiw)), float(np.std(cvmpiw))))
            if self.dataset == "Synth":
                x_file.write('\n')
                x_file.write("Overall DIFF %.6f (+/- %.6f)" % (float(np.mean(cvdiffs)), float(np.std(cvdiffs))))

        return cvmse, cvmpiw, cvpicp

    def calculate_metrics(self, Xval, Yval, maxs, mins, filepath=None):
        """Calculate metrics using MC-Dropout to measure model uncertainty"""
        startsplit = time.time()

        if self.method in ['DualAQD', 'MCDropout']:  # These methods use only one model
            self.model.loadModel(filepath)  # Load model
            # Get outputs using trained model
            yout = self.model.evaluateFoldUncertainty(valxn=Xval, maxs=None, mins=None, batch_size=32, MC_samples=50)
            yout = np.array(yout)
            if self.method in ['AQD', 'DualAQD']:
                # Obtain upper and lower bounds
                y_u = np.mean(yout[:, 0], axis=1)
                y_l = np.mean(yout[:, 1], axis=1)
                # Obtain expected target estimates
                ypred = np.mean(yout[:, 2], axis=1)
                ypred = utils.reverseMinMaxScale(ypred, maxs, mins)
                y_u = utils.reverseMinMaxScale(y_u, maxs, mins)
                y_l = utils.reverseMinMaxScale(y_l, maxs, mins)
            else:
                # Load validation MSE
                with open(filepath + '_validationMSE', 'rb') as f:
                    val_MSE = pickle.load(f)
                # Obtain expected target estimates
                yout = utils.reverseMinMaxScale(yout, maxs, mins)
                ypred = np.mean(yout, axis=1)
                # Obtain upper and lower bounds
                model_uncertainty = np.std(yout, axis=1)
                y_u = ypred + 1.96 * np.sqrt(model_uncertainty ** 2 + val_MSE)
                y_l = ypred - 1.96 * np.sqrt(model_uncertainty ** 2 + val_MSE)
        else:  # QD and QD+ need to aggregate all PIs
            yout = np.zeros((len(Xval), 3, len(filepath)))
            y_l = np.zeros((len(Xval)))
            y_u = np.zeros((len(Xval)))
            ypred = np.zeros((len(Xval)))
            # Evaluate each of the models of the ensemble
            for mi in range(len(filepath)):
                self.model.loadModel(filepath[mi])  # Load model
                # Get outputs using trained model. MC_samples is set to 1 because they use only 1 forward pass
                yout[:, :, mi] = self.model.evaluateFoldUncertainty(valxn=Xval,
                                                                    maxs=None, mins=None, batch_size=5000,
                                                                    MC_samples=1)[:, :, 0]
            if self.method == 'QD':  # QD aggregation
                # Obtain upper and lower bounds
                y_u = np.mean(yout[:, 0], axis=1) + 1.96 * np.std(yout[:, 0], axis=1)
                y_l = np.mean(yout[:, 1], axis=1) - 1.96 * np.std(yout[:, 1], axis=1)
                # Obtain expected target estimates
                ypred = np.mean(yout[:, 2], axis=1)
            else:  # QD+ aggregation
                for s in range(len(yout)):  # Aggregate each sample
                    yp = yout[s, :, :].transpose()
                    yp[:, [1, 0]] = yp[:, [0, 1]]  # Swap columns: y_l, y_u, y_p
                    y_p_agg, y_l_agg, y_u_agg = _split_normal_aggregator(alpha=0.05, y_pred=yp, seed=7)
                    y_l[s] = y_l_agg
                    y_u[s] = y_u_agg
                    ypred[s] = y_p_agg
            # Reverse normalization
            ypred = utils.reverseMinMaxScale(ypred, maxs, mins)
            y_u = utils.reverseMinMaxScale(y_u, maxs, mins)
            y_l = utils.reverseMinMaxScale(y_l, maxs, mins)

        # Reverse normalization process
        Yval = utils.reverseMinMaxScale(Yval, maxs, mins)

        # Calculate MSE
        mse = utils.mse(Yval, ypred)
        # Calculate the coverage vector
        y_true = torch.from_numpy(Yval).float().to(self.device)
        y_ut = torch.from_numpy(y_u).float().to(self.device)
        y_lt = torch.from_numpy(y_l).float().to(self.device)
        K_U = torch.max(torch.zeros(y_true.size()).to(self.device), torch.sign(y_ut - y_true))
        K_L = torch.max(torch.zeros(y_true.size()).to(self.device), torch.sign(y_true - y_lt))
        K = torch.mul(K_U, K_L)
        # Calculate MPIW
        MPIW = torch.mean(y_ut - y_lt).item()
        # Calculate PICP
        PICP = torch.mean(K).item()

        endsplit = time.time()
        print("It took " + str(endsplit - startsplit) + " seconds to execute this batch")

        return [mse, PICP, MPIW, ypred, y_u, y_l]

    def tune(self):
        """Perform grid search for hyperparameters tuning"""
        # If the folder does not exist, create it
        folder = "TuningResults//" + self.dataset + "//" + self.method
        if not os.path.exists("TuningResults//" + self.dataset):
            os.mkdir("TuningResults//" + self.dataset)
        if not os.path.exists(folder):
            os.mkdir(folder)

        # Define number of epochs depending on the dataset
        epochs = 2000
        if self.dataset == 'Kin8nm':
            epochs = 400
        elif self.dataset == 'Naval':
            epochs = 400
        elif self.dataset == 'Boston':
            epochs = 3500
        elif self.dataset == 'Concrete':
            epochs = 2500
        elif self.dataset == 'Energy':
            epochs = 1500
        elif self.dataset == 'Yacht':
            epochs = 4500
        elif self.dataset == 'Wine':
            epochs = 500
        elif self.dataset == 'Power':
            epochs = 4000

        # Define hyperparameter space
        if self.method == 'DualAQD':
            beta_ = [0.05, 0.01, 0.005, 0.001]  # [0.01, 0.05, 0.1]
        elif self.method == 'QD+':
            lambda_1 = np.arange(0.2, 1, .1)
            lambda_2 = np.arange(0.2, .6, .1)
            beta_ = list(itertools.product(lambda_1, lambda_2))  # All possible parameters combinations
        else:  # QD
            beta_ = np.arange(0.021054, 0.05, 0.0025)

        # Start search for AQD, QD+, or QD
        count = 0
        results = []
        for bi in beta_:
            print("*****************************************")
            print("Trainining: " + str(count) + " / " + str(len(beta_)))
            print("*****************************************")
            iterator = self.kfold.split(self.X)
            count += 1
            ntrain = 1
            cvmse = []
            cvpicp = []
            cvmpiw = []
            for first, second in iterator:
                if ntrain >= 1:
                    train = np.array(first)
                    test = np.array(second)
                    print("\n******************************")
                    print("Starting fold: " + str(ntrain))
                    print("******************************")
                    # Define path where the temporal models will be saved
                    filepath = folder + "//weights-" + self.method + "-" + self.dataset + "-temp"
                    if self.method == 'DualAQD':
                        filepath = folder + "//weights-" + self.method + "-" + self.dataset + "-" + str(ntrain)
                    # Normalize using the training set
                    Xtrain, means, stds = utils.normalize(self.X[train])
                    Ytrain, maxs, mins = utils.minMaxScale(self.Y[train])
                    Xval = utils.applynormalize(self.X[test], means, stds)
                    Yval = utils.applyMinMaxScale(self.Y[test], maxs, mins)
                    metrics = None
                    # Train model(s). AQD and MCDropout use one model while QD and QD+ use an ensemble of 5 models
                    if self.method in ['AQD', 'DualAQD', 'MCDropout']:
                        m = 1
                    else:
                        m = 5
                        filepath = [filepath] * m  # Array that will contain the filepath of each model of the ensemble
                    for mi in range(m):
                        if self.method in ['AQD', 'DualAQD', 'MCDropout']:
                            f = filepath
                        else:
                            filepath[mi] = filepath[mi] + "-Model" + str(mi)
                            f = filepath[mi]
                        metrics = self.model.trainFold(Xtrain=Xtrain, Ytrain=Ytrain, Xval=Xval, Yval=Yval,
                                                       batch_size=16, epochs=epochs, filepath=f, printProcess=False,
                                                       alpha_=bi, yscale=[maxs, mins])
                        # Reset all weights
                        self.model = self.reset_model()

                    # Calculate metrics after including model uncertainty
                    if self.method not in ['AQD',
                                           'DualAQD']:  # AQD already performs validation and aggregation in "trainFold"
                        metrics = self.calculate_metrics(Xval, Yval, maxs, mins, filepath)
                    cvmse.append(metrics[3])
                    cvpicp.append(metrics[4])
                    cvmpiw.append(metrics[5])
                    print("############################")
                    print('FOLD PERFORMANCE')
                    print("############################")
                    print("Val MSE: " + str(metrics[3]) + " Val PICP: " + str(metrics[4]) + " Val MPIW: " + str(metrics[5]))
                    # Reset all weights
                    self.model = self.reset_model()
                ntrain += 1

            print("########################################")
            print('AVERAGE CV PERFORMANCE AFTER AGGREGATION')
            print("########################################")
            av_mse, av_picp, av_mpiw = np.mean(cvmse), np.mean(cvpicp), np.mean(cvmpiw)
            print("Val MSE: " + str(av_mse) + " Val PICP: " + str(av_picp) +
                  " Val MPIW: " + str(np.mean(av_mpiw)))
            results.append([np.mean(cvmse), np.mean(cvpicp), np.mean(cvmpiw)])
            # Save results to a txt file
            file_name = folder + "//tuning_results.txt"
            if self.method in ['DualAQD', 'QD']:
                with open(file_name, 'a') as x_file:
                    x_file.write("Beta %.6f%%: MSE %.6f%%, PICP %.6f%%, MPIW %.6f%%" %
                                 (float(bi), float(av_mse), float(av_picp), float(av_mpiw)))
                    x_file.write('\n')
            else:
                with open(file_name, 'a') as x_file:
                    x_file.write("Lambda_1 %.6f%% - Lambda_2 %.6f%%: MSE %.6f%%, PICP %.6f%%, MPIW %.6f%%" %
                                 (float(bi[0]), float(bi[1]), float(av_mse), float(av_picp), float(av_mpiw)))
                    x_file.write('\n')
        # Save results
        np.save(folder + '//tuning_results_' + self.method + '-' + self.dataset + '.npy', np.array(results))


if __name__ == '__main__':
    name = 'Boston'
    predictor = PIGenerator(dataset=name, method='DualAQD')
    predictor.train(crossval='10x1', batch_size=16, epochs=3000, printProcess=True, alpha_=0.01)
    name = 'Concrete'
    predictor = PIGenerator(dataset=name, method='DualAQD')
    predictor.train(crossval='10x1', batch_size=16, epochs=2500, printProcess=False, alpha_=0.01)
    name = 'Energy'
    predictor = PIGenerator(dataset=name, method='DualAQD')
    predictor.train(crossval='10x1', batch_size=16, epochs=3500, printProcess=False, alpha_=0.05)
    name = 'Kin8nm'
    predictor = PIGenerator(dataset=name, method='DualAQD')
    predictor.train(crossval='10x1', batch_size=16, epochs=1000, printProcess=False, alpha_=0.005)
    name = 'Power'
    predictor = PIGenerator(dataset=name, method='DualAQD')
    predictor.train(crossval='10x1', batch_size=16, epochs=4000, printProcess=False, alpha_=0.05)
    name = 'Yacht'
    predictor = PIGenerator(dataset=name, method='DualAQD')
    predictor.train(crossval='10x1', batch_size=16, epochs=4500, printProcess=True, alpha_=0.005)
    name = 'Synth'
    predictor = PIGenerator(dataset=name, method='DualAQD')
    predictor.train(crossval='10x1', batch_size=16, epochs=4500, printProcess=False, alpha_=0.005)

    # TUNING EXAMPLE
    name = 'Power'
    predictor = PIGenerator(dataset=name, method='DualAQD')
    predictor.tune()
