import torch
import pickle
from ..utils import *
from ..models.NNModel import NNModel


class Trainer:
    def __init__(self, X: np.array, Y: np.array, Xval: np.array, Yval: np.array, method: str = 'DualAQD', normData: bool = True):
        """
        Train a PI-generation NN using DualAQD
        :param X: Input data (explainable variables). 2-D numpy array, shape (#samples, #features)
        :param Y: Target data (response variable). 1-D numpy array, shape (#samples, #features)
        :param Xval: Validation input data. 2-D numpy array, shape (#samples, #features)
        :param Yval: Validation target data. 1-D numpy array, shape (#samples, #features)
        :param method: PI-generation method. Options: 'DualAQD' or 'MCDropout'
        :param normData: If True, apply z-score normalization to the inputs and min-max normalization to the outputs
        """
        # Class variables
        self.method = method
        if X.ndim == 1:
            self.n_features = 1
        else:
            self.n_features = X.shape[1]
        self.name = 'temp_' + method  # Save the model in a temp folder
        # Configure model
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = self.reset_model()
        self.f = self._set_folder()

        # Normalization
        self.normData = normData
        if normData:
            self.X, self.means, self.stds = normalize(X)
            self.Y, self.maxs, self.mins = minMaxScale(Y)
            self.Xval = applynormalize(Xval, self.means, self.stds)
            self.Yval = applyMinMaxScale(Yval, self.maxs, self.mins)
        else:
            self.X, self.means, self.stds = X, None, None
            self.Y, self.maxs, self.mins = Y, None, None
            self.Xval, self.Yval = Xval, Yval

    def reset_model(self):
        return NNModel(device=self.device, nfeatures=self.n_features, method=self.method)

    def _set_folder(self):
        root = get_project_root()
        folder = os.path.join(root, "PredictionIntervals//models//temp_weights")
        if not os.path.exists(os.path.join(root, "PredictionIntervals//models//temp_weights")):
            os.mkdir(os.path.join(root, "PredictionIntervals//models//temp_weights"))
        if not os.path.exists(folder):
            os.mkdir(folder)
        return folder + "//weights-NN-" + self.name

    def train(self, batch_size=16, epochs=1000, eta_=0.01, printProcess=False):
        """Train method
        :param batch_size: Mini batch size. It is recommended a small number, like 16
        :param epochs: Number of training epochs
        :param eta_: Scale factor used to update the self-adaptive coefficient lambda (Eq. 6 of the paper)
        :param printProcess: If True, print the training process (loss and validation metrics after each epoch)
        """
        _, _, _, val_mse, PICP, MPIW = self.model.trainFold(Xtrain=self.X, Ytrain=self.Y, Xval=self.Xval, Yval=self.Yval,
                                                            batch_size=batch_size, epochs=epochs, filepath=self.f,
                                                            printProcess=printProcess, alpha_=eta_,
                                                            yscale=[self.maxs, self.mins])
        # Run the model over the validation set 'MC-samples' times and Calculate PIs and metrics
        if self.method not in ['DualAQD']:  # DualAQD already performs validation and aggregation in "trainFold"
            [val_mse, PICP, MPIW, _, _, _] = self.evaluate(self.Xval, self.Yval)
        print('PERFORMANCE AFTER AGGREGATION:')
        print("Val MSE: " + str(val_mse) + " Val PICP: " + str(PICP) + " Val MPIW: " + str(MPIW))

    def _apply_normalization(self, Xeval, Yeval):
        Xeval = applynormalize(Xeval, self.means, self.stds)
        if Yeval is not None:
            Yeval = applyMinMaxScale(Yeval, self.maxs, self.mins)
        return Xeval, Yeval

    def evaluate(self, Xeval, Yeval=None, normData: bool = False):
        """Calculate metrics using a PI-generation method to quantify uncertainty
        :param Xeval: Evaluation data
        :param Yeval : Optional. Evaluation targets
        :param normData: If True, apply the same normalization that was applied to the training set
        """
        if normData:
            Xeval, Yeval = self._apply_normalization(Xeval, Yeval)

        self.model.loadModel(self.f)  # Load model
        # Get outputs using trained model
        yout = self.model.evaluateFoldUncertainty(valxn=Xeval, maxs=None, mins=None, batch_size=32, MC_samples=50)
        yout = np.array(yout)
        if self.method == 'DualAQD':
            # Obtain upper and lower bounds
            y_u = np.mean(yout[:, 0], axis=1)
            y_l = np.mean(yout[:, 1], axis=1)
            # Obtain expected target estimates
            ypred = np.mean(yout[:, 2], axis=1)
            ypred = reverseMinMaxScale(ypred, self.maxs, self.mins)
            y_u = reverseMinMaxScale(y_u, self.maxs, self.mins)
            y_l = reverseMinMaxScale(y_l, self.maxs, self.mins)
        else:
            # Load validation MSE
            with open(self.f + '_validationMSE', 'rb') as f:
                val_MSE = pickle.load(f)
            # Obtain expected target estimates
            yout = reverseMinMaxScale(yout, self.maxs, self.mins)
            ypred = np.mean(yout, axis=1)
            # Obtain upper and lower bounds
            model_uncertainty = np.std(yout, axis=1)
            y_u = ypred + 1.96 * np.sqrt(model_uncertainty ** 2 + val_MSE)
            y_l = ypred - 1.96 * np.sqrt(model_uncertainty ** 2 + val_MSE)

        if Yeval is not None:
            # Reverse normalization process
            Yeval = reverseMinMaxScale(Yeval, self.maxs, self.mins)
            # Calculate MSE
            val_mse = mse(Yeval, ypred)
            # Calculate the coverage vector
            y_true = torch.from_numpy(Yeval).float().to(self.device)
            y_ut = torch.from_numpy(y_u).float().to(self.device)
            y_lt = torch.from_numpy(y_l).float().to(self.device)
            K_U = torch.max(torch.zeros(y_true.size()).to(self.device), torch.sign(y_ut - y_true))
            K_L = torch.max(torch.zeros(y_true.size()).to(self.device), torch.sign(y_true - y_lt))
            K = torch.mul(K_U, K_L)
            # Calculate MPIW
            MPIW = torch.mean(y_ut - y_lt).item()
            # Calculate PICP
            PICP = torch.mean(K).item()
            return [val_mse, PICP, MPIW, ypred, y_u, y_l]
        else:  # If the targets of the evaluation data are not known, just return predictions
            return ypred, y_u, y_l


if __name__ == '__main__':

    Xin, Yin, _, _ = create_synth_data(n=1000, plot=False)
    indices = np.arange(len(Yin))
    np.random.shuffle(indices)
    Xtrain, Ytrain = Xin[indices[:int(len(Yin) / 3)]], Yin[indices[:int(len(Yin) / 3)]]
    xval, yval = Xin[indices[int(len(Yin) / 3):int(len(Yin) / 3 * 2)]], Yin[indices[int(len(Yin) / 3):int(len(Yin) / 3 * 2)]]
    xtest, ytest = Xin[indices[int(len(Yin) / 3 * 2):]], Yin[indices[int(len(Yin) / 3 * 2):]]

    trainer = Trainer(Xtrain, Ytrain, xval, yval)
    trainer.train(printProcess=False, epochs=1000)

    mset, PICPt, MPIWt, ypredt, y_uT, y_lT = trainer.evaluate(xtest, ytest, normData=True)
