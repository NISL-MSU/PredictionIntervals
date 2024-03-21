import torch
import pickle
from src.utils import *
from sklearn.model_selection import KFold
from src.DualAQD.models.NNModel import NNModel


class Trainer:
    def __init__(self, X: np.array, Y: np.array, method: str = 'DualAQD'):
        """
        Train a PI-generation NN using DualAQD
        :param X: Explainable variables. 2-D numpy array, shape (#samples, #features)
        :param Y: Response variable. 1-D numpy array, shape (#samples, #features)
        :param method: PI-generation method. Options: 'DualAQD' or 'MCDropout'
        """
        # Class variables
        self.X, self.Y, self.method = X, Y, method
        if self.X.ndim == 1:
            self.n_features = 1
        else:
            self.n_features = self.X.shape[1]
        self.name = 'temp_' + method  # Save the model in a temp folder
        # Configure model
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = self.reset_model()
        self.means, self.stds, self.maxs, self.mins = None, None, None, None  # Will save statistics of the dataset
        self.f = None

    def reset_model(self):
        return NNModel(device=self.device, nfeatures=self.n_features, method=self.method)

    def init_kfold(self):
        # Initialize kfold object
        kfold = KFold(n_splits=10, shuffle=True, random_state=7)
        iterator = kfold.split(self.X)
        return iterator

    def train(self, batch_size=32, epochs=500, eta_=0.01, printProcess=True, normData: bool = True, scratch: bool = True):
        """Train using 90% of the data for training and 10% for validation
        :param batch_size: Mini batch size. It is recommended a small number, like 16
        :param epochs: Number of training epochs
        :param eta_: Scale factor used to update the self-adaptive coefficient lambda (Eq. 6 of the paper)
        :param printProcess: If True, print the training process (loss and validation metrics after each epoch)
        :param normData: If True, apply z-score normalization to the inputs and min-max normalization to the outputs
        :param scratch: If True, re-train all networks from scratch
        """
        # If the temp folder does not exist, create it
        root = get_project_root()
        folder = os.path.join(root, "src//DualAQD//models//temp_weights")
        if not os.path.exists(os.path.join(root, "src//DualAQD//models//temp_weights")):
            os.mkdir(os.path.join(root, "src//DualAQD//models//temp_weights"))
        if not os.path.exists(folder):
            os.mkdir(folder)

        iterator = self.init_kfold()  # Use only first partition
        for first, second in iterator:
            train, test = np.array(first), np.array(second)
            # Normalization
            if normData:
                Xtrain, self.means, self.stds = normalize(self.X[train])
                Ytrain, self.maxs, self.mins = minMaxScale(self.Y[train])
                Xval = applynormalize(self.X[test], self.means, self.stds)
                Yval = applyMinMaxScale(self.Y[test], self.maxs, self.mins)
            else:
                Xtrain, self.means, self.stds = self.X[train], None, None
                Ytrain, self.maxs, self.mins = self.Y[train], None, None
                Xval, Yval = self.X[test], self.Y[test]

            # Train the model using the current training-validation split
            self.f = folder + "//weights-NN-" + self.name
            if scratch or not os.path.exists(self.f):
                _, _, _, val_mse, PICP, MPIW = self.model.trainFold(Xtrain=Xtrain, Ytrain=Ytrain, Xval=Xval, Yval=Yval,
                                                                    batch_size=batch_size, epochs=epochs, filepath=self.f,
                                                                    printProcess=printProcess, alpha_=eta_,
                                                                    yscale=[self.maxs, self.mins])
                # Run the model over the validation set 'MC-samples' times and Calculate PIs and metrics
                if self.method not in ['DualAQD']:  # DualAQD already performs validation and aggregation in "trainFold"
                    [val_mse, PICP, MPIW, _, _, _] = self.calculate_metrics(Xval, Yval)
            else:  # Or just load pre-trained NN
                self.model.loadModel(path=self.f)
                # Evaluate model on the validation set
                [val_mse, PICP, MPIW, _, _, _] = self.calculate_metrics(Xval, Yval)
            print('PERFORMANCE AFTER AGGREGATION:')
            print("Val MSE: " + str(val_mse) + " Val PICP: " + str(PICP) + " Val MPIW: " + str(MPIW))
            break

    def calculate_metrics(self, Xval, Yval):
        """Calculate metrics using a PI-generation method to quantify uncertainty
        :param Xval: Validation data
        :param Yval : Validation targets
        """
        self.model.loadModel(self.f)  # Load model
        # Get outputs using trained model
        yout = self.model.evaluateFoldUncertainty(valxn=Xval, maxs=None, mins=None, batch_size=32, MC_samples=50)
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

        # Reverse normalization process
        Yval = reverseMinMaxScale(Yval, self.maxs, self.mins)
        # Calculate MSE
        val_mse = mse(Yval, ypred)
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

        return [val_mse, PICP, MPIW, ypred, y_u, y_l]


if __name__ == '__main__':

    Xin, Yin, _, _ = create_synth_data(n=1000, plot=False)
    trainer = Trainer(Xin, Yin)
    trainer.train(printProcess=False)
