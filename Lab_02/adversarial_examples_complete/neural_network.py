import itertools

import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm

import constants

act_dict = {'relu': nn.ReLU, 'sfotmax': nn.Softmax, 'sigmoid': nn.Sigmoid}


__torch_device = None

def get_torch_device():
    global __torch_device

    if __torch_device is not None:
        return __torch_device

    if  torch.backends.mps.is_available():
        print('Using MPS device for hardware acceleration')
        __torch_device = 'mps'
    elif torch.cuda.is_available():
        print('Using CUDA device for hardware acceleration')
        __torch_device = 'cuda'
    else:
        print('Hardware acceleration not available, using CPU')
        __torch_device = 'cpu'
    return __torch_device


class NeuralNetwork(nn.Module):
    def __init__(self, shape, activations, lr, loss_fn_avg=None, loss_fn_individual=None, optim=torch.optim.SGD):
        """
        @param shape: list with size of each layer
        @param activations: type of activation function for each layer (all neurons in a layer use same activation)
        @param lr: learning rate
        @loss_fn_avg: loss functions averaging over all elements of a minibatch to use. For training
        @loss_fn_individual: loss functions computing individual, per-sample losses to use. Useful for PGD restarts
        @optim: optimization algorithm to use
        """
        super().__init__()
        self.shape = shape
        self.activations = activations
        self.lr = lr
        # create list of layers
        layer_list = [nn.Linear(x, y) for x, y in zip(shape[:-2], shape[1:-1])]
        # create list of activations
        activations = [act_dict[a]() for a in activations]
        # create list of layers alternating with activations: [layer1, act1, ..., layerN, actN]
        layer_act_list = list(itertools.chain.from_iterable(zip(layer_list, activations)))
        # add last layer and, if output is binary, sigmoid at last layer to squash to [0,1] interval
        layer_act_list.append(nn.Linear(shape[-2], shape[-1]))
        if shape[-1] == 1:
            layer_act_list.append(nn.Sigmoid())
        # create sequential module. It chains all the layers and activations sequentially
        self.model = nn.Sequential(*layer_act_list)
        # initialize optimizer
        self.optimizer = optim(self.parameters(), lr=lr, weight_decay=1e-4)
        # initialize loss_fn_avg (average over batch) and loss_fn_individual (individual for each sample in the batch)
        # individual loss_fn does not average, useful to keep track of best perturbation found with PGD
        if loss_fn_avg is None:
            self.loss_fn = F.binary_cross_entropy if shape[-1] == 1 else F.cross_entropy
        else:
            self.loss_fn = loss_fn_avg
        if loss_fn_individual is not None:
            self.loss_fn_indiv = nn.BCELoss(reduction='none') if shape[-1] == 1 else nn.CrossEntropyLoss(reduction='none')
        else:
            self.loss_fn_indiv = loss_fn_individual
        self.device = get_torch_device()

    def forward(self, x):
        """ implement forward method by calling the forward method of the Sequential module self.model """
        return self.model(x)

    def get_class(self, x):
        if self.shape[-1] == 1:
            # only one output neuron -> binary classification
            return (x >= 0.5) * 1.
        else:
            # multiclass classification, return index of max prediction
            return x.argmax(dim=1)

    def pred(self, x):
        x = x.to(self.device)
        return self.get_class(self(x))

    def fit(self, trainloader, epochs, log_interval=10, attacker=None):
        """ fit (train) the model
        @param trainloader: training set in form of dataloader
        @param epochs: number of iterations over the dataset to train
        @param log_interval: interval between output logs
        @param attacker: adv. attacker to perform adversarial training
        @return: loss and accuracy history
        """
        loss_history = []
        acc_history = []
        desc = 'Training neural network' + f' with {attacker.type }adversarial training' if attacker is not None else ''
        for epoch in tqdm(range(epochs), desc=desc):
            accuracy, running_loss = 0., 0.
            for x_batch, y_batch in trainloader:
                x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
                # check if we're performing adv. training
                if attacker is not None:
                    # use half clean samples half adv samples
                    perturbation = attacker.perturb(x_batch, y_batch, log=False)
                    x_batch[:len(x_batch / 2)] = x_batch[:len(x_batch / 2)] + perturbation[:len(x_batch / 2)]
                # get predictions for the batch
                pred = self(x_batch)
                # compute loss
                loss = self.loss_fn(pred, y_batch)
                # backpropagate the loss and compute gradients using autograd
                loss.backward()
                # perform optimization step (update weights with computed gradients)
                self.optimizer.step()
                # zero-out gradients before next iteration
                self.optimizer.zero_grad()
                running_loss += loss.item()
                accuracy += (self.get_class(pred) == y_batch).sum().item() / x_batch.shape[0]

            if epoch % log_interval == 0:
                accuracy /= len(trainloader)
                print(f'Epoch {epoch}: loss {running_loss/len(trainloader)}; train accuracy {accuracy}')
                loss_history.append((epoch, running_loss))
                acc_history.append((epoch, accuracy))
        return loss_history, acc_history

    def test(self, testloader):
        correct, total = 0., 0.
        # we don't want to compute any gradiends here, we're just testing
        with torch.no_grad():
            for x_batch, y_batch in testloader:
                x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
                # predict outputs
                pred = self.pred(x_batch)
                # get number of correct predictions
                correct += (pred == y_batch).sum().item()
                total += y_batch.shape[0]
        # return accuracy
        return correct/total

    def save(self, fname):
        constants.model_dir.mkdir(parents=True, exist_ok=True)
        to_save = {'shape': self.shape,
                   'activations': self.activations,
                   'lr': self.lr,
                   'loss': (self.loss_fn, self.loss_fn_indiv),
                   'state_dict': self.state_dict()}
        torch.save(to_save, constants.model_dir / fname)

    @staticmethod
    def load(fname):
        device = get_torch_device()
        # weights_only = False needed to load old model
        saved = torch.load(constants.model_dir / fname, map_location=device, weights_only=False)
        model = NeuralNetwork(saved['shape'], saved['activations'], saved['lr'], saved['loss'][0], saved['loss'][1])
        model.load_state_dict(saved['state_dict'])
        return model
