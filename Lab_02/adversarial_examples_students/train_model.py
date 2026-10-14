import argparse
import dataset
import torch
import matplotlib.pyplot as plt

from adversarial_examples.neural_network import get_torch_device
from neural_network import NeuralNetwork
import adv_attacks

optimizer = torch.optim.SGD

train_setups = {
    'mnist': {
        'learning_rates': [0.0075],
        'epochs': 100,
        'nn_shape': [20, 7, 5],
        # 'nn_shape': [40, 25, 15],
        'nn_activations': ['relu', 'relu', 'relu'],
        'log_interval': 10},
    'cat': {
        'learning_rates': [0.0075],
        'epochs': 1600,
        'nn_shape': [20, 7, 5],
        'nn_activations': ['relu', 'relu', 'relu'],
        'log_interval': 100}
}


def plot_loss_acc(train_results):
    loss_fig, loss_ax = plt.subplots()
    acc_fig, acc_ax = plt.subplots()
    for lr in train_results:
        loss_history, acc_history = train_results[lr]
        loss_ax.set_title('Train Loss History')
        loss_ax.plot(*list(zip(*loss_history)), label=f'lr: {lr}')
        loss_ax.legend(loc='best')
        acc_ax.set_title('Train Accuracy History')
        acc_ax.plot(*list(zip(*acc_history)), label=f'lr: {lr}')
        acc_ax.legend(loc='best')
    plt.show()


def test_model(fname, dataset_name):
    device = get_torch_device()
    print(f'Running model on {device}')
    data = dataset.get_dataset(dataset_name=dataset_name, bs=256)
    model = NeuralNetwork.load(fname)
    model.to(device)
    accuracy = model.test(data.testloader)
    print(f'Test accuracy: {accuracy}')


def train_model(dataset_name, adv_training=None):
    device = get_torch_device()
    print(f'Running model on {device}')
    # get training setup based on dataset
    train_setup = train_setups[dataset_name]
    nn_shape = train_setup['nn_shape']
    nn_activations = train_setup['nn_activations']
    learning_rates = train_setup['learning_rates']
    epochs = train_setup['epochs']
    log_interval = train_setup['log_interval']

    # get dataset
    data = dataset.get_dataset(dataset_name=dataset_name, bs=128)
    # update input feature size and last layer size based on dataset
    nn_shape.insert(0, data.flattened_shape)
    if data.class_num == 2:
        # if binary classification, last layer is a single neuron
        nn_shape.append(1)
    else:
        # if multiclass classification, last layer has as many neurons as classes
        nn_shape.append(data.class_num)

    # used to store training loss and accuracy for each learning rate used
    train_results = {}
    # used to store the trained model for each learning rate used
    models = {}
    attacker = None
    for lr in learning_rates:
        model_name_suffix = ''
        # initialize the model with given shape, activations and learning rate, and move to cpu/gpu
        model = NeuralNetwork(shape=nn_shape, activations=nn_activations, lr=lr, optim=optimizer)
        model.to(device)
        # instantiate attacker if doing adversarial training
        if adv_training:
            attacker = adv_attacks.get_attacker(adv_training, model, dataset_name, iterations=6)
            epochs = int(epochs * 1.2)
            # add _robust suffix to saved model name
            model_name_suffix = f'_{attacker.type}_robust'
        # train (fit) the model
        loss_history, acc_history = model.fit(data.trainloader, epochs, log_interval=log_interval, attacker=attacker)
        train_results[str(lr)] = (loss_history, acc_history)
        models[str(lr)] = model
        model.save(f'model_{dataset_name}_lr{lr}{model_name_suffix}')
    # plotting
    plot_loss_acc(train_results)

    # testing
    for lr in models:
        accuracy = models[lr].test(data.testloader)
        print(f'Learning rate {lr} test accuracy: {accuracy}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-test', help='model file name to test with')
    parser.add_argument('-dataset', choices=['cat', 'mnist'], help='dataset to use. Either cats or mnist',
                        default='cat')
    parser.add_argument('-adv', choices=adv_attacks.attacks,
                        help='perform adversarial training using the given algorithm')
    parser.add_argument('-draw', action='store_true', help='plot loss surface of the model')
    args = parser.parse_args()
    if args.test is not None:
        test_model(args.test, args.dataset)
    else:
        train_model(args.dataset, args.adv)
