import argparse
import adv_utils
import dataset
import plotting
from torch.utils.data import DataLoader, TensorDataset
from neural_network import NeuralNetwork, get_torch_device

# initialize parameters for different attacks
attacks = ['fgsm', 'ifgsm', 'pgd']
alpha, num_iter, restarts = 1e-2, 40, 10
epsilons = {'cat': 0.02, 'mnist': 0.04}
# testloader limit to do a quick test on some of the data
test_limit = 1000


def get_attacker(att_type, model, dataset_name, iterations=num_iter):
    """ instantiate requested attack with standard parameters. Returns class implementing the requested attack """
    epsilon = epsilons[dataset_name]
    if att_type == 'fgsm':
        return adv_utils.FGSM(model, epsilon=epsilon)
    elif att_type == 'ifgsm':
        return adv_utils.IFGSM(model, epsilon=epsilon, alpha=alpha, num_iter=iterations)
    elif att_type == 'pgd':
        return adv_utils.PGD(model, epsilon=epsilon, alpha=alpha, num_iter=iterations, restarts=restarts)
    else:
        print(f'Attack {att_type} not yet implemented')
        exit(1)


def get_data_and_model(file_name, dataset_name):
    device = get_torch_device()
    # get dataset and move data to correct device (cpu or cuda)
    data = dataset.get_dataset(dataset_name=dataset_name, bs=128)
    # used to store training loss and accuracy for each learning rate used
    model = NeuralNetwork.load(file_name)
    model.to(device)
    return model, data


def attack(model, data, att_type, file_name, dataset_name):
    device = get_torch_device()
    # get attacker
    attacker = get_attacker(att_type, model, dataset_name)
    print(f'Running {att_type} attack with l-inf bound {attacker.epsilon} against {file_name} model')
    # generate perturbations for test_limit samples. We limit to test_limit only for time requirements
    # if doing at home, you can compute over all testset
    perturbations = attacker.perturb_all(data.get_sub_testloader(test_limit), device=device)
    # get test_limit number of testset Xs and Ys
    test_x, test_y = data.get_test_data(test_limit)
    perturbed_test = test_x + perturbations

    # classification
    # get a few predictions for plotting later
    clean_predicted = model.pred(test_x)[:min(test_limit, 50)]
    adv_predicted = model.pred(perturbed_test)[:min(test_limit, 50)]
    # test on all adversarial examples
    adv_loader = DataLoader(TensorDataset(perturbed_test, test_y), batch_size=128)
    adv_accuracy = model.test(adv_loader)

    # plot clean vs perturbed images and respective classification
    test_x = data.unnormalize_data(data.unflatten(test_x))
    perturbed_test = data.unnormalize_data(data.unflatten(perturbed_test))
    plotting.plot_images(test_x, test_y, clean_predicted, 4, 8, grayscale=data.is_grayscale())
    plotting.plot_images(perturbed_test, test_y, adv_predicted, 4, 8, grayscale=data.is_grayscale())

    return adv_accuracy


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-file_name', help='model file name', required=True)
    parser.add_argument('-att_type', choices=attacks, help='type of attack between fgsm, pgd',
                        default='fgsm')
    parser.add_argument('-compare', action='store_true', help='compare all implemented attacks')
    args = parser.parse_args()
    # get dataset name from the name of the model
    dataset_name = args.file_name.split('_')[1]
    iterator = attacks if args.compare else [args.att_type]
    adv_accuracies = []
    model, data = get_data_and_model(args.file_name, dataset_name)
    clean_acc = model.test(data.testloader)
    # iterate over all requested attacks
    for attack_type in iterator:
        adv_accuracies.append(attack(model, data, attack_type, args.file_name, dataset_name=dataset_name))
    print(f'Clean accuracy: {clean_acc}')
    # print adversarial accuracy
    for attack_type, acc in zip(iterator, adv_accuracies):
        print(f'Adversarial accuracy for {attack_type}: {acc}')
