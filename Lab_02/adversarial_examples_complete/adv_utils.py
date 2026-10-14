import torch
from tqdm import tqdm


class Attack:
    """ base class implementing different attacks """
    def __init__(self, model, targeted=False):
        self.target_model = model
        self.targeted = targeted

    def perturb(self, x, y, log=True):
        pass

    def perturb_all(self, dataloader, device):
        """ compute perturbation over all the samples in dataloader. Returns computed perturbations """
        deltas = []
        for x_batch, y_batch in dataloader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            deltas.append(self.perturb(x_batch, y_batch))
        return torch.cat(deltas, dim=0).detach().cpu()


class FGSM(Attack):
    def __init__(self, model, epsilon, targeted=False):
        super().__init__(model, targeted)
        self.type = 'fgsm'
        self.targeted = False
        self.epsilon = epsilon

    def perturb(self, x, y, log=True):
        delta = torch.zeros_like(x, requires_grad=True)
        predicted = self.target_model(x + delta)
        loss = self.target_model.loss_fn(predicted, y)
        loss.backward()
        adv_image = x.clone().detach() + self.epsilon * delta.grad.detach().sign()
        perturbation = adv_image.clamp(0, 1) - x
        return perturbation

class IFGSM(FGSM):
    def __init__(self, model, epsilon, alpha, num_iter, targeted=False):
        super().__init__(model, epsilon, targeted=targeted)
        self.type = 'ifgsm'
        self.alpha = alpha
        self.num_iter = num_iter

    def _perturb(self, x, y, base_instance=None, log=True):
        if base_instance is None:
            base_instance = x
        adv_data = x.clone().detach()
        iterator = range(self.num_iter)
        if log:
            # add logging through TQDM
            iterator = tqdm(iterator, desc='Generating IFGSM perturbation')
        for _ in iterator:
            adv_data.requires_grad = True
            predicted = self.target_model(adv_data)
            loss = self.target_model.loss_fn(predicted, y)
            grad = torch.autograd.grad(loss, adv_data, retain_graph=False, create_graph=False)[0]
            step = self.alpha * grad.sign()
            adv_data = adv_data.detach() + step
            delta = (adv_data - base_instance).clamp(-self.epsilon, self.epsilon)
            adv_data = (base_instance + delta).clamp(0, 1)
        delta = adv_data.detach() - base_instance
        return delta

    def perturb(self, x, y, log=True):
        return self._perturb(x, y, log=log)


class PGD(IFGSM):
    def __init__(self, model, epsilon, alpha, num_iter, restarts, targeted=False):
        super().__init__(model, epsilon, alpha, num_iter, targeted=targeted)
        self.type = 'pgd'
        self.restarts = restarts

    def perturb(self, x, y, log=True):
        """ computes adversarial example using PGD. Returns computed perturbation"""
        # data structures to keep track of max loss found and max perturbation to achieve the max loss
        max_loss = torch.zeros(y.shape[0]).to(y.device)
        max_delta = torch.zeros_like(x)

        iterator = range(self.restarts)
        if log:
            iterator = tqdm(iterator, desc='Generating PGD perturbation')
        for _ in iterator:
            delta = torch.empty_like(x).uniform_(-self.epsilon, self.epsilon)
            adv_data = (x.clone().detach() + delta).clamp(0, 1)
            delta = self._perturb(adv_data, y, base_instance=x, log=False)
            loss = self.target_model.loss_fn_indiv(self.target_model(x + delta), y).squeeze()
            max_delta[loss >= max_loss] = delta.detach()[loss >= max_loss]
            max_loss = torch.max(max_loss, loss)

        return max_delta

