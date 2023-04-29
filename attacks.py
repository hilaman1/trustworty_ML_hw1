import torch
import torch.nn as nn
import torch.nn.functional as F

class PGDAttack:
    """
    White-box L_inf PGD attack using the cross-entropy loss
    """
    def __init__(self, model, eps=8/255., n=50, alpha=1/255.,
                 rand_init=True, early_stop=True):
        """
        Parameters:
        - model: model to attack
        - eps: attack's maximum norm
        - n: max # attack iterations
        - alpha: step size at each iteration
        - rand_init: a flag denoting whether to randomly initialize
              adversarial samples in the range [x-eps, x+eps]
        - early_stop: a flag denoting whether to stop perturbing a
              sample once the attack goal is met. If the goal is met
              for all samples in the batch, then the attack returns
              early, before completing all the iterations.
        """
        self.model = model
        self.n = n
        self.alpha = alpha
        self.eps = eps
        self.rand_init = rand_init
        self.early_stop = early_stop
        self.loss_func = nn.CrossEntropyLoss(reduction='none')


    def execute(self, x, y, targeted=False):
        """
        Executes the attack on a batch of samples x. y contains the true labels
        in case of untargeted attacks, and the target labels in case of targeted
        attacks. The method returns the adversarially perturbed samples, which
        lie in the ranges [0, 1] and [x-eps, x+eps]. The attack optionally
        performs random initialization and early stopping, depending on the
        self.rand_init and self.early_stop flags.
        """
        # pick an initial point
        self.model.eval()
        self.model.requires_grad_(False)
        x_orig = x.clone().detach()
        x_adv = x.clone().detach()
        y = y.clone().detach()
        if self.rand_init:
            # create [x-eps, x+eps]
            delta = torch.zeros_like(x_orig).uniform_(-self.eps, self.eps)
            x_adv = torch.clamp(x + delta, 0, 1).detach()
        # loop until stopping condition is met:

        for i in range(self.n):
            x_adv.requires_grad_()
            batch_predictions = self.model(x_adv)
            loss = self.loss_func(batch_predictions, y)
            grad_sign = -1 if targeted else 1
            grad = torch.sign(torch.autograd.grad(loss.mean(), x_adv, retain_graph=False, create_graph=False)[0])

            if self.early_stop:
                # stops if all examples are miss classified
                if targeted:
                    misclassified = batch_predictions.max(1)[1] == y
                    if misclassified.sum().item() == 0:
                        return x_adv
                else:
                    misclassified = batch_predictions.max(1)[1] != y
                    if misclassified.sum().item() == 0:
                        return x_adv
            x_adv = x_adv.detach() + grad_sign * self.alpha * grad
            # assert [x-eps, x+eps]
            x_adv = torch.clamp(x_adv, x_orig - self.eps, x_orig + self.eps)
            # projection = min{max{0,x},1}
            x_adv = torch.clamp(x_adv, 0, 1).detach_()
        return x_adv




class NESBBoxPGDAttack:
    """
    Query-based black-box L_inf PGD attack using the cross-entropy loss,
    where gradients are estimated using Natural Evolutionary Strategies
    (NES).
    """
    def __init__(self, model, eps=8/255., n=50, alpha=1/255., momentum=0.,
                 k=200, sigma=1/255., rand_init=True, early_stop=True):
        """
        Parameters:
        - model: model to attack
        - eps: attack's maximum norm
        - n: max # attack iterations
        - alpha: PGD's step size at each iteration
        - momentum: a value in [0., 1.) controlling the "weight" of
             historical gradients estimating gradients at each iteration
        - k: the model is queries 2*k times at each iteration via
              antithetic sampling to approximate the gradients
        - sigma: the std of the Gaussian noise used for querying
        - rand_init: a flag denoting whether to randomly initialize
              adversarial samples in the range [x-eps, x+eps]
        - early_stop: a flag denoting whether to stop perturbing a
              sample once the attack goal is met. If the goal is met
              for all samples in the batch, then the attack returns
              early, before completing all the iterations.
        """
        self.model = model
        self.eps = eps
        self.n = n
        self.alpha = alpha
        self.momentum = momentum
        self.k = k
        self.sigma=sigma
        self.rand_init = rand_init
        self.early_stop = early_stop
        self.loss_func = nn.CrossEntropyLoss(reduction='none')

    def execute(self, x, y, targeted=False):
        """
        Executes the attack on a batch of samples x. y contains the true labels
        in case of untargeted attacks, and the target labels in case of targeted
        attacks. The method returns:
        1- The adversarially perturbed samples, which lie in the ranges [0, 1]
            and [x-eps, x+eps].
        2- A vector with dimensionality len(x) containing the number of queries for
            each sample in x.
        """
        num_queries = torch.zeros(x.shape[0])
        self.model.eval()
        self.model.requires_grad_(False)
        x_orig = x.clone().detach()
        x_adv = x.clone().detach()
        y = y.clone().detach()
        prev_grad = 0

        # pick an initial point
        if self.rand_init:
            # create [x-eps, x+eps]
            delta = torch.zeros_like(x_orig).uniform_(-self.eps, self.eps)
            x_adv = torch.clamp(x + delta, 0, 1).detach()
        # loop until stopping condition is met:
        for i in range(self.n):
            x_adv.requires_grad_()
            batch_predictions = self.model(x_adv)
            # estimate the gradient using NES
            grad = 0
            N = x.shape[1] * x.shape[2] * x.shape[3]
            noise = torch.randn(self.k, x.shape[0], N)
            deltas = noise.view(self.k, x.shape[0], x.shape[1], x.shape[2], x.shape[3])
            thetas = x.unsqueeze(0) + (self.sigma * deltas)
            outputs = self.model(thetas.view(-1, x.shape[1], x.shape[2], x.shape[3]))
            outputs = outputs.view(self.k, x.shape[0], -1)
            loss = torch.zeros((self.k, x.shape[0]))
            for i in range(self.k):
                loss[i] = self.loss_func(outputs[i], y)
            loss = loss.view(self.k, x.shape[0], 1, 1, 1)
            grad += (loss * deltas).mean(dim=0)

            thetas = x.unsqueeze(0) - (self.sigma * deltas)
            outputs = self.model(thetas.view(-1, x.shape[1], x.shape[2], x.shape[3]))
            outputs = outputs.view(self.k, x.shape[0], -1)
            loss = torch.zeros((self.k, x.shape[0]))
            for i in range(self.k):
                loss[i] = self.loss_func(outputs[i], y)
            loss = loss.view(self.k, x.shape[0], 1, 1, 1)
            grad -= (loss * deltas).mean(dim=0)
            estimated_grad = grad / (2 * self.sigma)

            # for i in range(self.k):
            #     # positive samples
            #     thetas = x + (self.sigma * deltas[i])
            #     outputs = self.model(thetas)
            #     loss = self.loss_func(outputs, y)
            #     grad += loss.reshape(-1, 1, 1, 1) * deltas[i]
            #     # negative samples
            #     thetas = x - (self.sigma * deltas[i])
            #     outputs = self.model(thetas)
            #     loss = self.loss_func(outputs, y)
            #     grad -= loss.reshape(-1, 1, 1, 1) * deltas[i]
            # estimated_grad = grad / (2 * self.sigma)
            grad_sign = -1 if targeted else 1
            # Compute delta for the next update
            estimated_grad = prev_grad * self.momentum + (1 - self.momentum) * estimated_grad
            prev_grad = estimated_grad
            if self.early_stop:
                # stops if all examples are miss classified
                if targeted:
                    if (batch_predictions.max(1)[1] == y).sum().item() == 0:
                        break
                else:
                    if (batch_predictions.max(1)[1] != y).sum().item() == 0:
                        break
            num_queries += 2 * self.k
            x_adv = x_adv + grad_sign * self.alpha * torch.sign(estimated_grad)
            # assert [x-eps, x+eps]
            x_adv = torch.clamp(x_adv, x - self.eps, x + self.eps)
            # projection = min{max{0,x},1}
            x_adv = torch.clamp(x_adv, 0, 1)
        return x_adv , num_queries


class PGDEnsembleAttack:
    """
    White-box L_inf PGD attack against an ensemble of models using the
    cross-entropy loss
    """
    def __init__(self, models, eps=8/255., n=50, alpha=1/255.,
                 rand_init=True, early_stop=True):
        """
        Parameters:
        - models (a sequence): an ensemble of models to attack (i.e., the
              attack aims to decrease their expected loss)
        - eps: attack's maximum norm
        - n: max # attack iterations
        - alpha: PGD's step size at each iteration
        - rand_init: a flag denoting whether to randomly initialize
              adversarial samples in the range [x-eps, x+eps]
        - early_stop: a flag denoting whether to stop perturbing a
              sample once the attack goal is met. If the goal is met
              for all samples in the batch, then the attack returns
              early, before completing all the iterations.
        """
        self.models = models
        self.n = n
        self.alpha = alpha
        self.eps = eps
        self.rand_init = rand_init
        self.early_stop = early_stop
        self.loss_func = nn.CrossEntropyLoss()

    def execute(self, x, y, targeted=False):
        """
        Executes the attack on a batch of samples x. y contains the true labels
        in case of untargeted attacks, and the target labels in case of targeted
        attacks. The method returns the adversarially perturbed samples, which
        lie in the ranges [0, 1] and [x-eps, x+eps].
        """
        pass # FILL ME
