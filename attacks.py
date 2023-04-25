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
        if self.rand_init:
            # create [x-eps, x+eps]
            delta = torch.zeros_like(x).uniform_(-self.eps, self.eps)
        else:
            delta = torch.zeros_like(x) - x
        # loop until stopping condition is met:
        x.requires_grad_()
        for i in range(self.n):
            batch_predictions = self.model(x + delta)
            loss = self.loss_func(batch_predictions, y)
            grad = torch.sign(torch.autograd.grad(loss.sum(), x, retain_graph=True)[0])
            delta = delta + self.alpha * grad
            # assert [x-eps, x+eps]
            delta = torch.clamp(delta, -self.eps, self.eps)
            # projection = min{max{0,x},1}
            delta = torch.clamp(x + delta, 0, 1) - x
            if self.early_stop:
                # stops if all examples are miss classified
                adversary = x + delta
                if targeted:
                    if (batch_predictions.max(1)[1] == y).sum().item() == 0:
                        return adversary
                else:
                    if (batch_predictions.max(1)[1] != y).sum().item() == 0:
                        return adversary
        return x + delta

        # # pick an initial point
        # if self.rand_init:
        #     # create [x-eps, x+eps]
        #     delta = x + torch.zeros_like(x).uniform_(-self.eps, self.eps)
        # else:
        #     delta = x
        # # loop until stopping condition is met:
        # for i in range(self.n):
        #     delta.requires_grad = True
        #     self.model.zero_grad()
        #     batch_predictions = self.model(delta)
        #     loss = self.loss_func(batch_predictions, y)
        #     loss.backward()
        #     x_adv = delta + self.alpha * delta.grad.sign()
        #     # assert [x-eps, x+eps]
        #     eta = torch.clamp(x_adv - x, min=-self.eps, max=self.eps)
        #     # projection = min{max{0,x},1}
        #     delta = torch.clamp(x + eta, min=0, max=1).detach_()
        #     if self.early_stop:
        #         adversary = x + delta
        #         if torch.all(adversary >= 0) and torch.all(adversary <= 1):
        #             return adversary
        # return x + delta



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
        pass # FILL ME


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
