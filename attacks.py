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
        if self.rand_init:
            # create [x-eps, x+eps]
            delta = torch.zeros_like(x).uniform_(-self.eps, self.eps)
            x_adv = torch.clamp(x + delta, 0, 1)

        # loop until stopping condition is met:

        for i in range(self.n):
            x_adv.requires_grad_()
            batch_predictions = self.model(x_adv)
            loss = self.loss_func(batch_predictions, y)
            grad_sign = -1 if targeted else 1
            grad = torch.sign(torch.autograd.grad(loss.mean(), x_adv, retain_graph=False, create_graph=False)[0])

            if self.early_stop:
                max_class = batch_predictions.max(1)
                target_reached_flag = max_class.indices == y if targeted else max_class.indices != y
                grad = (1 - target_reached_flag.int()).reshape(-1, 1, 1, 1) * grad
                if torch.all(target_reached_flag):
                    return x_adv
            x_adv = x_adv + grad_sign * self.alpha * grad
            # assert [x-eps, x+eps]
            x_adv = torch.clamp(x_adv, x-self.eps, x+self.eps)
            # projection = min{max{0,x},1}
            x_adv = torch.clamp(x_adv, 0, 1)
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

        # pick an initial point
        if self.rand_init:
            # create [x-eps, x+eps]
            delta = torch.zeros_like(x).uniform_(-self.eps, self.eps)
            delta = torch.clamp(x + delta, 0, 1) - x
        else:
            delta = torch.zeros_like(x)
        # loop until stopping condition is met:
        for i in range(self.n):
            batch_predictions = self.model(x + delta)
            # estimate the gradient
            noise_pos = torch.normal(0, 1, x.shape)
            theta_pos = x + self.sigma * noise_pos
            # Query the model to compute gradients using antithetic sampling
            positive_predictions = self.model(theta_pos)
            positive_theta_loss = self.loss_func(positive_predictions, y)

            noise_neg = (-1) * noise_pos
            theta_neg = x + self.sigma * noise_neg
            negative_predictions = self.model(theta_neg)
            negative_theta_loss = self.loss_func(negative_predictions, y)

            all_theta = torch.cat([positive_theta_loss.T * noise_pos.T, negative_theta_loss.T * noise_neg.T], dim=0)
            approx_gradient = all_theta.mean() / self.sigma

            delta = delta + self.alpha * torch.sign(approx_gradient)
            # assert [x-eps, x+eps]
            delta = torch.clamp(delta, -self.eps, self.eps)
            # projection = min{max{0,x},1}
            delta = torch.clamp(x + delta, 0, 1) - x

            # Compute delta for the next update
            delta = self.momentum * delta + (1 - self.momentum) * delta
            if self.early_stop:
                # stops if all examples are miss classified
                if targeted:
                    if (batch_predictions.max(1)[1] == y).sum().item() == 0:
                        num_queries += 2 * self.k
                        break
                else:
                    if (batch_predictions.max(1)[1] != y).sum().item() == 0:
                        num_queries += 2 * self.k
                        break
            num_queries += 2 * self.k
        return x + delta , num_queries


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
