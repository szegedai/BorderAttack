# Copyright (c) 2020-present, Francesco Croce
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree
#

import time
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from autoattack.other_utils import L0_norm, L1_norm, L2_norm  # (and you ALSO need L1_projection)
from autoattack.checks import check_zero_gradients
from autoattack.autopgd_base import L1_projection

class BorderAPGD():
    """
    AutoPGD
    https://arxiv.org/abs/2003.01690

    :param predict:       forward pass function
    :param norm:          Lp-norm of the attack ('Linf', 'L2', 'L0' supported)
    :param n_restarts:    number of random restarts
    :param n_iter:        number of iterations
    :param eps:           bound on the norm of perturbations
    :param seed:          random seed for the starting point
    :param loss:          loss to optimize ('ce', 'dlr' supported)
    :param eot_iter:      iterations for Expectation over Trasformation
    :param rho:           parameter for decreasing the step size
    """

    def __init__(
            self,
            predict,
            n_iter=100,
            norm='Linf',
            n_restarts=1,
            eps=None,
            seed=0,
            loss='combined',
            eot_iter=1,
            rho=.75,
            topk=None,
            verbose=False,
            device=None,
            use_largereps=False,
            is_tf_model=False,
            logger=None,
            losses=None,
            bs = None,
            model_name=None):
        """
        AutoPGD implementation in PyTorch
        """
        
        self.model = predict
        self.n_iter = n_iter
        self.eps = eps
        self.norm = norm
        self.n_restarts = n_restarts
        self.seed = seed
        self.loss = loss
        self.eot_iter = eot_iter
        self.thr_decr = rho
        self.topk = topk
        self.verbose = verbose
        self.device = device
        self.use_rs = True
        #self.init_point = None
        self.use_largereps = use_largereps
        #self.larger_epss = None
        #self.iters = None
        self.n_iter_orig = n_iter + 0
        self.eps_orig = eps + 0.
        self.is_tf_model = is_tf_model
        self.y_target = None
        self.logger = logger
        self.losses = losses
        self.model_name = model_name

        self.restart_num = n_restarts
        self.bs = 1

        print(self.csv_path)
        self.clear_csv = True
        if self.csv_path and self.clear_csv:
            # Truncate the file a single time at construction (not per batch)
            with open(self.csv_path, 'w', newline=''):
                pass

        assert self.norm in ['Linf', 'L2', 'L1']
        assert not self.eps is None

        ### set parameters for checkpoints
        self.n_iter_2 = max(int(0.22 * self.n_iter), 1)
        self.n_iter_min = max(int(0.06 * self.n_iter), 1)
        self.size_decr = max(int(0.03 * self.n_iter), 1)

    def get_border_mask(self, shape, thickness=1):
        """
        Create a binary mask where the border pixels are set to 1 and the rest are 0.
        Thickness can be specified.
        """
        b, c, h, w = shape
        mask = torch.zeros((1, c, h, w), device=self.device)

        # Top and bottom borders
        mask[:, :, :thickness, :] = 1
        mask[:, :, -thickness:, :] = 1

        # Left and right borders
        mask[:, :, :, :thickness] = 1
        mask[:, :, :, -thickness:] = 1

        return mask

    def apply_border_constraints(self, x, x_adv):
        delta = x_adv - x
        delta = delta * self.border_mask
        delta = delta.clamp(-self.eps, self.eps)
        x_adv = x + delta
        x_adv = x_adv.clamp(0., 1.)
        assert x_adv.min() >= 0.0 and x_adv.max() <= 1.0, \
            "apply_border_constraints failed!"
        return x_adv

    def init_hyperparam(self, x):
        """
        Initialize hyperparameters and create the border mask.
        """
        if self.device is None:
            self.device = x.device
        self.orig_dim = list(x.shape[1:])
        self.ndims = len(self.orig_dim)
        self.border_mask = self.get_border_mask(x.shape, self.bs).to(self.device)
        if self.seed is None:
            self.seed = time.time()

    def check_oscillation(self, x, j, k, y5, k3=0.75):
        t = torch.zeros(x.shape[1]).to(self.device)
        for counter5 in range(k):
          t += (x[j - counter5] > x[j - counter5 - 1]).float()

        return (t <= k * k3 * torch.ones_like(t)).float()

    def check_shape(self, x):
        return x if len(x.shape) > 0 else x.unsqueeze(0)

    def normalize(self, x):
        if self.norm == 'Linf':
            t = x.abs().view(x.shape[0], -1).max(1)[0]

        elif self.norm == 'L2':
            t = (x ** 2).view(x.shape[0], -1).sum(-1).sqrt()

        elif self.norm == 'L1':
            try:
                t = x.abs().view(x.shape[0], -1).sum(dim=-1)
            except:
                t = x.abs().reshape([x.shape[0], -1]).sum(dim=-1)

        return x / (t.view(-1, *([1] * self.ndims)) + 1e-12)

    def dlr_loss(self, x, y):
        x_sorted, ind_sorted = x.sort(dim=1)
        ind = (ind_sorted[:, -1] == y).float()
        u = torch.arange(x.shape[0])

        return -(x[u, y] - x_sorted[:, -2] * ind - x_sorted[:, -1] * (
            1. - ind)) / (x_sorted[:, -1] - x_sorted[:, -3] + 1e-12)

    #

    def wasserstein_loss(self, logits, y, epsilon=0.1, max_iters=10):
        # Convert labels to one-hot vectors
        num_classes = logits.size(1)
        y_onehot = torch.zeros_like(logits).scatter_(1, y.unsqueeze(1), 1.0)

        # Apply softmax to logits to get probability distributions
        p = F.softmax(logits, dim=1)

        # Batch size
        batch_size = logits.size(0)

        # Cost matrix: Squared Euclidean distance between each class index
        class_idx = torch.arange(num_classes, device=logits.device, dtype=torch.float).unsqueeze(0)
        cost_matrix = torch.cdist(class_idx.T, class_idx.T, p=2).pow(2)  # Shape (num_classes, num_classes)

        # Initialize Sinkhorn iterations
        K = torch.exp(-cost_matrix / epsilon)  # Gibbs kernel (regularized transport)
        u = torch.ones((batch_size, num_classes), device=logits.device)  # Uniform marginals
        v = torch.ones((batch_size, num_classes), device=logits.device)

        for _ in range(max_iters):
            u = y_onehot / (torch.matmul(K, v.T).T + 1e-8)  # Update u
            v = p / (torch.matmul(K.T, u.T).T + 1e-8)  # Update v

        # Transport plan
        transport_plan = u.unsqueeze(2) * K * v.unsqueeze(1)

        # Sinkhorn distance
        sinkhorn_distance = torch.sum(transport_plan * cost_matrix.unsqueeze(0), dim=(1, 2))

        return sinkhorn_distance


    def margin_loss(self, logits, y, base_margin=2.0, negative_slope=0.15, scale_factor=0.15):
        # True logits for the correct class
        true_logits = logits.gather(1, y.unsqueeze(1)).squeeze(1)

        # Mask out true logits and get maximum logits for incorrect classes
        mask = torch.ones_like(logits, dtype=torch.bool)
        mask.scatter_(1, y.unsqueeze(1), False)
        other_logits = logits.masked_select(mask).view(logits.size(0), -1)
        max_other_logits, _ = other_logits.max(dim=1)

        # Dynamic margin based on confidence
        confidence = F.softmax(logits, dim=1).max(dim=1)[0]
        dynamic_margin = base_margin * (1.0 - confidence)

        # Compute margin
        margin = max_other_logits - true_logits + dynamic_margin

        # Apply aggressive Leaky ReLU
        leaky_margin = F.leaky_relu(margin, negative_slope=negative_slope)

        # Scale small margins for a stronger attack
        scaled_loss = torch.max(leaky_margin, scale_factor * leaky_margin)

        return scaled_loss

    def cw_loss(self, logits, y, c=0.6):
        one_hot = torch.eye(logits.size(1)).to(logits.device)
        target_one_hot = one_hot[y]
        real = (logits * target_one_hot).sum(dim=1)
        other = (logits * (1 - target_one_hot)).max(dim=1)[0]
        loss = torch.clamp(other - real + c, min=0.0)
        return loss

    def truncation_loss(self, x_adv):
        # Simulate integer conversion (assuming pixel range [0, 1], scaled to [0, 255])
        x_adv_scaled = x_adv * 255.0
        x_adv_rounded = torch.round(x_adv_scaled)
        x_adv_truncated = x_adv_rounded / 255.0

        # Compute discrepancy
        discrepancy = torch.abs(x_adv - x_adv_truncated)

        # Truncation loss as L1 norm of discrepancies
        loss_trunc = torch.mean(discrepancy)

        return loss_trunc

    def get_second_most_confident_class(self, logits, y, exclude_true_class=True):
        if exclude_true_class:
            # Clone logits to avoid modifying the original tensor
            masked_logits = logits.clone()
            masked_logits[torch.arange(logits.size(0)), y] = float('-inf')  # Exclude true class
        else:
            masked_logits = logits

        # Find the indices of the second most confident class
        _, second_class = masked_logits.topk(1, dim=1, largest=True, sorted=True)

        return second_class.squeeze(1)

    def targeted_cross_entropy_loss(self, logits, y, reduction='none'):
        target_classes = self.get_second_most_confident_class(logits, y, exclude_true_class=False)
        loss = F.cross_entropy(logits, target_classes, reduction=reduction)
        return loss

    def combined_loss(self, logits, y):
        """Sum the active losses listed in self.losses."""
        # Compute every loss once ...

        #return F.cross_entropy(logits, y, reduction='none') + self.cw_loss(logits, y)
        
        loss_bank = {
            'CE':  F.cross_entropy(logits, y, reduction='none'),
            'CW':  self.cw_loss(logits, y),
            'MAR': self.margin_loss(logits, y),
            'WAS': self.wasserstein_loss(logits, y),
            'DLR': self.dlr_loss(logits, y),
        }

        # then keep only the ones requested
        active = [loss_bank[name] for name in self.losses if name in loss_bank]

        if not active:
            raise ValueError("self.losses is empty or contains unknown loss names.")

        return sum(active)
                
    def attack_single_run(self, x, y, x_init=None):
        if len(x.shape) < self.ndims:
            x = x.unsqueeze(0)
            y = y.unsqueeze(0)

        if self.norm == 'Linf':
            t = 2 * torch.rand(x.shape).to(self.device).detach() - 1
            t = t * self.border_mask  # Only perturb border pixels
            x_adv = x + self.eps * self.normalize(t)
            x_adv = self.apply_border_constraints(x, x_adv)  # Apply constraints
            x_adv = x_adv.clamp(0., 1.)
        elif self.norm == 'L2':
            t = torch.randn(x.shape).to(self.device).detach()
            x_adv = x + self.eps * torch.ones_like(x
                ).detach() * self.normalize(t)
        elif self.norm == 'L1':
            t = torch.randn(x.shape).to(self.device).detach()
            delta = L1_projection(x, t, self.eps)
            x_adv = x + t + delta
            
        
        
            
        if not x_init is None:
            x_adv = x_init.clone()
            if self.norm == 'L1' and self.verbose:
                print('[custom init] L1 perturbation {:.5f}'.format(
                    (x_adv - x).abs().view(x.shape[0], -1).sum(1).max()))
            
        
        x_adv = x_adv.clamp(0., 1.)
        x_best = x_adv.clone()
        x_best_adv = x_adv.clone()
        loss_steps = torch.zeros([self.n_iter, x.shape[0]]
            ).to(self.device)
        loss_best_steps = torch.zeros([self.n_iter + 1, x.shape[0]]
            ).to(self.device)
        acc_steps = torch.zeros_like(loss_best_steps)

        if not self.is_tf_model:
            if self.loss == 'ce':
                criterion_indiv = nn.CrossEntropyLoss(reduction='none')
            elif self.loss == 'ce-targeted-cfts':
                criterion_indiv = lambda x, y: -1. * F.cross_entropy(x, y,
                    reduction='none')
            elif self.loss == 'dlr':
                criterion_indiv = self.dlr_loss
            elif self.loss == 'dlr-targeted':
                criterion_indiv = self.dlr_loss_targeted
            elif self.loss == 'ce-targeted':
                criterion_indiv = self.ce_loss_targeted
            elif self.loss == 'combined':
                criterion_indiv = self.combined_loss
            else:
                raise ValueError('unknowkn loss')
        else:
            if self.loss == 'ce':
                criterion_indiv = self.model.get_logits_loss_grad_xent
            elif self.loss == 'dlr':
                criterion_indiv = self.model.get_logits_loss_grad_dlr
            elif self.loss == 'dlr-targeted':
                criterion_indiv = self.model.get_logits_loss_grad_target
            else:
                raise ValueError('unknowkn loss')
        
        
        x_adv.requires_grad_()
        grad = torch.zeros_like(x)
        for _ in range(self.eot_iter):
            if not self.is_tf_model:
                with torch.enable_grad():
                    logits = self.model(x_adv)
                    loss_indiv = criterion_indiv(logits, y)
                    loss = loss_indiv.sum()

                grad += torch.autograd.grad(loss, [x_adv])[0].detach()
            else:
                if self.y_target is None:
                    logits, loss_indiv, grad_curr = criterion_indiv(x_adv, y)
                else:
                    logits, loss_indiv, grad_curr = criterion_indiv(x_adv, y,
                        self.y_target)
                grad += grad_curr
        
        grad /= float(self.eot_iter)
        grad_best = grad.clone()

        if self.loss in ['dlr', 'dlr-targeted']:
            # check if there are zero gradients
            check_zero_gradients(grad, logger=self.logger)
        
        acc = logits.detach().max(1)[1] == y
        acc_steps[0] = acc + 0
        loss_best = loss_indiv.detach().clone()

        alpha = 2. if self.norm in ['Linf', 'L2'] else 1. if self.norm in ['L1'] else 2e-2
        step_size = alpha * self.eps * torch.ones([x.shape[0], *(
            [1] * self.ndims)]).to(self.device).detach()
        x_adv_old = x_adv.clone()
        counter = 0
        k = self.n_iter_2 + 0
        n_fts = math.prod(self.orig_dim)
        if self.norm == 'L1':
            k = max(int(.04 * self.n_iter), 1)
            if x_init is None:
                topk = .2 * torch.ones([x.shape[0]], device=self.device)
                sp_old =  n_fts * torch.ones_like(topk)
            else:
                topk = L0_norm(x_adv - x) / n_fts / 1.5
                sp_old = L0_norm(x_adv - x)
            #print(topk[0], sp_old[0])
            adasp_redstep = 1.5
            adasp_minstep = 10.
            #print(step_size[0].item())
        counter3 = 0

        loss_best_last_check = loss_best.clone()
        reduced_last_check = torch.ones_like(loss_best)
        n_reduced = 0

        u = torch.arange(x.shape[0], device=self.device)
        for i in range(self.n_iter):
            #print(f"Start of iteration {i+1}: x_adv min={x_adv.min()}, max={x_adv.max()}")

            ### Gradient step
            with torch.no_grad():
                x_adv = x_adv.detach()
                grad2 = (x_adv - x_adv_old).clamp(-self.eps, self.eps)  # Restrict grad2
                x_adv_old = x_adv.clone()

                a = 0.75 if i > 0 else 1.0

                # Zero out gradients on non-border pixels
                grad = grad * self.border_mask
                grad = grad.clamp(-self.eps, self.eps)  # Clamp gradients

                if self.norm == 'Linf':
                    x_adv_1 = x_adv + step_size * torch.sign(grad)
                    x_adv = x_adv + (x_adv_1 - x_adv) * a  # Apply first update
                    x_adv = x_adv + grad2 * (1 - a)  # Apply second update
                    x_adv = self.apply_border_constraints(x, x_adv)  # Enforce constraints
                    x_adv = x_adv.clamp(0., 1.)  # Ensure valid range

                elif self.norm == 'L2':
                    x_adv_1 = x_adv + step_size * self.normalize(grad)
                    x_adv_1 = torch.clamp(
                        x + self.normalize(x_adv_1 - x) *
                        torch.min(self.eps * torch.ones_like(x).detach(),
                                L2_norm(x_adv_1 - x, keepdim=True)),
                        0.0, 1.0)
                    x_adv = x_adv + (x_adv_1 - x_adv) * a  # Apply first update
                    x_adv = x_adv + grad2 * (1 - a)  # Apply second update
                    x_adv = self.apply_border_constraints(x, x_adv)  # Enforce constraints
                    x_adv = x_adv.clamp(0., 1.)  # Ensure valid range

                elif self.norm == 'L1':
                    grad_topk = grad.abs().view(x.shape[0], -1).sort(-1)[0]
                    topk_curr = torch.clamp((1. - topk) * n_fts, min=0, max=n_fts - 1).long()
                    grad_topk = grad_topk[u, topk_curr].view(-1, *[1] * (len(x.shape) - 1))
                    sparsegrad = grad * (grad.abs() >= grad_topk).float()
                    x_adv_1 = x_adv + step_size * sparsegrad.sign() / (
                        L1_norm(sparsegrad.sign(), keepdim=True) + 1e-10)

                    delta_u = x_adv_1 - x
                    delta_p = L1_projection(x, delta_u, self.eps)
                    x_adv_1 = x + delta_u + delta_p

                    x_adv = x_adv_1 + 0.

                # Debugging logs
                #print(f"Iteration {i}: x_adv after step min={x_adv.min()}, max={x_adv.max()}")
                assert x_adv.min() >= 0.0 and x_adv.max() <= 1.0, \
                    f"x_adv out of bounds! Min: {x_adv.min()}, Max: {x_adv.max()}"

            ### get gradient
            x_adv.requires_grad_()
            grad = torch.zeros_like(x)
            for _ in range(self.eot_iter):
                if not self.is_tf_model:
                    with torch.enable_grad():
                        logits = self.model(x_adv)
                        loss_indiv = criterion_indiv(logits, y)
                        loss = loss_indiv.sum()
    
                    grad += torch.autograd.grad(loss, [x_adv])[0].detach()
                else:
                    if self.y_target is None:
                        logits, loss_indiv, grad_curr = criterion_indiv(x_adv, y)
                    else:
                        logits, loss_indiv, grad_curr = criterion_indiv(x_adv, y, self.y_target)
                    grad += grad_curr
            
            grad /= float(self.eot_iter)

            pred = logits.detach().max(1)[1] == y
            acc = torch.min(acc, pred)
            acc_steps[i + 1] = acc + 0
            ind_pred = (pred == 0).nonzero().squeeze()
            x_best_adv[ind_pred] = x_adv[ind_pred] + 0.
            if self.verbose:
                str_stats = ' - step size: {:.5f} - topk: {:.2f}'.format(
                    step_size.mean(), topk.mean() * n_fts) if self.norm in ['L1'] else ''
                print('[m] iteration: {} - best loss: {:.6f} - robust accuracy: {:.2%}{}'.format(
                    i, loss_best.sum(), acc.float().mean(), str_stats))
                #print('pert {}'.format((x - x_best_adv).abs().view(x.shape[0], -1).sum(-1).max()))
            
            ### check step size
            with torch.no_grad():
              y1 = loss_indiv.detach().clone()
              loss_steps[i] = y1 + 0
              ind = (y1 > loss_best).nonzero().squeeze()
              x_best[ind] = x_adv[ind].clone()
              grad_best[ind] = grad[ind].clone()
              loss_best[ind] = y1[ind] + 0
              loss_best_steps[i + 1] = loss_best + 0

              counter3 += 1

              if counter3 == k:
                  if self.norm in ['Linf', 'L2']:
                      fl_oscillation = self.check_oscillation(loss_steps, i, k,
                          loss_best, k3=self.thr_decr)
                      fl_reduce_no_impr = (1. - reduced_last_check) * (
                          loss_best_last_check >= loss_best).float()
                      fl_oscillation = torch.max(fl_oscillation,
                          fl_reduce_no_impr)
                      reduced_last_check = fl_oscillation.clone()
                      loss_best_last_check = loss_best.clone()
    
                      if fl_oscillation.sum() > 0:
                          ind_fl_osc = (fl_oscillation > 0).nonzero().squeeze()
                          step_size[ind_fl_osc] /= 2.0
                          n_reduced = fl_oscillation.sum()
    
                          x_adv[ind_fl_osc] = x_best[ind_fl_osc].clone()
                          grad[ind_fl_osc] = grad_best[ind_fl_osc].clone()

                      k = max(k - self.size_decr, self.n_iter_min)
                  
                  elif self.norm == 'L1':
                      sp_curr = L0_norm(x_best - x)
                      fl_redtopk = (sp_curr / sp_old) < .95
                      topk = sp_curr / n_fts / 1.5
                      step_size[fl_redtopk] = alpha * self.eps
                      step_size[~fl_redtopk] /= adasp_redstep
                      step_size.clamp_(alpha * self.eps / adasp_minstep, alpha * self.eps)
                      sp_old = sp_curr.clone()
                  
                      x_adv[fl_redtopk] = x_best[fl_redtopk].clone()
                      grad[fl_redtopk] = grad_best[fl_redtopk].clone()
                  
                  counter3 = 0
                  #k = max(k - self.size_decr, self.n_iter_min)
        #
        
        return (x_best, acc, loss_best, x_best_adv)


    def perturb(self, x, y=None, best_loss=False, x_init=None):
        """
        :param x:           clean images
        :param y:           clean labels, if None we use the predicted labels
        :param best_loss:   if True the points attaining highest loss
                            are returned, otherwise adversarial examples
        """

        assert self.loss in ['ce', 'dlr', 'combined'] #'ce-targeted-cfts'
        if not y is None and len(y.shape) == 0:
            x.unsqueeze_(0)
            y.unsqueeze_(0)
        self.init_hyperparam(x)

        x = x.detach().clone().float().to(self.device)
        if not self.is_tf_model:
            y_pred = self.model(x).max(1)[1]
        else:
            y_pred = self.model.predict(x).max(1)[1]
        if y is None:
            #y_pred = self.predict(x).max(1)[1]
            y = y_pred.detach().clone().long().to(self.device)
        else:
            y = y.detach().clone().long().to(self.device)

        adv = x.clone()
        if self.loss != 'ce-targeted':
            acc = y_pred == y
        else:
            acc = y_pred != y
        loss = -1e10 * torch.ones_like(acc).float()
        if self.verbose:
            print('-------------------------- ',
                'running {}-attack with epsilon {:.5f}'.format(
                self.norm, self.eps),
                '--------------------------')
            print('initial accuracy: {:.2%}'.format(acc.float().mean()))

        
        
        if self.use_largereps:
            epss = [3. * self.eps_orig, 2. * self.eps_orig, 1. * self.eps_orig]
            iters = [.3 * self.n_iter_orig, .3 * self.n_iter_orig,
                .4 * self.n_iter_orig]
            iters = [math.ceil(c) for c in iters]
            iters[-1] = self.n_iter_orig - sum(iters[:-1]) # make sure to use the given iterations
            if self.verbose:
                print('using schedule [{}x{}]'.format('+'.join([str(c
                    ) for c in epss]), '+'.join([str(c) for c in iters])))
        
        startt = time.time()
        if not best_loss:
            torch.random.manual_seed(self.seed)
            torch.cuda.random.manual_seed(self.seed)

            for counter in range(self.n_restarts):
                ind_to_fool = acc.nonzero().squeeze()
                if len(ind_to_fool.shape) == 0:
                    ind_to_fool = ind_to_fool.unsqueeze(0)
                if ind_to_fool.numel() != 0:
                    x_to_fool = x[ind_to_fool].clone()
                    y_to_fool = y[ind_to_fool].clone()
                    
                    
                    if not self.use_largereps:
                        res_curr = self.attack_single_run(x_to_fool, y_to_fool)
                    else:
                        res_curr = self.decr_eps_pgd(x_to_fool, y_to_fool, epss, iters)
                    best_curr, acc_curr, loss_curr, adv_curr = res_curr
                    ind_curr = (acc_curr == 0).nonzero().squeeze()

                    acc[ind_to_fool[ind_curr]] = 0
                    adv[ind_to_fool[ind_curr]] = adv_curr[ind_curr].clone()
                    if self.verbose:
                        print('restart {} - robust accuracy: {:.2%}'.format(
                            counter, acc.float().mean()),
                            '- cum. time: {:.1f} s'.format(
                            time.time() - startt))

            return adv

        else:
            adv_best = x.detach().clone()
            loss_best = torch.ones([x.shape[0]]).to(
                self.device) * (-float('inf'))
            for counter in range(self.n_restarts):
                best_curr, _, loss_curr, _ = self.attack_single_run(x, y)
                ind_curr = (loss_curr > loss_best).nonzero().squeeze()
                adv_best[ind_curr] = best_curr[ind_curr] + 0.
                loss_best[ind_curr] = loss_curr[ind_curr] + 0.

                if self.verbose:
                    print('restart {} - loss: {:.5f}'.format(
                        counter, loss_best.sum()))

            return adv_best




    def decr_eps_pgd(self, x, y, epss, iters, use_rs=True):
        assert len(epss) == len(iters)
        assert self.norm in ['L1']
        self.use_rs = False
        if not use_rs:
            x_init = None
        else:
            x_init = x + torch.randn_like(x)
            x_init += L1_projection(x, x_init - x, 1. * float(epss[0]))
        eps_target = float(epss[-1])
        if self.verbose:
            print('total iter: {}'.format(sum(iters)))
        for eps, niter in zip(epss, iters):
            if self.verbose:
                print('using eps: {:.2f}'.format(eps))
            self.n_iter = niter + 0
            self.eps = eps + 0.
            #
            if not x_init is None:
                x_init += L1_projection(x, x_init - x, 1. * eps)
            x_init, acc, loss, x_adv = self.attack_single_run(x, y, x_init=x_init)

        return (x_init, acc, loss, x_adv)


class BorderAPGD_targeted:

    def __init__(self,
                 predict,
                 n_iter = 100,
                 norm = "Linf",
                 n_restarts = 1,
                 eps = None,
                 seed = 0,
                 eot_iter = 1,
                 rho = .75,
                 topk = None,
                 n_target_classes = 9,
                 verbose = False,
                 device = None,
                 use_largereps = False,
                 is_tf_model = False,
                 logger = None,
                 losses = None,
                 bs = None,
                 model_name = None):

        from copy import deepcopy
        from types import SimpleNamespace
        self.base = SimpleNamespace()

        self.model = predict
        self.n_iter = n_iter
        self.norm = norm
        self.n_restarts = n_restarts
        self.eps = eps
        self.seed = seed
        self.eot_iter = eot_iter
        self.thr_decr = rho
        self.topk = topk
        self.n_target_classes = n_target_classes
        self.verbose = verbose
        self.device = device
        self.use_largereps = use_largereps
        self.is_tf_model = is_tf_model
        self.logger = logger

        self.loss = "combined"
        self.n_iter_orig = n_iter
        self.eps_orig = float(eps)
        self.y_target = None

        self.losses = losses
        self.model_name = model_name

        self.bs = 1

        assert self.norm in ["Linf", "L2", "L1"], "Unsupported norm."
        assert self.eps is not None, "eps must be specified."

        self.n_iter_2 = max(int(0.22 * self.n_iter), 1)
        self.n_iter_min = max(int(0.06 * self.n_iter), 1)
        self.size_decr = max(int(0.03 * self.n_iter), 1)

    def get_border_mask(self, shape, thickness=1):
        """
        Create a binary mask where the border pixels are set to 1 and the rest are 0.
        Thickness can be specified.
        """
        b, c, h, w = shape
        mask = torch.zeros((1, c, h, w), device=self.device)

        # Top and bottom borders
        mask[:, :, :thickness, :] = 1
        mask[:, :, -thickness:, :] = 1

        # Left and right borders
        mask[:, :, :, :thickness] = 1
        mask[:, :, :, -thickness:] = 1

        return mask

    def apply_border_constraints(self, x, x_adv):
        """Clamp perturbation to *eps* **and** force it to lie on the frame."""
        delta = (x_adv - x) * self.border_mask  # zero-out interior
        if self.norm == "Linf":
            delta = delta.clamp(-self.eps, self.eps)
        elif self.norm == "L2":
            delta = self.normalize(delta) * torch.min(self.eps * torch.ones_like(delta), L2_norm(delta, keepdim=True))
        elif self.norm == "L1":
            delta = delta + L1_projection(x, delta, self.eps)
        x_adv = (x + delta).clamp(0., 1.)
        return x_adv

    def check_oscillation(self, x, j, k, y5, k3=0.75):
        t = torch.zeros(x.shape[1]).to(self.device)
        for counter5 in range(k):
          t += (x[j - counter5] > x[j - counter5 - 1]).float()

        return (t <= k * k3 * torch.ones_like(t)).float()

    def check_shape(self, x):
        return x if x.dim() > 0 else x.unsqueeze(0)

    def normalize(self, x):
        if self.norm == "Linf":
            t = x.abs().view(x.shape[0], -1).max(1)[0]
        elif self.norm == "L2":
            t = L2_norm(x)
        elif self.norm == "L1":
            t = L1_norm(x)
        return x / (t.view(-1, *([1] * self.ndims)) + 1e-12)

    # --- loss functions (targeted) ---------------------------------------------------------------
    def dlr_loss_targeted(self, logits, y):
        x_sorted, ind_sorted = logits.sort(dim=1)
        u = torch.arange(logits.shape[0], device=logits.device)
        return -(logits[u, y] - logits[u, self.y_target]) / (
            x_sorted[:, -1] - .5 * (x_sorted[:, -3] + x_sorted[:, -4]) + 1e-12)

    def ce_loss_targeted(self, logits, y):
        return -F.cross_entropy(logits, self.y_target, reduction="none")

    def wasserstein_loss(self, logits, y=None, epsilon=0.1, max_iters=10):
        """Targeted Sinkhorn Wasserstein loss toward self.y_target."""
        # One-hot for target class
        num_classes = logits.size(1)
        y_target = self.y_target
        y_onehot = torch.zeros_like(logits).scatter_(1, y_target.unsqueeze(1), 1.0)

        # Predicted distribution
        p = F.softmax(logits, dim=1)
        batch_size = logits.size(0)

        # Cost matrix: squared Euclidean between class indices
        class_idx = torch.arange(num_classes, device=logits.device, dtype=torch.float).unsqueeze(0)
        cost_matrix = torch.cdist(class_idx.T, class_idx.T, p=2).pow(2)

        # Sinkhorn iterations
        K = torch.exp(-cost_matrix / epsilon)
        u = torch.ones((batch_size, num_classes), device=logits.device)
        v = torch.ones((batch_size, num_classes), device=logits.device)

        for _ in range(max_iters):
            u = y_onehot / (torch.matmul(K, v.T).T + 1e-8)
            v = p / (torch.matmul(K.T, u.T).T + 1e-8)

        transport_plan = u.unsqueeze(2) * K * v.unsqueeze(1)
        sinkhorn_distance = torch.sum(transport_plan * cost_matrix.unsqueeze(0), dim=(1, 2))
        return sinkhorn_distance

    def cw_loss(self, logits, y=None, c=0.6):
        """Targeted Carlini-Wagner loss toward self.y_target."""
        # Prepare one-hot for target
        num_classes = logits.size(1)
        y_target = self.y_target
        one_hot = torch.eye(num_classes, device=logits.device)
        target_one_hot = one_hot[y_target]

        # Logit for target class
        real = (logits * target_one_hot).sum(dim=1)
        # Maximum logit among all other classes
        mask = 1 - target_one_hot
        other = (logits * mask).max(dim=1)[0]

        # CW margin
        loss = torch.clamp(other - real + c, min=0.0)
        return loss

    def margin_loss(self, logits, y=None, base_margin=2.0, negative_slope=0.15, scale_factor=0.15):
        """Targeted margin loss pushing logits toward self.y_target."""
        # Logit for target class
        y_target = self.y_target
        true_logits = logits.gather(1, y_target.unsqueeze(1)).squeeze(1)

        # Mask out target and find max of other logits
        mask = torch.ones_like(logits, dtype=torch.bool)
        mask.scatter_(1, y_target.unsqueeze(1), False)
        other_logits = logits.masked_select(mask).view(logits.size(0), -1)
        max_other, _ = other_logits.max(dim=1)

        # Dynamic margin based on confidence
        confidence = F.softmax(logits, dim=1).max(dim=1)[0]
        dynamic_margin = base_margin * (1.0 - confidence)

        # Compute and scale margin
        margin = max_other - true_logits + dynamic_margin
        leaky = F.leaky_relu(margin, negative_slope=negative_slope)
        scaled = torch.max(leaky, scale_factor * leaky)
        return scaled

    def combined_loss(self, logits, y):
        """Sum the active losses listed in self.losses."""
        
        loss_bank = {
            'CE':  self.ce_loss_targeted(logits, y),
            'CW':  self.cw_loss(logits, y),
            'MAR': self.margin_loss(logits, y),
            'WAS': self.wasserstein_loss(logits, y),
            'DLR': self.dlr_loss_targeted(logits, y),
        }

        active = [loss_bank[name] for name in self.losses if name in loss_bank]

        if not active:
            raise ValueError("self.losses is empty or contains unknown loss names.")
        
        return sum(active)
        
        
    def init_hyperparam(self, x):
        if self.device is None:
            self.device = x.device
        self.orig_dim = list(x.shape[1:])
        self.ndims = len(self.orig_dim)
        self.border_mask = self.get_border_mask(x.shape, self.bs).to(self.device)
        if self.seed is None:
            self.seed = time.time()


    def attack_single_run(self, x, y, x_init=None):
        """Identical to your *frame* variant but with targeted losses."""
        if x.dim() < self.ndims:
            x, y = x.unsqueeze(0), y.unsqueeze(0)

        if self.norm == "Linf":
            t = 2 * torch.rand_like(x) - 1
            t *= self.border_mask
            x_adv = x + self.eps * self.normalize(t)
            x_adv = self.apply_border_constraints(x, x_adv)
        elif self.norm == "L2":
            t = torch.randn_like(x)
            t *= self.border_mask
            x_adv = x + self.eps * self.normalize(t)
            x_adv = self.apply_border_constraints(x, x_adv)
        elif self.norm == "L1":
            t = torch.randn_like(x)
            t *= self.border_mask
            delta = L1_projection(x, t, self.eps)
            x_adv = x + t + delta
            x_adv = self.apply_border_constraints(x, x_adv)

        if x_init is not None:
            x_adv = x_init.clone()

        x_adv = x_adv.clamp(0., 1.)
        x_best = x_adv.clone()
        x_best_adv = x_adv.clone()
        loss_steps = torch.zeros(self.n_iter, x.size(0), device=self.device)
        loss_best_steps = torch.zeros(self.n_iter + 1, x.size(0), device=self.device)
        acc_steps = torch.zeros_like(loss_best_steps)

        if self.loss == "ce-targeted":
            criterion_indiv = self.ce_loss_targeted
        elif self.loss == "dlr-targeted":
            criterion_indiv = self.dlr_loss_targeted
        elif self.loss == "combined":
            criterion_indiv = self.combined_loss
        else:
            raise ValueError("Unsupported loss")

        x_adv.requires_grad_()
        grad = torch.zeros_like(x)
        for _ in range(self.eot_iter):
            with torch.enable_grad():
                logits = self.model(x_adv)
                loss_indiv = criterion_indiv(logits, y)
                loss = loss_indiv.sum()
            grad += torch.autograd.grad(loss, [x_adv])[0].detach()
        grad /= float(self.eot_iter)
        grad_best = grad.clone()
        check_zero_gradients(grad, logger=self.logger)

        acc = (logits.detach().max(1)[1] == y)  
        acc_steps[0] = acc.float()
        loss_best = loss_indiv.detach().clone()

        alpha = 2.0 if self.norm in ["Linf", "L2"] else 1.0
        step_size = alpha * self.eps * torch.ones(x.size(0), *([1] * self.ndims), device=self.device)
        x_adv_old = x_adv.clone()
        counter3 = 0
        k = self.n_iter_2
        n_fts = math.prod(self.orig_dim)
        if self.norm == "L1":
            k = max(int(.04 * self.n_iter), 1)
            if x_init is None:
                topk = .2 * torch.ones(x.size(0), device=self.device)
                sp_old = n_fts * torch.ones_like(topk)
            else:
                topk = L0_norm(x_adv - x) / n_fts / 1.5
                sp_old = L0_norm(x_adv - x)
            adasp_redstep = 1.5
            adasp_minstep = 10.

        loss_best_last_check = loss_best.clone()
        reduced_last_check = torch.ones_like(loss_best)

        for i in range(self.n_iter):

            with torch.no_grad():
                x_adv = x_adv.detach()
                grad2 = (x_adv - x_adv_old) * self.border_mask  # momentum term masked
                grad2 = grad2.clamp(-self.eps, self.eps)
                x_adv_old = x_adv.clone()

                a = 0.75 if i > 0 else 1.0

                # mask gradients so we never touch interior pixels
                grad = (grad * self.border_mask).clamp(-self.eps, self.eps)

                if self.norm == "Linf":
                    x_adv_1 = x_adv + step_size * torch.sign(grad)
                    x_adv = x_adv + (x_adv_1 - x_adv) * a + grad2 * (1 - a)
                    x_adv = self.apply_border_constraints(x, x_adv)
                elif self.norm == "L2":
                    x_adv_1 = x_adv + step_size * self.normalize(grad)
                    x_adv_1 = x + self.normalize(x_adv_1 - x) * torch.min(self.eps * torch.ones_like(x), L2_norm(x_adv_1 - x, keepdim=True))
                    x_adv = x_adv + (x_adv_1 - x_adv) * a + grad2 * (1 - a)
                    x_adv = self.apply_border_constraints(x, x_adv)
                elif self.norm == "L1":
                    grad_topk = grad.abs().view(x.size(0), -1).sort(-1)[0]
                    topk_curr = torch.clamp((1 - topk) * n_fts, min=0, max=n_fts - 1).long()
                    grad_topk = grad_topk[torch.arange(x.size(0), device=self.device), topk_curr].view(-1, *([1] * (x.dim() - 1)))
                    sparse_grad = grad * (grad.abs() >= grad_topk).float()
                    x_adv_1 = x_adv + step_size * sparse_grad.sign() / (L1_norm(sparse_grad.sign(), keepdim=True) + 1e-10)
                    delta_u = x_adv_1 - x
                    delta_p = L1_projection(x, delta_u, self.eps)
                    x_adv = x + delta_u + delta_p
                    x_adv = self.apply_border_constraints(x, x_adv)


            x_adv.requires_grad_()
            grad.zero_()
            for _ in range(self.eot_iter):
                with torch.enable_grad():
                    logits = self.model(x_adv)
                    loss_indiv = criterion_indiv(logits, y)
                    loss = loss_indiv.sum()
                grad += torch.autograd.grad(loss, [x_adv])[0].detach()
            grad /= float(self.eot_iter)

            pred = logits.detach().max(1)[1] == y  # still untargeted measure
            acc = torch.min(acc, pred)
            acc_steps[i + 1] = acc.float()
            ind_pred = (pred == 0).nonzero(as_tuple=False).squeeze()
            x_best_adv[ind_pred] = x_adv[ind_pred].clone()


            with torch.no_grad():
                y1 = loss_indiv.detach().clone()
                loss_steps[i] = y1
                ind = (y1 > loss_best).nonzero(as_tuple=False).squeeze()
                x_best[ind] = x_adv[ind].clone()
                grad_best[ind] = grad[ind].clone()
                loss_best[ind] = y1[ind]
                loss_best_steps[i + 1] = loss_best

                counter3 += 1
                if counter3 == k:
                    if self.norm in ["Linf", "L2"]:
                        fl_osc = self.check_oscillation(loss_steps, i, k, loss_best, k3=self.thr_decr)
                        fl_red_no_impr = (1 - reduced_last_check) * (loss_best_last_check >= loss_best).float()
                        fl_osc = torch.max(fl_osc, fl_red_no_impr)
                        reduced_last_check = fl_osc.clone()
                        loss_best_last_check = loss_best.clone()
                        if fl_osc.sum() > 0:
                            ind_fl = (fl_osc > 0).nonzero(as_tuple=False).squeeze()
                            step_size[ind_fl] /= 2.0
                            x_adv[ind_fl] = x_best[ind_fl].clone()
                            grad[ind_fl] = grad_best[ind_fl].clone()
                        k = max(k - self.size_decr, self.n_iter_min)
                    elif self.norm == "L1":
                        sp_curr = L0_norm(x_best - x)
                        fl_redtopk = (sp_curr / sp_old) < .95
                        topk = sp_curr / n_fts / 1.5
                        step_size[fl_redtopk] = alpha * self.eps
                        step_size[~fl_redtopk] /= adasp_redstep
                        step_size.clamp_(alpha * self.eps / adasp_minstep, alpha * self.eps)
                        sp_old = sp_curr.clone()
                        x_adv[fl_redtopk] = x_best[fl_redtopk].clone()
                        grad[fl_redtopk] = grad_best[fl_redtopk].clone()
                    counter3 = 0

        return x_best, acc, loss_best, x_best_adv


    def perturb(self, x, y=None, x_init=None):
        """
        :param x:           clean images
        :param y:           clean labels, if None we use the predicted labels
        """

        assert self.loss in ['dlr-targeted', 'combined'] #'ce-targeted'
        if not y is None and len(y.shape) == 0:
            x.unsqueeze_(0)
            y.unsqueeze_(0)
        self.init_hyperparam(x)

        x = x.detach().clone().float().to(self.device)
        if not self.is_tf_model:
            y_pred = self.model(x).max(1)[1]
        else:
            y_pred = self.model.predict(x).max(1)[1]
        if y is None:
            #y_pred = self._get_predicted_label(x)
            y = y_pred.detach().clone().long().to(self.device)
        else:
            y = y.detach().clone().long().to(self.device)

        adv = x.clone()
        acc = y_pred == y
        if self.verbose:
            print('-------------------------- ',
                'running {}-attack with epsilon {:.5f}'.format(
                self.norm, self.eps),
                '--------------------------')
            print('initial accuracy: {:.2%}'.format(acc.float().mean()))

        startt = time.time()

        torch.random.manual_seed(self.seed)
        torch.cuda.random.manual_seed(self.seed)

        #
        
        if self.use_largereps:
            epss = [3. * self.eps_orig, 2. * self.eps_orig, 1. * self.eps_orig]
            iters = [.3 * self.n_iter_orig, .3 * self.n_iter_orig,
                .4 * self.n_iter_orig]
            iters = [math.ceil(c) for c in iters]
            iters[-1] = self.n_iter_orig - sum(iters[:-1])
            if self.verbose:
                print('using schedule [{}x{}]'.format('+'.join([str(c
                    ) for c in epss]), '+'.join([str(c) for c in iters])))
        
        for target_class in range(2, self.n_target_classes + 2):
            for counter in range(self.n_restarts):
                ind_to_fool = acc.nonzero().squeeze()
                if len(ind_to_fool.shape) == 0:
                    ind_to_fool = ind_to_fool.unsqueeze(0)
                if ind_to_fool.numel() != 0:
                    x_to_fool = x[ind_to_fool].clone()
                    y_to_fool = y[ind_to_fool].clone()
                    
                    if not self.is_tf_model:
                        output = self.model(x_to_fool)
                    else:
                        output = self.model.predict(x_to_fool)
                    self.y_target = output.sort(dim=1)[1][:, -target_class]

                    if not self.use_largereps:
                        res_curr = self.attack_single_run(x_to_fool, y_to_fool)
                    else:
                        res_curr = self.decr_eps_pgd(x_to_fool, y_to_fool, epss, iters)
                    best_curr, acc_curr, loss_curr, adv_curr = res_curr
                    ind_curr = (acc_curr == 0).nonzero().squeeze()

                    acc[ind_to_fool[ind_curr]] = 0
                    adv[ind_to_fool[ind_curr]] = adv_curr[ind_curr].clone()
                    if self.verbose:
                        print('target class {}'.format(target_class),
                            '- restart {} - robust accuracy: {:.2%}'.format(
                            counter, acc.float().mean()),
                            '- cum. time: {:.1f} s'.format(
                            time.time() - startt))

        return adv