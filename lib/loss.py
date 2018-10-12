import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

# copyright https://github.com/marvis/pytorch-yolo2/blob/master/FocalLoss.py#L60
class FocalLoss(nn.Module):
    r"""
        This criterion is a implemenation of Focal Loss, which is proposed in 
        Focal Loss for Dense Object Detection.
            
            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])
    
        The losses are averaged across observations for each minibatch.
        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classi?ed examples (p > .5), 
                                   putting more focus on hard, misclassi?ed examples
            size_average(bool): size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.
    """
    def __init__(self, class_num, alpha=None, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):
    
        N = inputs.size(0)
        #print(N)
        C = inputs.size(1)
        P = F.softmax(inputs)

        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)
        #print(class_mask)
        

        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]
        
        probs = (P*class_mask).sum(1).view(-1,1)

        log_p = probs.log()
        #print('probs size= {}'.format(probs.size()))
        #print(probs)

        batch_loss = -alpha*(torch.pow((1-probs), self.gamma))*log_p 
        #print('-----bacth_loss------')
        #print(batch_loss)
 
        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss



class DiceLoss(nn.Module):

    def __init__(self, smooth=0, eps=1e-7):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.eps = eps

    def forward(self, output, target):
        return 1 - (2 * torch.sum(output * target) + self.smooth) / (
                torch.sum(output) + torch.sum(target) + self.smooth + self.eps)


def where(cond, x_1, x_2):
    cond = cond.long()
    return (cond * x_1) + ((1 - cond) * x_2)


def mixed_dice_bce_loss(output, target, dice_weight=0.2, dice_loss=None,bce_weight=0.9, bce_loss=None,smooth=0, dice_activation='sigmoid'):

    num_classes = output.size(1)
    target = target[:, :num_classes, :, :].long()
    if bce_loss is None:
        bce_loss = nn.BCEWithLogitsLoss()
    if dice_loss is None:
        dice_loss = multiclass_dice_loss
    return dice_weight * dice_loss(output, target, smooth, dice_activation) + bce_weight * bce_loss(output, target)


def mixed_dice_cross_entropy_loss(output, target, dice_weight=0.5, dice_loss=None,cross_entropy_weight=0.5, cross_entropy_loss=None, smooth=0,
dice_activation='softmax'):
    num_classes_without_background = output.size(1) - 1
    dice_output = output[:, 1:, :, :]
    dice_target = target[:, :num_classes_without_background, :, :].long()
    cross_entropy_target = torch.zeros_like(target[:, 0, :, :]).long()
    for class_nr in range(num_classes_without_background):
        cross_entropy_target = where(target[:, class_nr, :, :], class_nr + 1, cross_entropy_target)
    if cross_entropy_loss is None:
        cross_entropy_loss = nn.CrossEntropyLoss()
    if dice_loss is None:
        dice_loss = multiclass_dice_loss

    return dice_weight * dice_loss(dice_output, dice_target, smooth,dice_activation) + cross_entropy_weight * cross_entropy_loss(output,cross_entropy_target)


def multiclass_dice_loss(output, target, smooth=0, activation='softmax'):

    """Calculate Dice Loss for multiple class output.
    Args:
        output (torch.Tensor): Model output of shape (N x C x H x W).
        target (torch.Tensor): Target of shape (N x H x W).
        smooth (float, optional): Smoothing factor. Defaults to 0.
        activation (string, optional): Name of the activation function, softmax or sigmoid. Defaults to 'softmax'.

    Returns:
        torch.Tensor: Loss value.
    """
    if activation == 'softmax':
        activation_nn = torch.nn.Softmax2d()
    elif activation == 'sigmoid':
        activation_nn = torch.nn.Sigmoid()
    else:
        raise NotImplementedError('only sigmoid and softmax are implemented')

    loss = 0
    dice = DiceLoss(smooth=smooth)
    output = activation_nn(output)
    num_classes = output.size(1)
    target.data = target.data.float()
    for class_nr in range(num_classes):
        loss += dice(output[:, class_nr, :, :], target[:, class_nr, :, :])
    return loss / num_classes


def get_torch_loss_function(loss_function):
    class loss(nn.Module):
        def forward(self, output, target):
            return loss_function(output, target)
    return loss()

torch_loss_dict = {
                   'binary_crossentropy':torch.nn.BCELoss(),
                   'diceLoss':DiceLoss(),
                   'mixed_dice_bce_loss':get_torch_loss_function(mixed_dice_bce_loss),
                   'mixed_dice_cross_entropy_loss':get_torch_loss_function(mixed_dice_cross_entropy_loss),
                   'multiclass_dice_loss' : get_torch_loss_function(multiclass_dice_loss),
                  }