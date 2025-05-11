"""
Different explanation methods from the Integrated-Gradient Optimized Saliency map methods.
© copyright Tyler Lawson, Saeed khorram. https://github.com/saeed-khorram/IGOS
"""

from torch.autograd import Variable
from .methods_helper import *
#NOTE: upscale imported from methods_helper
'''
def IGOS(
        model,
        init_mask,
        images,
        baselines,
        labels,
        L1=1,
        L2=20,
        size=28,
        ig_iter=15,
        iterations=20,
        alpha=8,
        opt='LS',
        softmax=True,
        **kwargs
):

    """
        Generates explanation by optimizing a mask with integrated gradient.
        Paper title:  Visualizing Deep Networks by Optimizing with Integrated Gradients, AAAI 2020
        Link to the paper: https://aaai.org/ojs/index.php/AAAI/article/view/6863/6717

    :param model: The model to use for making predictions
    :param model_name: The model name to use for making predictions
    :param init_mask: The area in one cell of the K x K grid to use for initializing the mask
    :param image: The image to be explained
    :param baseline: The blured image as the baseline to use for making predictions
    :param label: The predicted class index of this image
    :param L1: The hyperparameter for TV/BTV norm
    :param L2: The hyperparameter for TV/BTV norm
    :param size: The size of the predicted mask 
    :param ig_iter: The step size of the integtated gradient accumulation
    :param iterations: The number of iterations required to predict the mask
    :param alpha: The step size for updating the mask
    :param opt: The optimization algorithm
    :param softmax: The output function for the model
    :param kwargs:
    :return:
    """

    def regularization_loss(masks):
        return L1 * torch.mean(torch.abs(1 - masks).view(masks.shape[0], -1), dim=1) + \
               L2 * tv_norm(masks)

        # Define loss function for regularization terms

    def loss_function(up_masks, masks, indices, noise=True):
        losses = interval_score(
            model,
            model_name, 
            images[indices],
            baselines[indices],
            labels[indices],
            up_masks,
            ig_iter,
            output_func,
            noise
            )
        return losses.sum(dim=1).view(-1) + regularization_loss(masks)

    # Create initial masks
    masks = torch.ones((images.shape[0], 1, size, size), dtype=torch.float32, device='cuda')
    masks = masks * init_mask.cuda()
    masks = Variable(masks, requires_grad=True)

    if softmax:
        output_func = softmax_output
    else:
        logit_output.original = torch.gather(torch.nn.Sigmoid()(model(image)), 1, label.view(-1, 1))
        output_func = logit_output
           TODO-aniket: why #
        raise NotImplementedError

    if opt == 'NAG':
        cita=torch.zeros(1).cuda()

    for i in range(iterations):

        up_masks = upscale(masks)

        losses = regularization_loss(masks)
        losses.sum().backward()
        total_grads = masks.grad.clone()
        masks.grad.zero_()

        # Computer the integrated gradient
        #integrated_gradient(model, model_name, image, baseline, label, up_masks, ig_iter, output_func)

        cnn_lossi = integrated_gradient(model, images, baselines, labels, up_masks, ig_iter, output_func)
        total_grads += masks.grad.clone()
        masks.grad.zero_()

        if opt == 'LS':
            alphas = line_search(masks, total_grads, loss_function, alpha)
            # Update the mask
            masks.data -= total_grads * alphas
        
        if opt == 'NAG':
            e = i / (i + 3)
            cita_p = cita
            cita = masks.data - alpha * total_grads
            masks.data = cita + e * (cita - cita_p)

        masks.grad.zero_()
        masks.data.clamp_(0, 1)

    return masks
'''

def iGOS_pp(
        model,
        model_name, 
        #init_mask,
        images,
        baselines,
        labels,
        size=28,
        iterations=15,
        ig_iter=20,
        L1=1,
        L2=20,
        alpha=1000,
        opt='LS',
        softmax=True,
        lambda_primary=1,
        bilateral_sigma=0.01,
       **kwargs):
    """
        Generates explanation by optimizing a separate masks for insertion and deletion.
        Paper title:  iGOS++: Integrated Gradient Optimized Saliency by Bilateral Perturbations
        Link to the paper: https://arxiv.org/pdf/2012.15783.pdf
        Paper title:  Diverse Explanations for Object Detectors with Nesterov-Accelerated iGOS++

    :param model: The model to use for making predictions
    :param model_name: The model name to use for making predictions
    :param init_mask: The area in one cell of the K x K grid to use for initializing the mask
    :param image: The image to be explained
    :param baseline: The blured image as the baseline to use for making predictions
    :param label: The predicted class index of this image
    :param size: The size of the predicted mask 
    :param iterations: The number of iterations required to predict the mask
    :param ig_iter: The step size of the integtated gradient accumulation
    :param L1: The hyperparameter for TV/BTV norm
    :param L2: The hyperparameter for TV/BTV norm
    :param alpha: The step size for updating the mask
    :param opt: The optimization algorithm
    :param softmax: The output function for the model
    :param kwargs:
    :return:
    """
    def regularization_loss(images, masks):
        return L1 * torch.mean(torch.abs(1 - masks).view(masks.shape[0], -1), dim=1) + \
               L2 * bilateral_tv_norm(images, masks, tv_beta=2, sigma=bilateral_sigma) # 0.01

    def ins_loss_function(up_masks, indices, noise=True):
        losses = -interval_score(
                    model,
                    model_name,
                    baselines[indices],
                    images[indices],
                    labels[indices],
                    up_masks,
                    ig_iter,
                    output_func,
                    noise
                    )
        return losses.sum(dim=1).view(-1)

    def del_loss_function(up_masks, indices, noise=True):
        losses = interval_score(
                    model,
                    model_name,
                    images[indices],
                    baselines[indices],
                    labels[indices],
                    up_masks,
                    ig_iter,
                    output_func,
                    noise,
                    )
        return losses.sum(dim=1).view(-1)

    def loss_function(up_masks, masks, indices):
        loss = del_loss_function(up_masks[:, 0], indices)
        loss += ins_loss_function(up_masks[:, 1], indices)
        loss += del_loss_function(up_masks[:, 0] * up_masks[:, 1], indices)
        loss += ins_loss_function(up_masks[:, 0] * up_masks[:, 1], indices)
        return lambda_primary *loss + regularization_loss(images[indices], masks[:, 0] * masks[:, 1])
    #TODO: remove cuda? and use device
    # sizeh,sizew?
    sizeh,sizew = (size/min(images.shape[2],images.shape[3]))*images.shape[2],(size/min(images.shape[2],images.shape[3]))*images.shape[3]
    sizeh,sizew = int(sizeh),int(sizew)
    #masks_del = torch.ones((images.shape[0], 1, size, size), dtype=torch.float32, device='cuda')
    masks_del = torch.ones((images.shape[0], 1, sizeh, sizew), dtype=torch.float32, device='cuda')
    masks_del = masks_del #* init_mask.cuda()
    masks_del = Variable(masks_del, requires_grad=True)
    #masks_ins = torch.ones((image.shape[0], 1, size, size), dtype=torch.float32, device='cuda')
    masks_ins = torch.ones((images.shape[0], 1, sizeh, sizew), dtype=torch.float32, device='cuda')
    masks_ins = masks_ins #* init_mask.cuda()
    masks_ins = Variable(masks_ins, requires_grad=True)

    if softmax:
        output_func = softmax_output
    else:
        logit_output.original = torch.gather(torch.nn.Sigmoid()(model(images)), 1, labels.view(-1,1))
        output_func = logit_output

    if opt == 'NAG':
        cita_d=torch.zeros(1).cuda()
        cita_i=torch.zeros(1).cuda()
    all_losses_deletion = []
    all_losses_insertion = []
    all_losses_combinedd = []
    all_losses_combinedi = []
    for i in range(iterations):
        up_masks1 = upscale(masks_del)
        up_masks2 = upscale(masks_ins)

        # Compute the integrated gradient for the combined mask, optimized for deletion
        #TODO: why is model_name being used everywhere?
        loss_combinedd = integrated_gradient(model, model_name, images, baselines, labels, up_masks1 * up_masks2, ig_iter, output_func)
        all_losses_combinedd.append(loss_combinedd)

        total_grads1 = masks_del.grad.clone()
        total_grads2 = masks_ins.grad.clone()
        masks_del.grad.zero_()
        masks_ins.grad.zero_()

        # Compute the integrated gradient for the combined mask, optimized for insertion
        #TODO-aniket: look at this integrated_gradient function
        loss_combinedi = integrated_gradient(model, model_name, baselines, images, labels, up_masks1 * up_masks2, ig_iter, output_func)
        all_losses_combinedi.append(loss_combinedi)
        total_grads1 -= masks_del.grad.clone()  # Negative because insertion loss is 1 - score.
        total_grads2 -= masks_ins.grad.clone()
        masks_del.grad.zero_()
        masks_ins.grad.zero_()

        # Compute the integrated gradient for the deletion mask
        loss_insertion = integrated_gradient(model, model_name, images, baselines, labels, up_masks1, ig_iter, output_func)
        all_losses_insertion.append(loss_insertion)
        total_grads1 += masks_del.grad.clone()
        masks_del.grad.zero_()

        # Compute the integrated graident for the insertion mask
        loss_deletion = integrated_gradient(model, model_name, baselines, images, labels, up_masks2, ig_iter, output_func)
        all_losses_deletion.append(loss_deletion)
        total_grads2 -= masks_ins.grad.clone()
        masks_ins.grad.zero_()

        # Average them to balance out the terms with the regularization terms
        total_grads1 /= 2
        total_grads2 /= 2

        # Computer regularization for combined masks
        losses = regularization_loss(images, masks_del * masks_ins)
        losses.sum().backward()
        total_grads1 += masks_del.grad.clone()
        total_grads2 += masks_ins.grad.clone()

        #TODO-aniket
        if opt == 'LS':
            masks = torch.cat((masks_del.unsqueeze(1), masks_ins.unsqueeze(1)), 1)
            total_grads = torch.cat((total_grads1.unsqueeze(1), total_grads2.unsqueeze(1)), 1)
            alphas = line_search(masks, total_grads, loss_function, alpha)
            masks_del.data -= total_grads1 * alphas
            masks_ins.data -= total_grads2 * alphas
        
        if opt == 'NAG':
            e = i / (i + 3)
            cita_d_p = cita_d
            cita_i_p = cita_i
            cita_d = masks_del.data - alpha * total_grads1
            cita_i = masks_ins.data - alpha * total_grads2
            masks_del.data = cita_d + e * (cita_d - cita_d_p)
            masks_ins.data = cita_i + e * (cita_i - cita_i_p)

        masks_del.grad.zero_()
        masks_ins.grad.zero_()
        masks_del.data.clamp_(0,1)
        masks_ins.data.clamp_(0,1)
    if dutils.hack('stop and visualize the igos masks',default=False):
        dutils.save_plot(all_losses_insertion,'loss_insertion','loss_insertion.png')
        dutils.save_plot(all_losses_deletion,'loss_deletion','loss_deletion.png')
        dutils.save_plot(all_losses_combinedd,'loss_combinedd','loss_combinedd.png')
        dutils.save_plot(all_losses_combinedi,'loss_combinedi','loss_combinedi.png')
        dutils.img_save(masks_del,'masks_del.png')
        dutils.img_save(masks_ins,'masks_ins.png')
        dutils.pause()
    return masks_del * masks_ins


def iGOS_p(
        model,
        model_name, 
        init_mask,
        images,
        baselines,
        labels,
        size=28,
        iterations=15,
        ig_iter=20,
        L1=1,
        L2=20,
        alpha=1000,
        opt='LS',
        softmax=True,
        bilateral_sigma=0.01,
        **kwargs):
    """
        Similar idea to iGOS++, but generates explanation only using one mask (optimized for both insertion and deletion).

    :param model: The model to use for making predictions
    :param model_name: The model name to use for making predictions
    :param init_mask: The area in one cell of the K x K grid to use for initializing the mask
    :param image: The image to be explained
    :param baseline: The blured image as the baseline to use for making predictions
    :param label: The predicted class index of this image
    :param size: The size of the predicted mask 
    :param iterations: The number of iterations required to predict the mask
    :param ig_iter: The step size of the integtated gradient accumulation
    :param L1: The hyperparameter for TV/BTV norm
    :param L2: The hyperparameter for TV/BTV norm
    :param alpha: The step size for updating the mask
    :param opt: The optimization algorithm
    :param softmax: The output function for the model
    :param kwargs:
    :return:
    """
    def regularization_loss(masks):
        return L1 * torch.mean(torch.abs(1-masks).view(masks.shape[0],-1), dim=1) +\
               L2 * bilateral_tv_norm(images, masks, tv_beta=2, sigma=bilateral_sigma)
               #TODO: default bilateral_sigma 0.01

    def loss_function(up_masks, masks, indices, noise=True):
        losses = -interval_score(
                    model,
                    model_name,
                    baselines[indices],
                    images[indices],
                    labels[indices],
                    up_masks,
                    ig_iter,
                    output_func,
                    noise
                    )
        losses += interval_score(
                    model,
                    model_name,
                    images[indices],
                    baselines[indices],
                    labels[indices],
                    up_masks,
                    ig_iter,
                    output_func,
                    noise
                    )
        return losses.sum(dim=1).view(-1) + regularization_loss(masks)

    masks = torch.ones((images.shape[0],1,size,size), dtype=torch.float32, device='cuda')
    masks = masks * init_mask.cuda()
    masks = Variable(masks, requires_grad=True)

    if softmax:
        output_func = softmax_output
    else:
        logit_output.original = torch.gather(torch.nn.Sigmoid()(model(images)), 1, labels.view(-1,1))
        output_func = logit_output

    if opt == 'NAG':
        cita=torch.zeros(1).cuda()

    for i in range(iterations):
        total_grads = torch.zeros(images.shape[0], 1, size, size, dtype=torch.float32).cuda()
        up_masks = upscale(masks)

        integrated_gradient(model, model_name, images, baselines, labels, up_masks, ig_iter, output_func)
        total_grads += masks.grad.clone()
        masks.grad.zero_()

        integrated_gradient(model, model_name, baselines, images, labels, up_masks, ig_iter, output_func)
        total_grads += -masks.grad.clone()
        masks.grad.zero_()

        losses = regularization_loss(masks)
        losses.sum().backward()
        total_grads += masks.grad.clone()
        masks.grad.zero_()

        if opt == 'LS':
            alphas = line_search(masks, total_grads, loss_function, alpha)
            masks.data -= total_grads * alphas
        
        if opt == 'NAG':
            e = i / (i + 3)
            cita_p = cita
            cita = masks.data - alpha * total_grads
            masks.data = cita + e * (cita - cita_p)

        masks.grad.zero_()
        masks.data.clamp_(0, 1)

    return masks
