import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

def tta_inference(inp, model, num_classes=8, scales=[1.0], flip=True):
    b, _, h, w = inp.size()
    preds = inp.new().resize_(b, num_classes, h, w).zero_().to(inp.device)
    for scale in scales:
        size = (int(scale*h), int(scale*w))
        resized_img = F.interpolate(inp, size=size, mode='bilinear', align_corners=True,)
        pred = model_inference(model, resized_img.to(inp.device), flip)
        pred = F.interpolate(pred, size=(h, w), mode='bilinear', align_corners=True,)
        preds += pred

    return preds/(len(scales))

def model_inference(model, image, flip=True):
    output = model(image)
    if flip:
        fimg = image.flip(2)
        output += model(fimg).flip(2)
        fimg = image.flip(3)
        output += model(fimg).flip(3)
        return output/3
    return output
    
def slide(model, scale_image, num_classes=6, crop_size=512, overlap=1/2, scales=[1.0], flip=True):

    N, C, H_, W_ = scale_image.shape
    print(f"Height: {H_} Width: {W_}")
    
    full_probs = torch.zeros((N, num_classes, H_, W_), device=scale_image.device) #
    count_predictions = torch.zeros((N, num_classes, H_, W_), device=scale_image.device) #

    h_overlap_length = int(overlap*crop_size)
    w_overlap_length = int(overlap*crop_size)

    h = 0
    slide_finish = False
    while not slide_finish:

        if h + crop_size <= H_:
            print(f"h: {h}")
            # set row flag
            slide_row = True
            # initial row start
            w = 0
            while slide_row:
                if w + crop_size <= W_:
                    print(f" h={h} w={w} -> h'={h+crop_size} w'={w+crop_size}")
                    patch_image = scale_image[:, :, h:h+crop_size, w:w+crop_size]
                    #
                    patch_pred_image = tta_inference(patch_image, model, num_classes=num_classes, scales=scales, flip=flip)
                    count_predictions[:,:,h:h+crop_size, w:w+crop_size] += 1
                    full_probs[:,:,h:h+crop_size, w:w+crop_size] += patch_pred_image

                else:
                    print(f" h={h} w={W_-crop_size} -> h'={h+crop_size} w'={W_}")
                    patch_image = scale_image[:, :, h:h+crop_size, W_-crop_size:W_]
                    #
                    patch_pred_image = tta_inference(patch_image, model, num_classes=num_classes, scales=scales, flip=flip)
                    count_predictions[:,:,h:h+crop_size, W_-crop_size:W_] += 1
                    full_probs[:,:,h:h+crop_size, W_-crop_size:W_] += patch_pred_image
                    slide_row = False

                w += w_overlap_length

        else:
            print(f"h: {h}")
            # set last row flag
            slide_last_row = True
            # initial row start
            w = 0
            while slide_last_row:
                if w + crop_size <= W_:
                    print(f"h={H_-crop_size} w={w} -> h'={H_} w'={w+crop_size}")
                    patch_image = scale_image[:,:,H_-crop_size:H_, w:w+crop_size]
                    #
                    patch_pred_image = tta_inference(patch_image, model, num_classes=num_classes, scales=scales, flip=flip)
                    count_predictions[:,:,H_-crop_size:H_, w:w+crop_size] += 1
                    full_probs[:,:,H_-crop_size:H_, w:w+crop_size] += patch_pred_image

                else:
                    print(f"h={H_-crop_size} w={W_-crop_size} -> h'={H_} w'={W_}")
                    patch_image = scale_image[:,:,H_-crop_size:H_, W_-crop_size:W_]
                    #
                    patch_pred_image = tta_inference(patch_image, model, num_classes=num_classes, scales=scales, flip=flip)
                    count_predictions[:,:,H_-crop_size:H_, W_-crop_size:W_] += 1
                    full_probs[:,:,H_-crop_size:H_, W_-crop_size:W_] += patch_pred_image

                    slide_last_row = False
                    slide_finish = True

                w += w_overlap_length

        h += h_overlap_length

    full_probs /= count_predictions

    return full_probs
    
def test(testloader, model, savedir, device):
    '''
    args:
        test_loaded for test dataset
        model: model
    return:
        mean,Iou,IoU class
    '''
    if not os.path.exists(savedir):
        os.mkdir(savedir)

    model.eval()
    total_batches = len(testloader)
    with torch.no_grad():
        for idx, batch in enumerate(testloader):
        
            # load data
            image, _,  name = batch
            image = image.to(device)
            N, C, H, W = image.shape

            # sliding eval
            output = slide(
                model=model,
                scale_image=image,
                num_classes=6,
                crop_size=512,
                overlap=1/2,
                scales=[0.75, 1.0, 1.25],
                flip=True)

            _, output = torch.max(output, 1)

            assert len(output.shape) == 3, f"Wrong shape!"
            # convert torch to array
            output = np.asarray(output.permute(1,2,0).data.cpu().numpy(), dtype=np.uint8)

            # input: [H, W, 3]
            imageout = decode_segmap(output.squeeze())

            # std output
            img_save_name = os.path.basename(name[0])
            img_save_name = os.path.splitext(img_save_name)[0]

            img_save_path = os.path.join(savedir, img_save_name+'_gt.png')
            imageout = Image.fromarray(imageout)
            imageout.save(img_save_path)
