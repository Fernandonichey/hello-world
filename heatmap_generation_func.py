import argparse
import cv2
import numpy as np
import torch
from PIL import Image
from pathlib import Path
from torchvision import models
import os
from pytorch_grad_cam import GradCAM, \
    ScoreCAM, \
    GradCAMPlusPlus, \
    AblationCAM, \
    XGradCAM, \
    EigenCAM, \
    EigenGradCAM, \
    LayerCAM, \
    FullGrad
from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import show_cam_on_image, \
    deprocess_image, \
    preprocess_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-cuda', action='store_true', default=False,
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument(
        '--image-path',
        type=str,
        default='./examples/both.png',
        help='Input image path')
    parser.add_argument('--aug_smooth', action='store_true',
                        help='Apply test time augmentation to smooth the CAM')
    parser.add_argument(
        '--eigen_smooth',
        action='store_true',
        help='Reduce noise by taking the first principle componenet'
             'of cam_weights*activations')
    parser.add_argument('--method', type=str, default='gradcam',
                        choices=['gradcam', 'gradcam++',
                                 'scorecam', 'xgradcam',
                                 'ablationcam', 'eigencam',
                                 'eigengradcam', 'layercam', 'fullgrad'],
                        help='Can be gradcam/gradcam++/scorecam/xgradcam'
                             '/ablationcam/eigencam/eigengradcam/layercam')

    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print('Using GPU for acceleration')
    else:
        print('Using CPU for computation')

    return args


def cam_model_initialize(model, method=None, use_cuda=False):
    '''

    '''
    methods = \
        {"gradcam": GradCAM,
         "scorecam": ScoreCAM,
         "gradcam++": GradCAMPlusPlus,
         "ablationcam": AblationCAM,
         "xgradcam": XGradCAM,
         "eigencam": EigenCAM,
         "eigengradcam": EigenGradCAM,
         "layercam": LayerCAM,
         "fullgrad": FullGrad}
    target_layers = [model.final_layer] # hrnet_w32
    # target_layers = [model.conv_head]  # efficientnet-b6-ns
    cam_algorithm = methods[method]
    return cam_algorithm(model=model,
                         target_layers=target_layers,
                         use_cuda=use_cuda)



def cam_generate(cam, use_cuda=True, eigen_smooth=False, aug_smooth=False, file_name_list=None):
    """ python cam.py -image-path <path_to_image>
      Example usage of loading an image, and computing:
          1. CAM
          2. Guided Back Propagation
          3. Combining both
      """
    # args = get_args()
    # model.train()
    # methods = \
    #     {"gradcam": GradCAM,
    #      "scorecam": ScoreCAM,
    #      "gradcam++": GradCAMPlusPlus,
    #      "ablationcam": AblationCAM,
    #      "xgradcam": XGradCAM,
    #      "eigencam": EigenCAM,
    #      "eigengradcam": EigenGradCAM,
    #      "layercam": LayerCAM,
    #      "fullgrad": FullGrad}

    # model = models.resnet50(pretrained=True)

    # Choose the target layer you want to compute the visualization for.
    # Usually this will be the last convolutional layer in the model.
    # Some common choices can be:
    # Resnet18 and 50: model.layer4
    # VGG, densenet161: model.features[-1]
    # mnasnet1_0: model.layers[-1]
    # You can print the model to help chose the layer
    # You can pass a list with several target layers,
    # in that case the CAMs will be computed per layer and then aggregated.
    # You can also try selecting all layers of a certain type, with e.g:
    # from pytorch_grad_cam.utils.find_layers import find_layer_types_recursive
    # find_layer_types_recursive(model, [torch.nn.ReLU])

    # target_layers = [model.final_layer]

    # We have to specify the target we want to generate
    # the Class Activation Maps for.
    # If targets is None, the highest scoring category (for every member in the batch) will be used.
    # You can target specific categories by
    # targets = [e.g ClassifierOutputTarget(281)]
    # targets = None

    targets = None
    # Using the with statement ensures the context is freed, and you can
    # recreate different CAM objects in a loop.
    # cam_algorithm = methods[method]
    # with cam_algorithm(model=model,
    #                    target_layers=target_layers,
    #                    use_cuda=use_cuda) as cam:
    # AblationCAM and ScoreCAM have batched implementations.
    # You can override the internal batch size for faster computation.
    cam.batch_size = 1
    for i in range(len(file_name_list)):

        filename = file_name_list[i][0]
        # rgb_img = cv2.imread(filename, 1)[:, :, ::-1]
        bgr_img = cv2.imdecode(np.fromfile(filename,dtype =np.uint8),cv2.IMREAD_COLOR)
        # rgb_img = Image.open(filename).convert('RGB')
        # rgb_img = np.array(rgb_img.resize((512, 512)))
        rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        rgb_img = cv2.resize(rgb_img, (512, 512))
        rgb_img = np.float32(rgb_img) / 255
        input_tensor = preprocess_image(rgb_img,
                                        mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
        if use_cuda:
            input_tensor = input_tensor.cuda()
        grayscale_cam = cam(input_tensor=input_tensor,
                            targets=targets,
                            aug_smooth=aug_smooth,
                            eigen_smooth=eigen_smooth)

        # Here grayscale_cam has only one image in the batch
        grayscale_cam = grayscale_cam[0, :]
        # below used for thresh the heatmap
        grayscale_cam = grayscale_cam * (grayscale_cam > 0.3)

        '''show_cam_on_img function update with below '''

        heatmap = cv2.applyColorMap(np.uint8(255 * grayscale_cam), cv2.COLORMAP_JET)

        heatmap = np.float32(heatmap) / 255
        #
        if np.max(rgb_img) > 1:
            raise Exception(
                "The input image should np.float32 in the range [0, 1]")

        cam_img = heatmap * 0.2 + rgb_img
        cam_img = cam_img / np.max(cam_img)
        cam_image = np.uint8(255 * cam_img)
        ############################################
        # cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

        # cam_image is RGB encoded whereas "cv2.imwrite" requires BGR encoding.
        # cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)
        # cv2.namedWindow('test', cv2.WINDOW_NORMAL)
        # cv2.imshow(cam_image)
        # cv2.waitKey(0)
        # gb_model = GuidedBackpropReLUModel(model=model, use_cuda=use_cuda)
        # gb = gb_model(input_tensor, target_category=None)
        #
        # cam_mask = cv2.merge([grayscale_cam, grayscale_cam, grayscale_cam])
        # cam_gb = deprocess_image(cam_mask * gb)
        # gb = deprocess_image(gb)
        folder =  './timm/images/effinet_heatmap/' + filename.split('/')[-2]
        if not os.path.exists(folder):
            os.mkdir(folder)
        write_imge_name = folder + '/' + filename.split('/')[-1].split('.')[0] + '_cam_0.4.jpg'
        # ret = cv2.imwrite(write_imge_name, cam_image)
        cv2.imencode('.jpg', cam_image)[1].tofile(write_imge_name)
        # cv2.imwrite(f'{method}_gb.jpg', gb)
        # cv2.imwrite(f'{method}_cam_gb.jpg', cam_gb)
    return True
    # cv2.imwrite(f'{method}_gb.jpg', gb)
    # cv2.imwrite(f'{method}_cam_gb.jpg', cam_gb)


if __name__ == '__main__':
    """ python cam.py -image-path <path_to_image>
    Example usage of loading an image, and computing:
        1. CAM
        2. Guided Back Propagation
        3. Combining both
    """
    
    '''
    # initialize the cam model
    cam = cam_model_initialize(model, method='gradcam', use_cuda = True)
    loader = create_loader(
        ImageDataset(args.data),
        input_size=config['input_size'],
        batch_size=args.batch_size,
        use_prefetcher=True,
        interpolation=config['interpolation'],
        mean=config['mean'],
        std=config['std'],
        num_workers=args.workers,
        crop_pct=1.0 if test_time_pool else config['crop_pct'])
    
        """
        def create_loader(
        dataset,
        input_size,
        batch_size,
        is_training=False,
        use_prefetcher=True,
        no_aug=False,
        re_prob=0.,
        re_mode='const',
        re_count=1,
        re_split=False,
        scale=None,
        ratio=None,
        hflip=0.5,
        vflip=0.,
        color_jitter=0.4,
        auto_augment=None,
        num_aug_repeats=0,
        num_aug_splits=0,
        interpolation='bilinear',
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
        num_workers=1,
        distributed=False,
        crop_pct=None,
        collate_fn=None,
        pin_memory=False,
        fp16=False,
        tf_preprocessing=False,
        use_multi_epochs_loader=False,
        persistent_workers=True,
        worker_seeding='all',
):
    re_num_splits = 0
    if re_split:
        # apply RE to second half of batch if no aug split otherwise line up with aug split
        re_num_splits = num_aug_splits or 2
    dataset.transform = create_transform(
        input_size,
        is_training=is_training,
        use_prefetcher=use_prefetcher,
        no_aug=no_aug,
        scale=scale,
        ratio=ratio,
        hflip=hflip,
        vflip=vflip,
        color_jitter=color_jitter,
        auto_augment=auto_augment,
        interpolation=interpolation,
        mean=mean,
        std=std,
        crop_pct=crop_pct,
        tf_preprocessing=tf_preprocessing,
        re_prob=re_prob,
        re_mode=re_mode,
        re_count=re_count,
        re_num_splits=re_num_splits,
        separate=num_aug_splits > 0,
    )

    sampler = None
    if distributed and not isinstance(dataset, torch.utils.data.IterableDataset):
        if is_training:
            if num_aug_repeats:
                sampler = RepeatAugSampler(dataset, num_repeats=num_aug_repeats)
            else:
                sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        else:
            # This will add extra duplicate entries to result in equal num
            # of samples per-process, will slightly alter validation results
            sampler = OrderedDistributedSampler(dataset)
    else:
        assert num_aug_repeats == 0, "RepeatAugment not currently supported in non-distributed or IterableDataset use"

    if collate_fn is None:
        collate_fn = fast_collate if use_prefetcher else torch.utils.data.dataloader.default_collate

    loader_class = torch.utils.data.DataLoader
    if use_multi_epochs_loader:
        loader_class = MultiEpochsDataLoader

    loader_args = dict(
        batch_size=batch_size,
        shuffle=not isinstance(dataset, torch.utils.data.IterableDataset) and sampler is None and is_training,
        num_workers=num_workers,
        sampler=sampler,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        drop_last=is_training,
        worker_init_fn=partial(_worker_init, worker_seeding=worker_seeding),
        persistent_workers=persistent_workers
    )
    try:
        loader = loader_class(dataset, **loader_args)
    except TypeError as e:
        loader_args.pop('persistent_workers')  # only in Pytorch 1.7+
        loader = loader_class(dataset, **loader_args)
    if use_prefetcher:
        prefetch_re_prob = re_prob if is_training and not no_aug else 0.
        loader = PrefetchLoader(
            loader,
            mean=mean,
            std=std,
            fp16=fp16,
            re_prob=prefetch_re_prob,
            re_mode=re_mode, 
            re_count=re_count,
            re_num_splits=re_num_splits
        )

    return loader
        """
    
    
    
    for batch_idx, (input, target) in enumerate(loader):

        ''' used for heatmap generate'''
        # cam_input_tensor = input[0]

        image_list = loader.dataset.parser.samples[batch_idx*len(input): (batch_idx+1)*len(input)]
        # filenames =  loader.dataset.filenames
        # cam_input_tensor = input[0].unsqueeze(0)
        cam_img = cam_generate(cam, file_name_list= image_list)
    '''
    pass

    # args = get_args()
    # methods = \
    #     {"gradcam": GradCAM,
    #      "scorecam": ScoreCAM,
    #      "gradcam++": GradCAMPlusPlus,
    #      "ablationcam": AblationCAM,
    #      "xgradcam": XGradCAM,
    #      "eigencam": EigenCAM,
    #      "eigengradcam": EigenGradCAM,
    #      "layercam": LayerCAM,
    #      "fullgrad": FullGrad}
    #
    # model = models.resnet50(pretrained=True)
    #
    # # Choose the target layer you want to compute the visualization for.
    # # Usually this will be the last convolutional layer in the model.
    # # Some common choices can be:
    # # Resnet18 and 50: model.layer4
    # # VGG, densenet161: model.features[-1]
    # # mnasnet1_0: model.layers[-1]
    # # You can print the model to help chose the layer
    # # You can pass a list with several target layers,
    # # in that case the CAMs will be computed per layer and then aggregated.
    # # You can also try selecting all layers of a certain type, with e.g:
    # # from pytorch_grad_cam.utils.find_layers import find_layer_types_recursive
    # # find_layer_types_recursive(model, [torch.nn.ReLU])
    # target_layers = [model.layer4]
    #
    # rgb_img = cv2.imread(args.image_path, 1)[:, :, ::-1]
    # rgb_img = np.float32(rgb_img) / 255
    # input_tensor = preprocess_image(rgb_img,
    #                                 mean=[0.485, 0.456, 0.406],
    #                                 std=[0.229, 0.224, 0.225])
    #
    #
    # # We have to specify the target we want to generate
    # # the Class Activation Maps for.
    # # If targets is None, the highest scoring category (for every member in the batch) will be used.
    # # You can target specific categories by
    # # targets = [e.g ClassifierOutputTarget(281)]
    # targets = None
    #
    # # Using the with statement ensures the context is freed, and you can
    # # recreate different CAM objects in a loop.
    # cam_algorithm = methods[args.method]
    # with cam_algorithm(model=model,
    #                    target_layers=target_layers,
    #                    use_cuda=args.use_cuda) as cam:
    #
    #     # AblationCAM and ScoreCAM have batched implementations.
    #     # You can override the internal batch size for faster computation.
    #     cam.batch_size = 32
    #     grayscale_cam = cam(input_tensor=input_tensor,
    #                         targets=targets,
    #                         aug_smooth=args.aug_smooth,
    #                         eigen_smooth=args.eigen_smooth)
    #
    #     # Here grayscale_cam has only one image in the batch
    #     grayscale_cam = grayscale_cam[0, :]
    #
    #     cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    #
    #     # cam_image is RGB encoded whereas "cv2.imwrite" requires BGR encoding.
    #     cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)
    #
    # gb_model = GuidedBackpropReLUModel(model=model, use_cuda=args.use_cuda)
    # gb = gb_model(input_tensor, target_category=None)
    #
    # cam_mask = cv2.merge([grayscale_cam, grayscale_cam, grayscale_cam])
    # cam_gb = deprocess_image(cam_mask * gb)
    # gb = deprocess_image(gb)
    #
    # cv2.imwrite(f'{args.method}_cam.jpg', cam_image)
    # cv2.imwrite(f'{args.method}_gb.jpg', gb)
    # cv2.imwrite(f'{args.method}_cam_gb.jpg', cam_gb)
