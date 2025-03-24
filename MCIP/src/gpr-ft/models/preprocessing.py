from PIL import Image
import torchvision
from torchvision import transforms as pth_transforms

def open_and_preporcess(path, transform_fn):

    with open(path, 'rb') as f:
        
        img = Image.open(f)
        img = img.convert('RGB')

    img = transform_fn(img)
    return img

def get_ppc_fn(model_key, image_size, mode, include_open=True, means=(0.4850, 0.4560, 0.4060), stds=(0.2290, 0.2240, 0.2250)):

    resize_to = int(image_size * 1.125)
    ppc_fn = None
    if mode == "train":

        if "Vit" == model_key[:3] and "Mobile" not in model_key:
            train_transform = pth_transforms.Compose([
                pth_transforms.Resize((resize_to, resize_to),  interpolation=3),
                pth_transforms.RandomCrop(image_size),
                pth_transforms.RandomHorizontalFlip(p=0.5),
                torchvision.transforms.ColorJitter(0.4, 0.4, 0.4),
                pth_transforms.ToTensor(),
                pth_transforms.Normalize(mean=([0.5000, 0.5000, 0.5000]), std=([0.5000, 0.5000, 0.5000]))
            ])

        else:
            train_transform = pth_transforms.Compose([
                pth_transforms.Resize((resize_to, resize_to),  interpolation=3),
                pth_transforms.RandomCrop(image_size),
                pth_transforms.RandomHorizontalFlip(p=0.5),
                torchvision.transforms.ColorJitter(0.4, 0.4, 0.4),
                pth_transforms.ToTensor(),
                pth_transforms.Normalize(means, stds),
            ])

        print(train_transform)
        if include_open:
            ppc_fn = lambda path: open_and_preporcess(path, train_transform)
        else:
            ppc_fn = train_transform

    elif mode == "eval":

        
        if "Vit" == model_key[:3] and "Mobile" not in model_key:
            test_transform = pth_transforms.Compose([
                pth_transforms.Resize((resize_to, resize_to),  interpolation=3),
                pth_transforms.CenterCrop(image_size),
                pth_transforms.ToTensor(),
                pth_transforms.Normalize(mean=([0.5000, 0.5000, 0.5000]), std=([0.5000, 0.5000, 0.5000]))
            ])
        else:
            test_transform = pth_transforms.Compose([
                pth_transforms.Resize((resize_to, resize_to),  interpolation=3),
                pth_transforms.CenterCrop(image_size),
                pth_transforms.ToTensor(),
                pth_transforms.Normalize(means, stds),
            ])

        if include_open:
            ppc_fn = lambda path: open_and_preporcess(path, test_transform)
        else:
            ppc_fn = test_transform

    #print(model_key, mode, ppc_fn)
    return ppc_fn

def create_ppc_fn_from_transform(transform):
    return lambda path: open_and_preporcess(path, transform)