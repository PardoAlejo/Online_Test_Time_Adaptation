import os
import random
import torch
from torchvision import datasets, transforms
from torch.utils.data import TensorDataset, DataLoader

def initial_data(config):
    if config.model.method not in ['eta','eata', 'memo']:
        data = torch.randn((config.data.initial_data_size, 3, config.data.spatial_dim_size, config.data.spatial_dim_size))
        labs = torch.randint(0, 10, size=(config.data.initial_data_size,))
        dataset = TensorDataset(data, labs)
        data_loader = DataLoader(dataset, batch_size=config.run.initial_batch_size, shuffle=False)
        return data_loader
    else:
        return ImageNet_val_subset_data(data_dir=config.dataset.paths.imagenet, 
                                        batch_size=config.dataset.batch_size, 
                                        shuffle=config.dataset.shuffle, 
                                        subset_size=-1, 
                                        with_transforms=config.dataset.transforms, 
                                        return_dataset=config.dataset.transforms)

def ImageNet_C_data(corruption: str, 
                    level: int, 
                    data_dir: str, 
                    batch_size: int, 
                    shuffle: bool, 
                    with_transforms: bool = True, 
                    return_dataset: bool = False,
                    num_workers: int = 4):
        
    # define the data transform
    transform = transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load ImageNet-C dataset
    dataset = datasets.ImageFolder(os.path.join(data_dir, corruption, str(level)), transform=transform if with_transforms else None)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataset if return_dataset else data_loader


def ImageNet_val_subset_data(data_dir: str, 
                             batch_size: int, 
                             shuffle: bool, 
                             subset_size: int, 
                             with_transforms: bool = True, 
                             return_dataset:bool = False,
                             num_workers: int = 4):

    transform = transforms.Compose([transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    dataset = datasets.ImageFolder(os.path.join(data_dir,'val'), transform=transform if with_transforms else None)

    if subset_size!=-1:
        num_samples = len(dataset.targets)
        indices = list(range(num_samples))
        random.shuffle(indices)
        dataset.samples = [dataset.samples[i] for i in indices[:subset_size]]
        dataset.targets = [dataset.targets[i] for i in indices[:subset_size]]
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    if return_dataset: 
        return dataset
    return data_loader

class ImageNetCorruption_pth(datasets.ImageFolder):
    def __init__(self, imagenet_path, 
                 imagenetc_dataroot, 
                 corruption_name, 
                 level=5, 
                 transform=None, 
                 is_carry_index=False):
        
        super().__init__(os.path.join(imagenet_path, 'val'), transform=transform)
        self.imagenetc_dataroot = imagenetc_dataroot
        self.corruption_name = corruption_name
        self.transform = transform
        self.is_carry_index = is_carry_index
        self.load_data()
    
    def load_data(self):
        self.data = torch.load(os.path.join(self.imagenetc_dataroot, self.corruption_name + '.pth')).numpy()
        self.target = [i[1] for i in self.imgs]
        return
    
    def __getitem__(self, index):
        img = self.data[index, :, :, :]
        target = self.target[index]
        if self.transform is not None:
            img = self.transform(img)
        if self.is_carry_index:
            img = [img, index]
        return img, target
    
    def __len__(self):
        return self.data.shape[0]

def ImageNet_R_data( data_dir: str, 
                    batch_size: int, 
                    shuffle: bool, 
                    with_transforms: bool = True, 
                    return_dataset: bool = False,
                    num_workers: int = 4):
        
    # define the data transform
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load ImageNet-C dataset
    dataset = datasets.ImageFolder(os.path.join(data_dir), transform=transform if with_transforms else None)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataset if return_dataset else data_loader

def get_dataloader(config):
    if config.evaluation.corruption == 'val':
        return ImageNet_val_subset_data(data_dir=config.dataset.paths.imagenet, 
                                        batch_size=config.dataset.batch_size, 
                                        shuffle=config.dataset.shuffle, 
                                        subset_size=-1, 
                                        with_transforms=config.dataset.transforms, 
                                        return_dataset=config.dataset.r_dataset,
                                        num_workers=config.run.num_workers
                                        )
    if config.dataset.name == 'imagenetc':
        corrupted_dataloader  = ImageNet_C_data(data_dir=config.dataset.paths.imagenetc,  
                                                batch_size=config.dataset.batch_size, 
                                                shuffle=config.dataset.shuffle, 
                                                with_transforms=config.dataset.transforms, 
                                                return_dataset=config.dataset.r_dataset,
                                                corruption=config.evaluation.corruption, 
                                                level=config.evaluation.level,
                                                num_workers=config.run.num_workers
                                                )

    elif config.dataset.name == 'imagenet3dcc': 
        # The same procedure as for ImageNetC should work. The only difference is using `args.imagenet3dcc_path` instead of `args.imagenetc_path`
        corrupted_dataloader = ImageNet_C_data(cdata_dir=config.dataset.paths.imagenetc,  
                                                batch_size=config.dataset.batch_size, 
                                                shuffle=config.dataset.shuffle, 
                                                with_transforms=config.dataset.transforms, 
                                                return_dataset=config.dataset.r_dataset,
                                                corruption=config.evaluation.corruption, 
                                                level=config.evaluation.level,
                                                num_workers=config.run.num_workers
                                                )
    elif config.dataset.name == 'imagenetr':
        corrupted_dataloader = ImageNet_R_data(data_dir=config.dataset.paths.imagenetc,  
                                                batch_size=config.dataset.batch_size, 
                                                shuffle=config.dataset.shuffle, 
                                                with_transforms=config.dataset.transforms, 
                                                return_dataset=config.dataset.r_dataset,
                                                num_workers=config.run.num_workers
                                                )

    return corrupted_dataloader
    
def get_cp(config):
    # Episodic Experiments
    if config.dataset.name == 'imagenetr':
        return ['imagenetr']
    
    if config.evaluation.corruption not in ['all', 'all_ordered']:
        return [config.evaluation.corruption]
    
    # Continual Experiments
    if config.dataset.name == 'imagenetc':
        corrupts = ['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur', \
                        'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog', 'brightness', 'contrast', \
                        'elastic_transform', 'pixelate', 'jpeg_compression']
    else:
        corrupts = ['bit_error', 'color_quant', 'far_focus', 'flash', 'fog_3d', 'h265_abr', 'h265_crf', \
                        'iso_noise', 'low_light', 'near_focus', 'xy_motion_blur', 'z_motion_blur']
        
    if config.evaluation.corruption == 'all': #random order of all corruptions (experiment in Appendix)
        random.shuffle(corrupts)
    
    # Add clean validation set.
    if config.evaluation.test_val:
        corrupts = [*corrupts,'val']
        
    return corrupts 
