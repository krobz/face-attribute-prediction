import torch
import torch.utils.data as data

from PIL import Image
import os
import os.path


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


class CelebA(data.Dataset):
    def __init__(self, root, ann_file, transform=None, target_transform=None, loader=default_loader):
        images = []
        targets = []
        
        for line in open(os.path.join(root, ann_file), 'r'):
            sample = line.split()
            if len(sample) != 41:
                raise(RuntimeError("# Annotated face attributes of CelebA dataset should not be different from 40"))
            images.append(sample[0])
            targets.append([int(i) for i in sample[1:]])
        self.images = [os.path.join(root, 'img_align_celeba', img) for img in images]
        self.targets = targets
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
		
    def __getitem__(self, index):
        path = self.images[index]
        sample = self.loader(path)
        target = self.targets[index]
        target = torch.LongTensor(target)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        return len(self.images)


class ReadPrivateTestCelebA(data.Dataset):
#read 
    def __init__(self, root, transform=None, loader=default_loader):
        # save the file path, e.g. .\testset\Aaron_Eckhart\Aaron_Eckhart_0001.jpg
        self.filenames = []

        for dir_, _, files in os.walk(root):
            for file_name in files:
                dir_folder = os.path.relpath(dir_, root)
                dir_file = os.path.join(dir_folder, file_name)
                # if not file_name.endswith(".jpg"):
                if file_name[-3:]!='jpg':
                    continue
                self.filenames.append(dir_file)

        self.images = [os.path.join(root, img) for img in self.filenames]
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        filename = self.filenames[index]
        path = self.images[index]
        sample = self.loader(path)

        if self.transform is not None:
            sample = self.transform(sample)

        return sample, filename

    def __len__(self):
        return len(self.images)


class ReadPrivateTestCelebA_LABEL(data.Dataset):
#read 
    def __init__(self, root, ann_file, transform=None, target_transform=None, loader=default_loader):
        # save the file path, e.g. .\testset\Aaron_Eckhart\Aaron_Eckhart_0001.jpg
        self.filenames = []
        self.targets = []
        for dir_, _, files in os.walk(root):
            for file_name in files:
                dir_folder = os.path.relpath(dir_, root)
                dir_file = os.path.join(dir_folder, file_name)
                # if not file_name.endswith(".jpg"):
                if file_name[-3:]!='jpg':
                    continue
                self.filenames.append(dir_file)

        for line in open(os.path.join(root, ann_file), 'r'):
            sample = line.split()
            if len(sample) != 41:
                raise(RuntimeError("# Annotated face attributes of CelebA dataset should not be different from 40"))
            self.targets.append([int(i) for i in sample[1:]])
        self.images = [os.path.join(root, img) for img in self.filenames]
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        filename = self.filenames[index]
        path = self.images[index]
        sample = self.loader(path)
        target = self.targets[index]
        target = torch.LongTensor(target)
        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.transform is not None:
            sample = self.transform(sample)

        return sample, target

    def __len__(self):
        return len(self.images)

