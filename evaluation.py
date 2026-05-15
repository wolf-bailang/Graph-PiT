import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import inception_v3
from PIL import Image
from scipy import linalg
from torch.nn.functional import adaptive_avg_pool2d
import pathlib

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, folder, transform=None):
        self.folder = folder
        self.transform = transform
        self.image_paths = sorted([os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(('png', 'jpg', 'jpeg'))])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        image = Image.open(path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image

def get_inception_model():
    """载入预训练的 InceptionV3 模型"""
    inception_model = inception_v3(pretrained=True, transform_input=False)
    inception_model.fc = torch.nn.Identity()
    return inception_model.eval()

def get_activations(files_path, model, batch_size=50, dims=2048, device='cpu'):
    """计算给定目录下所有图像的 InceptionV3 激活"""
    model.to(device)
    
    transform = transforms.Compose([
        transforms.Resize(299),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = ImageDataset(files_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    pred_arr = np.empty((len(dataset), dims))

    start_idx = 0
    for batch in dataloader:
        batch = batch.to(device)
        with torch.no_grad():
            pred = model(batch)

        # 如果是 InceptionV3，我们需要处理辅助输出
        if isinstance(pred, tuple):
            pred = pred[0]

        # The output of the InceptionV3 model with `fc` layer as Identity is a 2D tensor of features (batch_size, 2048).
        # The adaptive_avg_pool2d is not needed as the pooling is already done within the model.
        pred = pred.cpu().numpy()
        
        pred_arr[start_idx:start_idx + pred.shape[0]] = pred
        start_idx = start_idx + pred.shape[0]

    return pred_arr

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """计算两个高斯分布之间的 Frechet 距离"""
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, 'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, 'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)

def calculate_fid(act1, act2):
    """计算 FID 分数"""
    mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)
    fid_value = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
    return fid_value

def calculate_iis(act1, act2):
    """计算 Image-Image Similarity (IIS)"""
    from sklearn.metrics.pairwise import cosine_similarity
    
    # act1: real, act2: generated
    # For each generated image, find the max similarity to any real image
    cosine_sim = cosine_similarity(act2, act1)
    max_sim = np.max(cosine_sim, axis=1)
    iis_score = np.mean(max_sim)
    return iis_score



# python evaluation.py ./evaluation/real_images ./evaluation/generated_images

def main():
    parser = argparse.ArgumentParser(description='Calculate FID and IIS for generated images.')
    parser.add_argument('real_images_path', type=str, help='Path to the directory with real images.')
    parser.add_argument('generated_images_path', type=str, help='Path to the directory with generated images.')
    parser.add_argument('--batch-size', type=int, default=50, help='Batch size for processing images.')
    parser.add_argument('--device', type=str, default=None, help='Device to use (e.g., "cuda:0" or "cpu").')

    args = parser.parse_args()

    if args.device is None:
        device = "cuda:6" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    print(f"Using device: {device}")

    real_path = pathlib.Path(args.real_images_path)
    generated_path = pathlib.Path(args.generated_images_path)

    if not real_path.exists():
        print(f"Error: Real images path does not exist: {real_path}")
        return
    if not generated_path.exists():
        print(f"Error: Generated images path does not exist: {generated_path}")
        return

    model = get_inception_model()

    print("Calculating activations for real images...")
    act_real = get_activations(str(real_path), model, args.batch_size, device=device)
    
    print("Calculating activations for generated images...")
    act_generated = get_activations(str(generated_path), model, args.batch_size, device=device)

    print("Calculating FID score...")
    fid_score = calculate_fid(act_real, act_generated)
    print(f"FID Score: {fid_score:.4f}")

    print("Calculating IIS score...")
    iis_score = calculate_iis(act_real, act_generated)
    print(f"IIS Score: {iis_score:.4f}")


if __name__ == '__main__':
    main()