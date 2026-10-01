import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from dataset import Autoencoder_dataset
from model import Autoencoder
import argparse
import torch.nn as nn
import numpy as np
from torch.utils.data import RandomSampler

torch.autograd.set_detect_anomaly(True)

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def cos_loss(network_output, gt):
    return 1 - F.cosine_similarity(network_output, gt, dim=0).mean()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--eval_step', type=int, default=1)
    parser.add_argument('--max_steps', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=16)

    parser.add_argument('--encoder_dims',
                    nargs = '+',
                    type=int,
                    default=[256, 128, 64, 32, 3],
                    )
    parser.add_argument('--decoder_dims',
                    nargs = '+',
                    type=int,
                    default=[16, 32, 64, 128, 256, 256, 512],
                    )
    parser.add_argument('--dataset_name', type=str, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required; refusing autoencoder training on CPU')
    if (args.max_steps is not None and args.max_steps <= 0) or args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError('max_steps and batch_size must be positive; num_workers must be nonnegative')
    dataset_path = args.dataset_path
    num_epochs = args.num_epochs
    data_dir = f"{dataset_path}"
    os.makedirs(f'ckpt/{args.dataset_name}', exist_ok=True)

    all_scenes = np.asarray(sorted(os.listdir(data_dir)))
    if len(all_scenes) < 2:
        raise ValueError('autoencoder training requires at least two scenes for the scene-level split')
    print(all_scenes)
    indices = np.random.permutation(len(all_scenes))
    train_indices = indices[:int(0.9*len(indices))]
    test_indices = indices[int(0.9*len(indices)):]
    print('Loading train set')
    print('Train indices: ', train_indices)
    train_dataset = Autoencoder_dataset(data_dir, indices=train_indices, subsample=20)
    print(len(train_dataset))
    print('Loading test set')
    print('Test indices: ', test_indices)
    test_dataset = Autoencoder_dataset(data_dir, indices=test_indices, subsample=20)
    print(len(test_dataset))

    random_sampler_train = RandomSampler(train_dataset)
    random_sampler_test = RandomSampler(test_dataset)


    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sampler=random_sampler_train,
        drop_last=False,
        pin_memory=True,
        persistent_workers=args.num_workers > 0
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sampler=random_sampler_test,
        drop_last=False  
    )
    
    encoder_hidden_dims = args.encoder_dims
    decoder_hidden_dims = args.decoder_dims

    model = Autoencoder()
    model = nn.DataParallel(model).cuda()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    logdir = f'ckpt/{args.dataset_name}'

    best_eval_loss = 100.0
    best_epoch = 0
    global_step = 0
    for epoch in tqdm(range(num_epochs)):
        model.train()
        epoch_loss = []
        epoch_cosloss = []
        epoch_l2loss = []
        for idx, feature in enumerate(train_loader):
            data = feature.cuda()
            outputs = model(data)
            
            l2loss = l2_loss(outputs, data) 
            cosloss = cos_loss(outputs, data)
            loss = l2loss + cosloss

            epoch_loss.append(loss.detach().cpu().numpy())
            epoch_l2loss.append(l2loss.detach().cpu().numpy())
            epoch_cosloss.append(cosloss.detach().cpu().numpy())

            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            global_step += 1
            if args.max_steps is not None and global_step >= args.max_steps:
                break
        tqdm.write('Epoch: {} Loss: {:.4f}, mse: {:.4f}, cos: {:.4f}'.format(str(epoch), np.mean(epoch_loss), np.mean(epoch_l2loss), np.mean(epoch_cosloss)))
        

        if epoch % args.eval_step == 0:
            eval_loss = 0.0
            eval_l2loss = 0.0
            eval_cosloss = 0.0
            count = 0
            model.eval()
            for idx, feature in enumerate(test_loader):
                data = feature.cuda()
                with torch.no_grad():
                    outputs = model(data) 
                l2loss = l2_loss(outputs, data)
                cosloss = cos_loss(outputs, data)
                loss = l2loss + cosloss
                eval_loss += loss
                eval_l2loss += l2loss
                eval_cosloss += cosloss
                count += 1
            eval_loss = eval_loss / count
            eval_l2loss = eval_l2loss / count
            eval_cosloss = eval_cosloss / count

            print("eval_loss:{:.8f}, l2loss: {:.8f}, cossloss: {:.8f}".format(eval_loss, eval_l2loss, eval_cosloss))
            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                best_epoch = epoch
                torch.save(model.state_dict(), f'ckpt/{args.dataset_name}/best_ckpt.pth')
        if args.max_steps is not None and global_step >= args.max_steps:
            break
            
    print(f"best_epoch: {best_epoch}")
    print("best_loss: {:.8f}".format(best_eval_loss))
