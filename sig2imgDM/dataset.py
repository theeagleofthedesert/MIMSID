from torch.utils.data import Dataset


import random
import torch
import numpy as np
import scipy.io
import os
import matplotlib.pyplot as plt
import skimage
import random



class MyDataset(Dataset):
    def __init__(self, A_path, Y_path, gt_path):
        # load A
        
        A0 = torch.zeros(16*5*72,20,20,dtype=torch.complex128)
        for j in range(20):
            for i in range(20):
                path = "adc_data_"+str(20*j+i+1)+".mat"
                mat = scipy.io.loadmat(os.path.join(A_path,path))
                A_now = torch.from_numpy(mat["adcData1Complex"])
                A_now = torch.mean(A_now,dim=-1)
                #print(A_now.shape)
                A_now = A_now[:,4:9,:]
                A_new = A_now.reshape(-1,1)
                A0[:,j,i] = A_new.reshape(-1)
        
        
        A0 = A0.reshape(-1, 20**2)
        self.A = torch.concat([A0.real,A0.imag],dim=0)
        
        
        if Y_path!=None:
            self.Y = []
            self.gt = []
            char = ['0','1','2','A','B']

            for i in range(len(char)*3):

                test = scipy.io.loadmat(os.path.join(Y_path,"adc_data_test"+char[i//3]+"_"+str(i%3+1)+".mat"))
                #test = scipy.io.loadmat(os.path.join(Y_path,"adc_data_test"+str(i%6+1)+".mat"))
                test=torch.from_numpy(test["adcData1Complex"]).to(torch.complex64)
                test = torch.mean(test,dim=-1)
                test = test[:,4:9,:] # 5x5


                test = test.reshape(-1).unsqueeze(-1)
                s_test = torch.concat([test.real,test.imag],dim=0).to(self.A.dtype)
                #concat A and Y
                s_test = torch.concat([self.A,s_test],dim=1)
                s_test=s_test.permute(1,0)
                #print(s_test.shape)

                self.Y.append(s_test)

                gt = np.load(os.path.join('./datasets/gt_20x20', 'gt_'+char[i//3]+'.npy'))

                gt = torch.tensor(gt, dtype=self.A.dtype)
                gt = gt.reshape(-1)
                self.gt.append(gt)

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        return self.Y[idx], self.gt[idx]

    def get_A(self):
        return self.A


class GTDataset(Dataset):
    def __init__(self, gt_path):
        self.gt = []
        self.label = []
        char = ['0','1','2','A','B']
        
        for i in range(len(char)*3):
            gt = np.load(os.path.join(gt_path, 'gt_'+char[i//3]+'.npy'))
            gt = torch.tensor(gt, dtype=torch.float32)

            #gt = gt[4:16,4:16]

            gt = gt.reshape((1, 20, 20))
            # gt = (gt-0.5)*2       # Norm to 0~1 or -1~1
            for _ in range(100):
                x_shift = random.randint(-4, 4)
                y_shift = random.randint(-4, 4)
                #self.label.append(i//3)
                self.label.append(i)
                self.gt.append(torch.roll(gt, shifts=(x_shift, y_shift), dims=(1, 2)))

    def __len__(self):
        return len(self.gt)

    def __getitem__(self, idx):
        return self.gt[idx], self.label[idx]
    
class ToolDataset(Dataset):
    def __init__(self, gt_path):
        self.gt = []
        self.label = []
        #char = ['0','1','2','3','4','A','B','C','cup']
        char = ['0','1','2','A','B']
        for i in range(27):
            gt = np.load(os.path.join(gt_path, 'gt_'+char[i//3]+'.npy'))
            gt = torch.tensor(gt, dtype=torch.float32)
            
            #gt = gt[4:16,4:16]
            
            gt = gt.reshape((1, 20, 20))
            # gt = (gt-0.5)*2       # Norm to 0~1 or -1~1
            for _ in range(90):
                x_shift = random.randint(-4, 4)
                y_shift = random.randint(-8, 0)
                self.label.append(i//3)
                self.gt.append(torch.roll(gt, shifts=(x_shift, y_shift), dims=(1, 2)))

    def __len__(self):
        return len(self.gt)

    def __getitem__(self, idx):
        return self.gt[idx], self.label[idx]




def show_images(images, title=""):
    """Shows the provided images as sub-pictures in a square"""

    # Converting images to CPU numpy arrays
    if type(images) is torch.Tensor:
        images = images.detach().cpu().numpy()

    # Defining number of rows and columns
    fig = plt.figure(figsize=(8, 8))
    rows = int(len(images) ** (1 / 2))
    cols = round(len(images) / rows)

    # Populating figure with sub-plots
    idx = 0
    for r in range(rows):
        for c in range(cols):
            fig.add_subplot(rows, cols, idx + 1)

            if idx < len(images):
                plt.imshow(images[idx][0], cmap="gray")
                #np.save("./datasets/fmnist/"+str(idx)+".npy",images[idx][0])
                idx += 1
    fig.suptitle(title, fontsize=30)
    

    # Showing the figure
    plt.show()

def show_first_batch(loader):
    for batch in loader:
        show_images(batch[0], "Images in the first batch")
        break
    

def noise_mask(mask_size=12):
    mask=np.zeros((mask_size,mask_size))
    noise_mask=np.multiply(skimage.util.random_noise(mask,mode='salt'),np.random.rand(mask_size,mask_size))
    return noise_mask

def apply_mask(img,mask):
    img_l=img.shape[0]
    mask_l=mask.shape[0]
    edge=int((img_l-mask_l)/2)
    center=int(img_l/2)
    whole_mask=np.zeros((img_l,img_l))
    whole_mask[center-int(mask_l/2):center+int(mask_l/2),center-int(mask_l/2):center+int(mask_l/2)]=mask
    whole_mask=torch.tensor(whole_mask, dtype=torch.float32)
    x_shift = random.randint(-edge, edge)
    y_shift = random.randint(-edge, edge)
    whole_mask=torch.roll(whole_mask, shifts=(x_shift, y_shift), dims=(0, 1))
    
    return img+whole_mask