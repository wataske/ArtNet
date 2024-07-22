import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torchvision.models import VGG19_Weights
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image
import os
import glob
from torch.utils.tensorboard import SummaryWriter
import sys
from torch.utils.checkpoint import checkpoint
from datasets import load_dataset
from tqdm import tqdm



class ImageDataset(Dataset):
    def __init__(self, hf_dataset):
        self.hf_dataset = hf_dataset
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),  # 画像サイズを統一
            transforms.ToTensor(),  # テンソルに変換
        ])

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        
        item = self.hf_dataset[idx]
        image = self.transform(item["image"])  # 前処理を適用
        # アーティストのインデックスを取得
        label = item['artist']
        return image, label
class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args
    def forward(self,x):
        return x.view(self.shape)

class VGGFeatures(nn.Module):
    def __init__(self):
        super(VGGFeatures, self).__init__()
        self.vgg = models.vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features
        self.layers = {'0': 'conv1_1',
                       '5': 'conv2_1',
                       '10': 'conv3_1',
        }
                    #    '19': 'conv4_1',
                    #    '21': 'conv4_2',
                    #    '28': 'conv5_1'}
        for param in self.vgg.parameters():
            param.requires_grad_(False)

    def forward(self, x):
        features = {}
        for name, layer in self.vgg._modules.items():
            x = layer(x)
            if name in self.layers:
                features[self.layers[name]] = x.clone()
        return features
# グラム行列計算
def gram_matrix(y):
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (h * w)
    return gram

# ネットワーク定義
class ArtNet(nn.Module):
    def __init__(self, feature_layers, fc_out_dim=128, img_size=64, img_channels=3,cls_num=129):
        super(ArtNet, self).__init__()
        self.img_size = img_size
        self.extractor =VGGFeatures()#VGG16で特徴を抽出する部分
        self.fc1 = nn.Linear(86016, fc_out_dim) #抽出した複数の特徴を1つのベクトルに圧縮してfc_out_dim=128次元に圧縮
        self.fc2= nn.Linear(fc_out_dim,cls_num)
        # self.softmax     = nn.LogSoftmax( dim = 1 )
    def forward(self, x):
        
        features = self.extractor(x)
        gram_features = [gram_matrix(f).view(f.size(0), -1) for f in features.values()]
        concatenated_features = torch.cat(gram_features, dim=1)
        compressed = self.fc1(concatenated_features)
        compressed=compressed/(torch.norm(compressed))
        output = self.fc2(compressed)
        # output_softmax=self.softmax(output)
        return output,compressed


# 学習部分
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ArtNet(feature_layers=["0", "5", "10"]).to(device)
    dataset = load_dataset("huggan/wikiart")
    train_test_split=dataset["train"].train_test_split(test_size=0.1,seed=14)
    train_dataset=ImageDataset(train_test_split["train"])
    valid_dataset=ImageDataset(train_test_split["test"])
    train_dataloader=DataLoader(train_dataset,batch_size=512,shuffle=True,drop_last=True)
    valid_dataloader=DataLoader(valid_dataset,batch_size=512,shuffle=False,drop_last=True)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    
    num=0
    # model.load_state_dict(torch.load("art-0.pth"))
    model_save_path = f"art-{num}.pth"
    epochs = 2000
    i=0
    w=0.1 #重み
    save_loss=1000000
    for epoch in range(epochs):
            writer = SummaryWriter(log_dir=f"log/art-{num}")
            for batch_idx, (imgs, label) in tqdm(enumerate(train_dataloader),total=len(train_dataloader)):
                model.train()
                imgs = imgs.to(device)
                label=label.to(device)
                optimizer.zero_grad()
                outputs,_= model(imgs)
                train_loss=criterion(outputs,label)
                writer.add_scalars('loss/',{"train":train_loss.item()},epoch)
                train_loss.backward()
                optimizer.step()
                
            for batch_idx, (imgs, label) in tqdm(enumerate(valid_dataloader),total=len(valid_dataloader)):
                with torch.no_grad():
                    
                    model.eval()
                    imgs=imgs.to(device)
                    label=label.to(device)
                    outputs,_=model(imgs)


                    valid_loss=criterion(outputs,label)
                    writer.add_scalars('loss/',{"valid":valid_loss.item()},epoch)
            print("--------------------------------------------------------------------")
            print(f"Epoch [{epoch+1}/{epochs}] Train_Loss: {train_loss.item()}")
            print(f"Epoch [{epoch+1}/{epochs}] Valid_Loss: {valid_loss.item()}")

            # モデルの保存
            if save_loss >= valid_loss: # 10エポックごとにモデルを保存
                torch.save(model.state_dict(), model_save_path)
                save_loss=valid_loss
                print(f"Model saved to {model_save_path}")

    writer.close()
