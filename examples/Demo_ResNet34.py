from ResNet34 import my_resnet34
from torch_combinators import reset_weights

from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

class HealthyOnlyDataset(ImageFolder):
  def find_classes(self, directory):
    # Get all classes, then filter to only 'healthy' folders
    classes, class_to_idx = super().find_classes(directory)
    healthy_classes = [c for c in classes if "healthy" in c.lower()]
    healthy_class_to_idx = {c: i for i, c in enumerate(healthy_classes)}
    return healthy_classes, healthy_class_to_idx



def load_dataset(path):
    transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406),
                         (0.229, 0.224, 0.225)),
    ])

    #dataset1 = HealthyOnlyDatasetWithCache(root=r+"/train", transform=transform)
    dataset1 = HealthyOnlyDataset(root=path+"/train", transform=transform)
    dataset2 = HealthyOnlyDataset(root=path+"/test", transform=transform)

    # keep 0 for CPU — avoids multiprocessing overhead
    train_loader = DataLoader(dataset1,batch_size=64,shuffle=True, num_workers=2)
    test_loader = DataLoader(dataset2,batch_size=64,shuffle=True, num_workers=4)

    print([a==b           for a,b in zip(dataset1.classes,dataset2.classes)])  
    print(len(dataset1.classes),len(dataset2.classes))

    return train_loader,test_loader,dataset1.classes


def trainModel(train_loader,model,device,num_epochs = 10):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    num_batches=len(train_loader)

    for epoch in range(num_epochs):
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")

        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        b=1
        for images, labels in loop:
            images = images.to(device)   # <-- missing
            labels = labels.to(device) 

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
            # print(f"current loss {running_loss}, batch {b} of {num_batches}")
            loop.set_postfix(loss=f"{loss:.4f}", total_loss=f"{running_loss:.4f}", batch=f"{b}/{num_batches}")

            b+=1
            #if b>2:
            #    break # artificiall to exit quickly
        
        acc = 100. * correct / total
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1:2d}/{num_epochs} | Loss: {avg_loss:.4f} | Acc: {acc:.1f}%")
        #print("")
        
    return model

if __name__ == '__main__':
    train,test,classes=load_dataset("C:/LocalOwn/ComputerVision/datasets/PlantVillage")
    
    numClasses=len(classes)
    
    myModel=my_resnet34(numClasses)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    myModel = myModel.to(device)  
    reset_weights(myModel)
    
    model=trainModel(train,myModel,device,num_epochs=2)
    
    
    