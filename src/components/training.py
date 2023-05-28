import torch.optim as optim
import torch.nn as nn
import torch

from matplotlib import pyplot as plt
from tqdm import tqdm 
import datetime

LEARNING_RATE = 0.00005
EPOCHS = 2

def validate_accuracy(model, val_loader):
    model.eval()
        
    for name, loader in [("val", val_loader)]:
        correct = 0
        total = 0

        with torch.no_grad():
            for imgs, labels in tqdm(loader, "Evaluating Performance"):
                outputs = model(imgs)

                # print("Out: " + str(outputs[0]))
                # print("Truth: " + str(labels[0]) + "\n")

                _, predicted = torch.max(outputs, dim=1)
                _, truth = torch.max(labels, dim=1)
                
                total += labels.shape[0]
                correct += int((predicted == truth).sum())

        print("Accuracy {}: {:.3f}".format(name , correct / total))


def training_loop(model, train_loader, val_loader, name):
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=0.1)
    loss_fn = nn.BCELoss() 

    print("Beginning training")

    for epoch in range(1, EPOCHS + 1):
        total_loss = 0.0
        total_val_loss = 0.0
        best_loss = 9999
        
        #Train
        for (imgs, labels) in tqdm(train_loader, desc="Training"):
            # #Uncomment to visualize data
            # plt.imshow(imgs[0].cpu().reshape(32, 32, 1))
            # plt.show()

            model.train(True)

            out = model(imgs)
            loss = loss_fn(out, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        #Validate
        for (val_imgs, val_labels) in tqdm(val_loader, desc="Validation"):
            model.train(False)

            val_out = model(val_imgs)
            val_loss = loss_fn(val_out, val_labels)

            total_val_loss += val_loss.item()

        epoch_val_loss = total_val_loss / len(val_loader)
        epoch_loss = total_loss / len(train_loader)
            
        #Save the best model
        if epoch_val_loss < best_loss:
            best_loss = epoch_val_loss
            torch.save(model.state_dict(), "data/" + f"{name}.pth")

        # if epoch == 1 or epoch % 10 == 0:
        now = datetime.datetime.now()
        print(f"{now}\nEpoch {epoch}\ntr_loss {epoch_loss:.5}\nval_loss {epoch_val_loss:.5}\n")
