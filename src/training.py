import torch.optim as optim

from tqdm import tqdm 
import datetime

LEARNING_RATE = 0.0001
EPOCHS = 20

def validate_accuracy(model, val_loader):
    pass

def loop(model, train_loader, val_loader):
    optimizer = optim.SGD(solo_model.parameters(), lr=LEARNING_RATE, momentum=0.9)
    loss_fn = nn.CrossEntropyLoss() 

    print("Beginning training")

    for epoch in range(1, EPOCHS + 1):
        total_loss = 0.0
        total_val_loss = 0.0
        best_loss = 9999
        
        #Train
        for (imgs, labels) in tqdm(train_loader, desc="Training"):
            print(imgs.get_device())

            model.train(True)

            out = model(imgs)
            loss = loss_fn(out, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        #Validate
        for (val_imgs, val_labels) in tqdm(val_loader, desc="Validation"):
            
            model.eval()

            val_out = model(val_imgs)
            val_loss = loss_fn(val_out, val_labels)

            total_val_loss += val_loss.item()

        epoch_val_loss = total_val_loss / len(val_loader)
        epoch_loss = total_loss / len(train_loader)
            
        #Save the best model
        if epoch_val_loss < best_loss:
            best_loss = epoch_val_loss
            torch.save(model.state_dict(), "data/" + f"MNIST_{target}.pth")

        # if epoch == 1 or epoch % 10 == 0:
        now = datetime.datetime.now()
        print(f"{now}\nEpoch {epoch}\ntr_loss {epoch_loss:.5}\nval_loss {epoch_val_loss:.5}\n")
