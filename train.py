from models_v02 import *
from utils import *


dataset = load_dataset('Cora')
data = dataset[0] 

def train():
      model.train()
      optimizer.zero_grad() 
      # Use all data as input, because all nodes have node features
      if "V2" not in md : out = model(data.x, data.edge_index) 
      else: out = model(data.x, data.edge_index, attributes)  
      # Only use nodes with labels available for loss calculation --> mask
      loss = criterion(out[data.train_mask], data.y[data.train_mask])  
      loss.backward() 
      optimizer.step()
      return loss

def test():
      model.eval()
      if "V2" not in md : out = model(data.x, data.edge_index) 
      else: out = model(data.x, data.edge_index, attributes) 
      # Use the class with highest probability.
      pred = out.argmax(dim=1)  
      # Check against ground-truth labels.
      test_correct = pred[data.test_mask] == data.y[data.test_mask]  
      # Derive ratio of correct predictions.
      test_acc = int(test_correct.sum()) / int(data.test_mask.sum())  
      return test_acc


for _ in range(5):
    
    
    if _ == 0:
        at = 'all'
    elif _ == 1:
        at = 'abs'
    elif _ == 2:
        at = 'concat'
    elif _ == 4:
        at = 'had'
    else: 
        at = 'cos'
    attributes = edge_attributes(at, dataset[0].edge_index, dataset[0].x)
    
    print(44*'==')
    print(f"Attribute type: {at}")
    
    
    model_1 =  GCN(input_=dataset.num_features, hidden_channels=16, output_=dataset.num_classes)
    model_2 =  GAT(input_=dataset.num_features, hidden_channels=8, output_=dataset.num_classes)
    model_3 =  GCN_V2(input_=dataset.num_features, hidden_channels=16, edge_dim=attributes.shape[1], output_=dataset.num_classes)
    model_4 =  GAT_V2(input_=dataset.num_features, hidden_channels=8, edge_dim=attributes.shape[1], output_=dataset.num_classes)
    
    
    models = {}

    models = {"GCN": model_1, "GAT": model_2, "GCN_V2": model_3, "GAT_V2": model_4}
    
    # Initialize Optimizer
    learning_rate = 0.005
    decay = 5e-4
    # Define loss function (CrossEntropyLoss for Classification Problems with 
    # probability distributions)
    criterion = torch.nn.CrossEntropyLoss()
    
    for md in models:
        model = models[md]
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        data = data.to(device)
        optimizer = torch.optim.Adam(model.parameters(), 
                                lr=learning_rate, 
                                weight_decay=decay)

        best_loss = float('inf')
        best_epoch = 0
        checkpoint_path = "best_model.ckpt"

        losses = []
        for epoch in range(0, 1001):
            loss = train()
            losses.append(loss)

            if loss < best_loss:
                    best_loss = loss
                    best_epoch = epoch
                    checkpoint_path = f"checkpoints_v2/model_{md}_epoch_{epoch}_loss_{loss:.4f}_{at}.ckpt"  # dynamic filename
            
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss
                    }, checkpoint_path)
            
            if epoch % 200 == 0:
                print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')

        test_acc = test()
        print(f"Model: {md}")
        print(f'Test Accuracy: {test_acc:.4f}')
        
