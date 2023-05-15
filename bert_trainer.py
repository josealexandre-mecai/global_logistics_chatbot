from requirements import *

class Dataset(torch.utils.data.Dataset):

    def __init__(self, df, labels, tokenizer):

        self.labels = [labels[label] for label in df['intent']]
        
        self.texts = [tokenizer(text, 
                                padding='max_length', max_length=512, truncation=True,
                                return_tensors="pt") for text in df['text']]
        
    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.labels)

    def get_batch_labels(self, idx):
        # Fetch a batch of labels
        return np.array(self.labels[idx])

    def get_batch_texts(self, idx):
        # Fetch a batch of inputs
        return self.texts[idx]

    def __getitem__(self, idx):

        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)

        return batch_texts, batch_y
    

class BertClassifier(nn.Module):   
    
    def __init__(self, bert, dropout=0.2, classes=11): 
        
        super(BertClassifier, self).__init__() 
        
        self.bert = bert 
      
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
      
        # Relu activation function
        self.relu =  nn.ReLU() 
        
        # Dense layers
        self.fc1 = nn.Linear(768, 512)       
        self.fc2 = nn.Linear(512, 256)       
        self.fc3 = nn.Linear(256, classes)  # 11 classes     
        
        # Softmax activation function
        self.softmax = nn.LogSoftmax(dim=1)       
        
        
    # Define the forward pass
    def forward(self, input_ids, mask):      
        
        # Pass the inputs to the model  
        cls_hs = self.bert(input_ids=input_ids, attention_mask=mask)[0][:,0]
      
        x = self.fc1(cls_hs)
        x = self.relu(x)
        x = self.dropout(x)
      
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)      
        
        # Output layer
        x = self.fc3(x)
   
        # Apply softmax activation
        x = self.softmax(x)      
        
        return x

    def reset_weights(self):
      # Reset weights of classifier layers only
      for layer in [self.fc1, self.fc2, self.fc3]:
        layer.reset_parameters()


class BertTrainer:
    
  def __init__(self, intents_json, learning_rate=1e-2, epochs=10, early_stopping=(10, 'val_loss'), freeze=True, test_size=0.3, 
               deterministic=True):
    
    self.use_cuda = torch.cuda.is_available()
    self.device = torch.device("cuda" if self.use_cuda else "cpu")

    self.is_model_loaded = False

    self.DATA_PATH = '/usr/src/app/data/'
    self.LOG_PATH = '/usr/src/app/logs/'

    self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    self.arch = DistilBertModel.from_pretrained('distilbert-base-uncased')

    self.intents = json.loads(open(self.DATA_PATH + intents_json).read())
    self.df = self.json_to_df(intents_json)
    self.labels = self.get_labels(self.df)
    self.class_weights = self.get_class_weights(self.df)
    self.train_data, self.val_data = train_test_split(self.df, test_size=test_size, random_state=42, stratify=self.df['intent'])

    self.criterion = nn.NLLLoss(weight=self.class_weights) # negative log likelihood loss
    
    if freeze:
      for param in self.arch.parameters():
        param.requires_grad = False

    self.model = BertClassifier(self.arch, classes=len(self.labels))

    if self.use_cuda:
      self.model = self.model.cuda()
      self.criterion = self.criterion.cuda()

    # Store training stats
    self.training_stats = []
    
    self.early_stopping = early_stopping
    self.lr = learning_rate
    self.epochs = epochs

    # Set Random Seed
    if deterministic:
      self.random_seed(101)

  def get_dataloader(self, data, batch_size, shuffle=False):
    dataset = Dataset(data, self.labels, self.tokenizer)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader

  def train_step(self, loader=None):

    if loader is None:
      loader = self.train_dataloader

    total_train_acc = 0
    total_train_loss = 0

    self.model.train()

    for train_input, train_label in tqdm(loader):

      self.optimizer.zero_grad()
      self.model.zero_grad()

      train_label = train_label.to(self.device)
      mask = train_input['attention_mask'].to(self.device)
      input_ids = train_input['input_ids'].squeeze(1).to(self.device)

      output = self.model(input_ids, mask)
                
      batch_loss = self.criterion(output, train_label.long())
      total_train_loss += batch_loss.item()
                
      acc = (output.argmax(dim=1) == train_label).sum().item()
      total_train_acc += acc

      batch_loss.backward()

      # Clip the norm of the gradients to 1.0.
      # This is to help prevent the "exploding gradients" problem.
      torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

      self.optimizer.step()
      self.scheduler.step()

    avg_train_acc = total_train_acc / len(loader.dataset)
    avg_train_loss = total_train_loss / len(loader.dataset)
    
    return avg_train_acc, avg_train_loss


  def eval_step(self, model=None, loader=None):

    if loader is None:
      loader = self.val_dataloader

    total_val_acc = 0
    total_val_loss = 0

    if model is None:
      model = self.model
    
    model.eval()

    with torch.no_grad():

      for val_input, val_label in loader:

        val_label = val_label.to(self.device)
        mask = val_input['attention_mask'].to(self.device)
        input_ids = val_input['input_ids'].squeeze(1).to(self.device)

        output = model(input_ids, mask)

        batch_loss = self.criterion(output, val_label.long())
        total_val_loss += batch_loss.item()
                    
        acc = (output.argmax(dim=1) == val_label).sum().item()
        total_val_acc += acc

      avg_val_acc = total_val_acc / len(loader.dataset)
      avg_val_loss = total_val_loss / len(loader.dataset)
      
      return avg_val_acc, avg_val_loss


  def train(self, model_version=1.0, model_name='bert', train_data=None, val_data=None, fold=None, graph=None):

    if train_data is None:
      train_data = self.train_data

    if val_data is None:
      val_data = self.val_data

    self.MODEL_VERSION = model_version
    self.MODEL_NAME = model_name
    self.MODEL_PATH = os.path.join('/usr/src/app/models/', "model-{}-{}.pth".format(re.sub("\.", "_", str(self.MODEL_VERSION)),str(self.MODEL_NAME)))
    self.MODEL_PATH_ONNX = os.path.join('/usr/src/app/models/', "model-{}-{}.onnx".format(re.sub("\.", "_", str(self.MODEL_VERSION)),str(self.MODEL_NAME)))

    # Avoid weight leakage in cross-validation  
    if fold is not None:
      self.model.reset_weights()

    self.train_dataloader = self.get_dataloader(train_data, batch_size=16, shuffle=True)
    self.val_dataloader = self.get_dataloader(val_data, batch_size=2)
    self.optimizer = AdamW(self.model.parameters(), lr=self.lr)
    self.total_steps = len(self.train_dataloader) * self.epochs
 

    # The scheduler adjusts the learning rate during training to optimize the model's performance 
    # This scheduler linearly increases the learning rate during a "warmup" phase (num_warmup_steps) and 
    # then linearly decreases it during the rest of training (num_training_steps - num_warmup_steps).
    self.scheduler = get_linear_schedule_with_warmup(self.optimizer, 
                                                     num_warmup_steps = 0, 
                                                     num_training_steps = self.total_steps)


    # Track the best validation accuracy and save the corresponding model weights
    best_val_loss = None
    best_val_acc = None
    best_weights = None
    early_stopping_counter = 0
    stopped_epoch = None

    # Measure the total training time for the whole run.
    total_t0 = time.time()

    for epoch in range(self.epochs):

        # Measure how long the training epoch takes.
        t0 = time.time()

        avg_train_acc, avg_train_loss = self.train_step()
        avg_val_acc, avg_val_loss = self.eval_step()

        # Measure how long this epoch took.
        training_time = self.format_time(time.time() - t0)

        # Record all statistics from this epoch.

        if fold is None:

          self.training_stats.append(
            {
            'epoch': epoch + 1,
            'Train Loss': avg_train_loss,
            'Train Accuracy': avg_train_acc,
            'Val Loss': avg_val_loss,
            'Val Accuracy': avg_val_acc,
            'Training Time': training_time,
            }
          )

        else:

          self.training_stats.append(
            {
            'fold': fold + 1,
            'epoch': epoch + 1,
            'Train Loss': avg_train_loss,
            'Train Accuracy': avg_train_acc,
            'Val Loss': avg_val_loss,
            'Val Accuracy': avg_val_acc,
            'Training Time': training_time,
            }
          )

        print(f'Epochs: {epoch + 1} | Train Loss: {avg_train_loss:.3f} | Train Accuracy: {avg_train_acc:.3f} | Val Loss: {avg_val_loss:.3f} | Val Accuracy: {avg_val_acc:.3f}')

        if self.early_stopping is not None:
            patience, metric = self.early_stopping

            if metric not in ['val_loss', 'val_acc']:
                raise ValueError("Invalid metric. Choose between 'val_loss' and 'val_acc'.")

            if best_val_loss is None or best_val_acc is None:
                best_val_loss = avg_val_loss
                best_val_acc = avg_val_acc
                best_weights = self.model.state_dict()
            else:
                if (metric == 'val_loss' and avg_val_loss < best_val_loss) or (metric == 'val_acc' and avg_val_acc > best_val_acc):
                    best_val_loss = avg_val_loss
                    best_val_acc = avg_val_acc
                    best_weights = self.model.state_dict()
                    early_stopping_counter = 0
                else:
                    early_stopping_counter += 1
                    if early_stopping_counter >= patience:
                        print(f'Early stopping: validation {metric} did not improve for {patience} epochs')
                        stopped_epoch = epoch
                        break
        else:
            best_val_loss = avg_val_loss
            best_val_acc = avg_val_acc
            best_weights = self.model.state_dict()

    # Loading the model with the best weights
    self.model.load_state_dict(best_weights)
 
    runtime = self.format_time(time.time()-total_t0)

    if fold is None:

      # Save model
      torch.save(self.model.state_dict(), self.MODEL_PATH)

      # Update Log
      self.update_train_log(runtime, self.training_stats, self.epochs, best_val_acc, best_val_loss, self.MODEL_VERSION, self.MODEL_NAME)

      # Export the model to ONNX
      input_ids_example = torch.tensor([[1] * 512], dtype=torch.long).to(self.device)
      attention_mask_example = torch.tensor([[1] * 512], dtype=torch.long).to(self.device)

      torch.onnx.export(
        model=self.model,
        args=(input_ids_example, attention_mask_example),  # Pass the example inputs as arguments
        f=self.MODEL_PATH_ONNX,  # Output file path for the ONNX model
        opset_version=11,  # ONNX opset version
        do_constant_folding=True,  # Optimize constant folding
        input_names=['input_ids', 'attention_mask'],  # Names for the input tensors
        output_names=['output'],  # Names for the output tensors
        dynamic_axes={'input_ids': {0: 'batch_size'},  # Dynamic axes for variable-length input
                      'attention_mask': {0: 'batch_size'},
                      'output': {0: 'batch_size'}}
        )

    if stopped_epoch is not None:
        print(f'Training stopped at epoch {stopped_epoch + 1}')

    print(f'\nFinal Val Accuracy: {avg_val_acc:.3f} | Final Val Loss: {avg_val_loss:.3f}')
    print(f'Best Val Accuracy: {best_val_acc:.3f} | Best Val Loss: {best_val_loss:.3f}')
    print("Total training took {:} (h:mm:ss:ms)".format(runtime))

    if graph is not None:
      self.train_graph(self.training_stats, graph)

    return self.model, self.training_stats


  def predict(self, model_version, model_name, data):

    self.MODEL_VERSION = model_version
    self.MODEL_NAME = model_name
    self.MODEL_PATH = os.path.join('/usr/src/app/models/', "model-{}-{}.pth".format(re.sub("\.", "_", str(self.MODEL_VERSION)),str(self.MODEL_NAME)))

    if not os.path.exists(self.MODEL_PATH):
      raise ValueError(f"Model file '{self.MODEL_PATH}' does not exist")

    ### Implement ERROR_THRESHOLD ? ### 

    # Measure the total prediction time for the whole run.
    total_t0 = time.time()

    if not self.is_model_loaded:
      # Load the saved model
      self.model.load_state_dict(torch.load(self.MODEL_PATH, map_location=torch.device(self.device)))
      self.is_model_loaded = True

    self.model.eval()

    # Tokenize the input data
    if isinstance(data, str):
      data = [data]

    input = self.tokenizer(data, padding='max_length', max_length=512, truncation=True, return_tensors="pt")
    
    # Define input_ids and attention_mask
    input_ids = input['input_ids'].clone().detach().to(self.device)
    mask = input['attention_mask'].clone().detach().to(self.device)

    try:

      with torch.no_grad():

        output = self.model(input_ids, mask)
                             
        # Calculate the predicted class and probability
        _, predicted = torch.max(output, 1)
        softmax_output = torch.nn.functional.softmax(output, dim=1)
        probability, _ = torch.max(softmax_output, 1)

        predicted = predicted.cpu().numpy()
        probability = probability.cpu().numpy()

      runtime = self.format_time(time.time()-total_t0)
      self.update_predict_log(runtime, predicted[0], probability[0], data, model_version, model_name)
        
      return predicted[0], probability[0]

    except Exception as e:
      print(f"Error during prediction: {e}")
      return None, None

  
  def get_response(self, intent):

    try:
      for i in self.intents['intents']:
        if i['intent'] == list(self.labels.keys())[list(self.labels.values()).index(intent)]:
          result = random.choice(i['responses'])
          break
    except IndexError:
      result = "I don't understand!"

    return result


  def json_to_df(self, json_file):

    df = pd.DataFrame(columns=['intent', 'text'])

    # Load the JSON file
    with open(self.DATA_PATH + json_file, 'r') as f:
      data = json.load(f)
      data = data['intents']

    # Create a list of dictionaries, where each dictionary represents a row in the DataFrame
    for i in data:
      intent = i['intent']
      for t in i['text']:
        row = {'intent': intent, 'text': t}
        #df = df.append(row, ignore_index=True)
        df = pd.concat([df, pd.DataFrame(row, index=[0])], ignore_index=True)

    return df


  def get_labels(self, df):

    # Get unique intents from the dataset
    unique_intents = set(df['intent'])

    # Create a dictionary to map intents to indices
    labels = {intent: idx for idx, intent in enumerate(sorted(unique_intents))}

    return labels


  def get_class_weights(self, df):

    train_labels = df['intent']

    # Compute the class weights
    class_wts = compute_class_weight(class_weight='balanced', classes=np.unique(train_labels), y=train_labels)
    #print(class_wts)

    # Convert class weights to tensor
    weights = torch.tensor(class_wts, dtype=torch.float)
    weights = weights.to(self.device)

    return weights


  def find_lr(self, init_value=1e-8, final_value=10., beta=0.98, freeze=True, num_epochs=10):

    if freeze:
      for param in self.arch.parameters():
        param.requires_grad = False

    model = BertClassifier(self.arch, classes=len(self.labels))
  
    criterion = nn.NLLLoss(weight=self.class_weights)  # negative log likelihood loss
    #optimizer = torch.optim.Adam(model.parameters(), lr=init_value)
    optimizer = AdamW(model.parameters(), lr=init_value)

    if self.use_cuda:
      model = model.cuda()
      criterion = criterion.cuda()

    self.train_dataloader = self.get_dataloader(self.train_data, batch_size=16, shuffle=True)
    num_steps = len(self.train_dataloader) * num_epochs
    
    curr_lr = init_value
    avg_loss = 0.
    best_loss = 0.
    losses = []
    lrs = []

    #scheduler = LambdaLR(optimizer, lr_lambda=lambda x: x)
    #scheduler = OneCycleLR(optimizer, max_lr=final_value, epochs=num_epochs, steps_per_epoch=len(train_dataloader))

    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,
                                                num_training_steps=num_steps)

    model.train()

    for epoch in range(num_epochs):
    
      print("Epoch: ", epoch)

      for i, (inputs, labels) in enumerate(self.train_dataloader):

        optimizer.zero_grad()
        model.zero_grad()
            
        #inputs = inputs.to(self.device)
            
        labels = labels.to(self.device)
        mask = inputs['attention_mask'].to(self.device)
        input_ids = inputs['input_ids'].squeeze(1).to(self.device)
            
        outputs = model(input_ids, mask)
        loss = criterion(outputs, labels.long())
            
        avg_loss = beta * avg_loss + (1 - beta) * loss.item()
        smoothed_loss = avg_loss / (1 - beta**(i+1))

        if smoothed_loss < best_loss or i==0:
          best_loss = smoothed_loss
       
        loss.backward()

        # Clip the norm of the gradients to 1.0.
        # This is to help prevent the "exploding gradients" problem.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        curr_lr = optimizer.param_groups[0]['lr']
        lrs.append(curr_lr)
        losses.append(smoothed_loss)

        optimizer.step()
        scheduler.step()
        
    return lrs, losses, model


  def get_optimum_lr(self, num_epochs=10):

    # Define the range of learning rates to test
    learning_rates = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]

    self.val_dataloader = self.get_dataloader(self.val_data, batch_size=2)

    # Train the model for each learning rate and evaluate on the validation set
    best_loss = float('inf')
    opt_lr = None
    val_losses = []

    for lr in learning_rates:
      lrs, losses, model = self.find_lr(init_value=lr, num_epochs=num_epochs)
      _, val_loss = self.eval_step(model=model)
      val_losses.append(val_loss) 
      print("LR:", lr, "Val Loss:", val_loss)
      if val_loss < best_loss:
        best_loss = val_loss
        opt_lr = lr

    plt.plot(learning_rates, val_losses)
    plt.xscale('log')
    plt.xlabel('Learning Rate')
    plt.ylabel('Loss')
    plt.show()
    print("Optimal learning rate:", opt_lr)

    return opt_lr


  def format_time(self, elapsed):

    '''
    Takes a time in seconds and returns a string hh:mm:ss:ms
    '''

    # Round to the nearest millisecond.
    elapsed_rounded = round((elapsed) * 1000) / 1000

    # Format as hh:mm:ss.ms
    return str(datetime.timedelta(seconds=elapsed_rounded))[:-3]


  def print_model_parameters(self):

    model = BertClassifier(self.arch, classes=len(self.labels))

    # Get all of the model's parameters as a list of tuples.
    params = list(model.named_parameters())

    print('The BERT model has {:} different named parameters.\n'.format(len(params)))

    print('==== Embedding Layers ====\n')

    for p in params[0:4]:
      print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

    print('\n==== First Transformer ====\n')

    for p in params[4:21]:
      print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

    print('\n==== Output Layers ====\n')

    for p in params[-4:]:
      print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))


  def train_cross_validation(self, n_splits=5):

    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
 
    # Measure the total training time for the whole run.
    total_t0 = time.time()

    self.training_stats = []

    for fold, (train_index, val_index) in enumerate(kfold.split(self.df['text'], self.df['intent'])):
      print(f"\nTraining fold {fold+1}")
      train_data, val_data = self.df.iloc[train_index], self.df.iloc[val_index]
      _, fold_stats = self.train(train_data=train_data, val_data=val_data, fold=fold)

    runtime = self.format_time(time.time()-total_t0)
    avg_fold_results, avg_val_acc, avg_val_loss = self.cross_validation_results(stats=fold_stats)

    self.update_train_log(runtime, fold_stats, self.epochs, avg_val_acc, avg_val_loss, n_splits=n_splits)

    # Print results
    print("\nAverage results for each fold:")
  
    for fold, (fold_avg_val_loss, fold_avg_val_acc) in avg_fold_results.items():
      print(f"Fold {fold}: Avg. val loss = {fold_avg_val_loss:.3f}, Avg. val accuracy = {fold_avg_val_acc:.3f}")
  
    print(f"\nOverall average results: Avg. val loss = {avg_val_loss:.3f}, Avg. val accuracy = {avg_val_acc:.3f}")
    print("Total training took {:} (h:mm:ss)".format(runtime))

    
  def cross_validation_results(self, stats):

    # Initialize variables
    fold_results = defaultdict(list)
    total_val_loss = 0.0
    total_val_accuracy = 0.0

    # Group results by fold
    for result in stats:
      fold = result["fold"]
      val_loss = result["Val Loss"]
      val_accuracy = result["Val Accuracy"]
    
      # Store fold results
      fold_results[fold].append((val_loss, val_accuracy))
    
      # Accumulate total results
      total_val_loss += val_loss
      total_val_accuracy += val_accuracy

    #print(fold_results)

    # Calculate average results for each fold
    avg_fold_results = {}
    for fold, results in fold_results.items():
      num_results = len(results)
      fold_avg_val_loss = sum(val_loss for val_loss, _ in results) / num_results
      fold_avg_val_accuracy = sum(val_accuracy for _, val_accuracy in results) / num_results
      avg_fold_results[fold] = (fold_avg_val_loss, fold_avg_val_accuracy)

    #print(avg_fold_results)

    # Calculate overall average results
    num_results = len(stats)
    avg_val_acc = total_val_accuracy / num_results
    avg_val_loss = total_val_loss / num_results
    
    return avg_fold_results, avg_val_acc, avg_val_loss


  def random_seed(self, seed_value):
    '''
    Sets the random seed for numpy, pytorch, python.random and pytorch GPU vars.
    '''
    np.random.seed(seed_value) # Numpy vars
    torch.manual_seed(seed_value) # PyTorch vars
    random.seed(seed_value) # Python

    if self.use_cuda: # GPU vars
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value) 
        torch.backends.cudnn.deterministic = True 
        torch.backends.cudnn.benchmark = False
    
    print(f'Random state set: {seed_value}, cuda used: {self.use_cuda}')


  def update_train_log(self, runtime, stats, epochs, avg_val_acc, avg_val_loss, model_version=None, model_name=None, n_splits=None):

    today = date.today()

    if n_splits is None:

      logfile = os.path.join(self.LOG_PATH,"train-{}-{}-{}.log".format(today.year, today.month, today.day))

      ## write the data to a csv file    
      header = ['unique_id','timestamp','n_epochs','stats','best_val_acc','best_val_loss','model_version','model_name','runtime']   

    else:

      logfile = os.path.join(self.LOG_PATH,"train-cv-{}-{}-{}.log".format(today.year, today.month, today.day))

      ## write the data to a csv file    
      header = ['unique_id','timestamp','n_folds','n_epochs_fold','stats','avg_val_acc','avg_val_loss','runtime']

    write_header = False

    local_time = time.localtime(time.time())
    time_string = time.strftime('%Y-%m-%d %H:%M:%S', local_time)
    
    if not os.path.exists(logfile):
        write_header = True

    with open(logfile,'a') as csvfile:

        writer = csv.writer(csvfile, delimiter=',')
        
        if write_header:
            writer.writerow(header)

        if n_splits is None:        
          to_write = map(str, [uuid.uuid4(), time_string, epochs, stats, avg_val_acc, avg_val_loss, model_version, model_name, runtime])
        else:
          to_write = map(str, [uuid.uuid4(), time_string, n_splits, epochs, stats, avg_val_acc, avg_val_loss, runtime])

        writer.writerow(to_write)


  def update_predict_log(self, runtime, pred, proba, query, model_version=None, model_name=None):

    today = date.today()

    logfile = os.path.join(self.LOG_PATH,"predict-{}-{}-{}.log".format(today.year, today.month, today.day))

    ## write the data to a csv file    
    header = ['unique_id','timestamp','pred','proba','query','model_version','model_name','runtime']

    write_header = False

    local_time = time.localtime(time.time())
    time_string = time.strftime('%Y-%m-%d %H:%M:%S', local_time)

    if not os.path.exists(logfile):
      write_header = True

    with open(logfile,'a') as csvfile:

      writer = csv.writer(csvfile, delimiter=',')
        
      if write_header:
        writer.writerow(header)

      to_write = map(str, [uuid.uuid4(), time_string, pred, proba, query, model_version, model_name, runtime])

      writer.writerow(to_write)


  def train_graph(self, stats, metric='Loss'):

    '''
    metric = 'Loss' or 'Accuracy'
    '''

    n_epochs = stats[-1]['epoch']

    # Create a DataFrame from our training statistics.
    df_stats = pd.DataFrame(data=stats)

    # Use the 'epoch' as the row index.
    df_stats = df_stats.set_index('epoch')

    df_stats[['Train Loss', 'Train Accuracy', 'Val Loss', 'Val Accuracy']] = df_stats[['Train Loss', 'Train Accuracy', 'Val Loss', 'Val Accuracy']].round(2)

    # Use plot styling from seaborn.
    sns.set(style='darkgrid')

    # Increase the plot size and font size.
    sns.set(font_scale=1.5)
    plt.rcParams["figure.figsize"] = (12,6)

    tick_spacing = 5  # Set the desired spacing between ticks (adjust as needed)

    # Plot the learning curve.
    plt.plot(df_stats['Train ' + metric], 'b-o', label="Training")
    plt.plot(df_stats['Val ' + metric], 'g-o', label="Validation")

    # Label the plot.
    plt.title("Training & Validation " + metric)
    plt.xlabel("Epoch")
    plt.ylabel(metric)
    plt.legend()
    #plt.xticks(list(range(1, n_epochs+1)))
    plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))  # Set tick spacing

    plt.show()