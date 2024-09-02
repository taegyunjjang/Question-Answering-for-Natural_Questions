from transformers import BertForQuestionAnswering, AdamW
import torch
from torch.utils.data import DataLoader, Dataset, ConcatDataset
import os
import json
from tqdm import tqdm
import logging
logging.disable(logging.WARNING)


class CustomDataset(Dataset):
    def __init__(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            'input_ids': torch.tensor(item['input_ids']),
            'token_type_ids': torch.tensor(item['token_type_ids']),
            'attention_mask': torch.tensor(item['attention_mask']),
            'start_positions': torch.tensor(item['start_positions']),
            'end_positions': torch.tensor(item['end_positions'])
        }

def get_data_loader(batch_size, data_type='train'):
    file_names = sorted([os.path.join('data/tokenized', f) for f in os.listdir('data/tokenized') if f.startswith(f'tokenized_cleaned_{data_type}_data')])
    datasets = [CustomDataset(file_name) for file_name in file_names]
    return DataLoader(ConcatDataset(datasets), batch_size=batch_size)
    
def train(model, train_data_loader, val_data_loader, optimizer, device, save_interval=60000):
    model.train()
    total_loss = 0
    processed_samples = 0
    
    for batch in tqdm(train_data_loader, desc="Training"):
        optimizer.zero_grad()

        input_ids = batch['input_ids'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        start_positions = batch['start_positions'].to(device)
        end_positions = batch['end_positions'].to(device)

        outputs = model(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            start_positions=start_positions,
            end_positions=end_positions
        )

        loss = outputs.loss
        total_loss += loss.item()
        processed_samples += input_ids.size(0)

        loss.backward()
        optimizer.step()
        
        if processed_samples % save_interval == 0:
            checkpoint_path = os.path.join('results', f"model_checkpoint_step_{processed_samples // save_interval}.pt")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved checkpoint at {checkpoint_path}")
            
            avg_train_loss = total_loss / len(train_data_loader)
            avg_val_loss = validate(model, val_data_loader, device)
            print(f"Average training loss: {avg_train_loss}")
            print(f"Average validation loss: {avg_val_loss}")

def validate(model, data_loader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Validation"):
            input_ids = batch['input_ids'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            start_positions = batch['start_positions'].to(device)
            end_positions = batch['end_positions'].to(device)

            outputs = model(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
                start_positions=start_positions,
                end_positions=end_positions
            )

            loss = outputs.loss
            total_loss += loss.item()

    return total_loss / len(data_loader)

def main():
    epochs = 3
    lr = 5e-5
    batch_size = 32
    save_interval = 60000
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    pretrained_model_name = 'bert-large-uncased-whole-word-masking-finetuned-squad'
    model = BertForQuestionAnswering.from_pretrained(pretrained_model_name).to(device)
    optimizer = AdamW(model.parameters(), lr=lr)
            
    os.makedirs('results', exist_ok=True)
    
    train_data_loader = get_data_loader(batch_size=batch_size, data_type='train')
    val_data_loader = get_data_loader(batch_size=batch_size, data_type='validation')

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")

        train(model, train_data_loader, val_data_loader, optimizer, device, save_interval=save_interval)

        checkpoint_path = os.path.join('results', f"model_checkpoint_epoch_{epoch + 1}")
        os.makedirs(checkpoint_path, exist_ok=True)
        model.save_pretrained(checkpoint_path)
        print(f"Saved model and tokenizer at {checkpoint_path}")


if __name__ == '__main__':
    main()
