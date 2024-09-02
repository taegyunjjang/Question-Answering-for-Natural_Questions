from transformers import BertTokenizerFast
from datasets import load_dataset
from tqdm import tqdm
import os
import json
from rank_bm25 import BM25Okapi
import pickle
from bs4 import BeautifulSoup
import logging
logging.disable(logging.WARNING)


pretrained_model_name = 'bert-large-uncased-whole-word-masking-finetuned-squad'
tokenizer = BertTokenizerFast.from_pretrained(pretrained_model_name, clean_up_tokenization_spaces=True)

def save_data(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f)
                
def create_bm25_index(file_name):
    tokenized_documents = []
    document_name = 'data/document/cleaned_train_document.json'
        
    with open(document_name, 'r', encoding='utf-8') as f:
        documents = json.load(f)
        for doc in tqdm(documents, desc="Tokenizing documents for BM25 index"):
            tokenized_documents.append(doc.split())
    
    bm25 = BM25Okapi(tokenized_documents)
    with open(file_name, 'wb') as f:  # 'data/bm25_index.pkl'
        pickle.dump(bm25, f)
        
def is_skip_data(flag, data):
    if flag == 'test':
        if data['annotations'][0]['yes_no_answer'] != "NONE":  # "YES", "NO"
            return True
        elif data['annotations'][0]['short_answers']:
            return True
    else:
        if data['annotations']['yes_no_answer'][0] != -1:
            return True
        elif data['annotations']['short_answers'][0]['start_token']:
            return True
    return False

def has_long_answer(start_token, end_token):
  return (start_token >= 0 and end_token >= 0)

def get_first_annotation(data):
    pass

def get_text(data, start, end):
    text = []
    for i in range(start, end):
        tokens = data["document"]['tokens']
        if not tokens["is_html"][i]:
            token = tokens["token"][i].replace(" ", "")
            text.append(token)
    return " ".join(text)

def token_to_char_offset(data, start, end):
    char_offset = 0
    for i in range(start, end):
        tokens = data["document"]['tokens']
        if not tokens["is_html"][i]:
            token = tokens["token"][i].replace(" ", "")
            char_offset += len(token) + 1
    return char_offset

def clean_html_tokens(dataset, flag, batch_size=10000):
    processed_dataset = []
    processed_documents = []
    batch_count = 0
    
    for i, data in enumerate(tqdm(dataset, desc='Cleaning dataset')):
        if is_skip_data(flag, data):  # short/yes/no answer
            continue
        
        if flag == 'train':
            document_token = data['document']['tokens']['token']
            start_token = data['annotations']['long_answer'][0]['start_token']
            end_token = data['annotations']['long_answer'][0]['end_token']
        
            clean_context = get_text(data, 0, len(document_token))
            clean_answer = get_text(data, start_token, end_token)
            
            start_position, end_position = -1, -1
            if has_long_answer(start_token, end_token):
                start_position = token_to_char_offset(data, 0, start_token)
                end_position = token_to_char_offset(data, 0, end_token) - 1
            
            processed_dataset.append({
                'question': data['question']['text'],
                'answer': clean_answer,
                'context': clean_context,
                'start_position': start_position,
                'end_position': end_position
            })
        
        # if flag != 'validation':
        #     processed_documents.append(clean_answer)
                
        # if (i + 1) % batch_size == 0:
        #     dataset_batch_file_name = f"cleaned_{flag}_data_{batch_count}.json"
        #     if flag != 'validation':
        #         document_batch_file_name = f"cleaned_{flag}_document_{batch_count}.json"
            
        #     save_data(processed_dataset, os.path.join('data/cleaned', dataset_batch_file_name))
        #     if flag != 'validation':
        #         save_data(processed_documents, os.path.join('data/document', document_batch_file_name))
            
        #     batch_count += 1
            
        #     processed_dataset.clear()
        #     processed_documents.clear()
    
    # if processed_dataset:
    #     dataset_batch_file_name = f"cleaned_{flag}_data_{batch_count}.json"
    #     if flag != 'validation':
    #         document_batch_file_name = f"cleaned_{flag}_document_{batch_count}.json"
        
    #     save_data(processed_dataset, os.path.join('data/cleaned', dataset_batch_file_name))
    #     if flag != 'validation':
    #         save_data(processed_documents, os.path.join('data/document', document_batch_file_name))
        
    # if flag != 'validation': 
    #     merge_documents(flag)

def tokenize_data(input):
    return tokenizer(
        input['question'], input['context'],
        truncation='longest_first', padding='max_length',
        max_length=512,
        return_tensors='pt'
    )

def tokenize_dataset(file_name, batch_count):
    tokenized_dataset = []  
    for batch_idx in range(batch_count):
        batch_file_name = f"{file_name}_{batch_idx}.json"  # 'cleaned_train_data_0.json'
        with open(os.path.join('data/cleaned', batch_file_name), 'r', encoding='utf-8') as f:
            cleaned_dataset = json.load(f)
        
        for data in tqdm(cleaned_dataset, desc=f'Tokenizing dataset'):
            t = tokenize_data(data)
            
            token_type_ids = t['token_type_ids'].squeeze().tolist()
            start_positions = token_type_ids.index(1)
            end_positions = len(token_type_ids) - token_type_ids[::-1].index(1) - 2  # 2번째 [SEP] 토큰까지
                    
            tokenized_dataset.append({
                'input_ids': t['input_ids'].squeeze().tolist(),
                'token_type_ids': token_type_ids,
                'attention_mask': t['attention_mask'].squeeze().tolist(),
                'start_positions': start_positions,
                'end_positions': end_positions
            })
                
        if tokenized_dataset:
            dataset_file = f"tokenized_{file_name}_{batch_idx}.json"  # 'tokenized_cleaned_train_data_1.pt"
            with open(os.path.join('data/tokenized', dataset_file), 'w', encoding='utf-8') as f:
                json.dump(tokenized_dataset, f, ensure_ascii=False, indent=4)
            tokenized_dataset.clear()
            
        
def merge_documents(flag):
    document_dir = 'data/document'
    all_documents = []
    
    for file_name in tqdm(sorted(os.listdir(document_dir)), desc="Merging documents"):
        if file_name.startswith(f"cleaned_{flag}_document"):
            with open(os.path.join(document_dir, file_name), 'r', encoding='utf-8') as f:
                documents = json.load(f)
                all_documents.extend(documents)

    save_data(all_documents, os.path.join(document_dir, f"cleaned_{flag}_document.json"))

def main():
    os.makedirs('data', exist_ok=True)
    os.makedirs('data/cleaned', exist_ok=True)
    os.makedirs('data/tokenized', exist_ok=True)
    os.makedirs('data/document', exist_ok=True)
    
    cache_dir = 'D:\\tgkim\\new_cache'
    test_file = 'D:\\tgkim\\new_cache\\v1.0-simplified_nq-dev-all.jsonl'
    
    train_dataset = load_dataset('google-research-datasets/natural_questions', split='train', cache_dir=cache_dir)
    val_dataset = load_dataset('google-research-datasets/natural_questions', split='validation', cache_dir=cache_dir)
    test_dataset = load_dataset('json', data_files=test_file, split='train')
    
    print(f"Train set: {len(train_dataset)}")
    print(f"Validation set: {len(val_dataset)}")
    print(f"Test set: {len(test_dataset)}")

    clean_html_tokens(train_dataset, flag='train')
    clean_html_tokens(val_dataset, flag='validation')
    clean_html_tokens(test_dataset, flag='test')
    
    create_bm25_index(file_name='data/bm25_index.pkl')
    
    train_batch_count = len([f for f in os.listdir('data/cleaned') if f.startswith('cleaned_train_data')])
    val_batch_count = len([f for f in os.listdir('data/cleaned') if f.startswith('cleaned_validation_data')])

    tokenize_dataset('cleaned_train_data', train_batch_count)
    tokenize_dataset('cleaned_validation_data', val_batch_count)


if __name__ == '__main__':
    main()
    