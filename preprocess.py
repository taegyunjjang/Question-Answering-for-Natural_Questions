from transformers import BertTokenizerFast
from datasets import load_dataset
from tqdm import tqdm
import os
import json
from rank_bm25 import BM25Okapi
import pickle
import collections
import logging
logging.disable(logging.WARNING)


MAX_QUERY_LENGTH = 64
MAX_SEQ_LENGTH = 512
DOC_STRIDE = 128

TextSpan = collections.namedtuple("TextSpan", "token_positions text")

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
        if data['annotations'][0]['yes_no_answer'] != "NONE":
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

def get_text_span(data, start, end, flag='train'):
    token_positions = []
    text = []
    if flag == 'train':
        for i in range(start, end):
            tokens = data["document"]['tokens']
            if not tokens["is_html"][i]:
                token_positions.append(i)
                token = tokens["token"][i].replace(" ", "")
                text.append(token)
        return TextSpan(token_positions, " ".join(text))
    elif flag == 'test':
        for i in range(start, end):
            tokens = data["document_tokens"]
            token = tokens[i]["token"].replace(" ", "")
            text.append(token)
        return TextSpan(token_positions, " ".join(text))

def token_to_char_offset(data, start, end):
    char_offset = 0
    for i in range(start, end):
        tokens = data["document"]['tokens']
        if not tokens["is_html"][i]:
            token = tokens["token"][i].replace(" ", "")
            char_offset += len(token) + 1
    return char_offset

def char_to_word_offset(contexts, offset):
    offset = []
    doc_tokens = []
    prev_is_whitespace = True
    
    def is_whitespace(c):
        return c in " \t\r\n" or ord(c) == 0x202F

    for i, c in enumerate(contexts):
        if is_whitespace(c):
            prev_is_whitespace = True
        else:
            if prev_is_whitespace:
                doc_tokens.append(c)
            else:
                doc_tokens[-1] += c
            prev_is_whitespace = False
        offset.append(len(doc_tokens) - 1)
    
    return offset
    
def merge_documents(flag):
    document_dir = 'data/document'
    all_documents = []
    for file_name in tqdm(sorted(os.listdir(document_dir)), desc="Merging documents"):
        if file_name.startswith(f"cleaned_{flag}_document"):
            with open(os.path.join(document_dir, file_name), 'r', encoding='utf-8') as f:
                documents = json.load(f)
                all_documents.extend(documents)

    save_data(all_documents, os.path.join(document_dir, f"cleaned_{flag}_document.json"))


def clean_html_tokens(dataset, flag, batch_size=10000):
    processed_datasets = []
    processed_documents = []
    batch_count = 0
    
    for i, data in enumerate(tqdm(dataset, desc='Cleaning dataset')):
        if is_skip_data(flag, data):  # short/yes/no answer
            continue
        
        if flag == 'train' or flag == 'validation':
            question = data['question']['text']
            document_token = data['document']['tokens']['token']
            start_token = data['annotations']['long_answer'][0]['start_token']
            end_token = data['annotations']['long_answer'][0]['end_token']
        
            clean_context = get_text_span(data, 0, len(document_token)).text
            clean_answer = get_text_span(data, start_token, end_token).text
            
            doc_tokens_length = len(document_token)
            doc_tokens = [document_token[i] for i in range(doc_tokens_length) if not data["document"]["tokens"]["is_html"][i]]

            start_position, end_position = -1, -1
            if has_long_answer(start_token, end_token):
                answer_start_offset = token_to_char_offset(data, 0, start_token)
                char_to_word_offset_list = char_to_word_offset(clean_context, answer_start_offset)
                start_position = char_to_word_offset_list[answer_start_offset]
                end_position = char_to_word_offset_list[answer_start_offset + len(clean_answer) - 1]
            
            processed_datasets.append({  # html 태그 제거 후 데이터 저장
                'question': question,
                'answer': clean_answer,
                'document': clean_context,
                'doc_tokens': doc_tokens,
                'start_position': start_position,  # 토큰 단위 위치
                'end_position': end_position
            })
            
            if flag == 'train':
                processed_documents.append(clean_context)
        
        elif flag == 'test':  # question, clean_answer만 저장
            question = data['question_text']
            start_token = data['annotations'][0]['long_answer']['start_token']
            end_token = data['annotations'][0]['long_answer']['end_token']
            clean_answer = get_text_span(data, start_token, end_token, flag='test').text
            
            processed_datasets.append({
                'question': question,
                'answer': clean_answer
            })
            
                
        if (i + 1) % batch_size == 0:
            if flag == 'train':
                document_batch_file_name = f"cleaned_{flag}_document_{batch_count}.json"
                save_data(processed_documents, os.path.join('data/document', document_batch_file_name))
            
            dataset_batch_file_name = f"cleaned_{flag}_data_{batch_count}.json"
            save_data(processed_datasets, os.path.join('data/cleaned', dataset_batch_file_name))
            
            batch_count += 1
            
            processed_datasets.clear()
            processed_documents.clear()
    
    if processed_datasets:
        if flag == 'train':
            document_batch_file_name = f"cleaned_{flag}_document_{batch_count}.json"
            save_data(processed_documents, os.path.join('data/document', document_batch_file_name))
        
        dataset_batch_file_name = f"cleaned_{flag}_data_{batch_count}.json"
        save_data(processed_datasets, os.path.join('data/cleaned', dataset_batch_file_name))

    if flag == 'train': 
        merge_documents(flag)


def convert_dataset(file_name, batch_count):
    for batch_idx in range(batch_count):
        batch_file_name = f"{file_name}_{batch_idx}.json"  # 'cleaned_train_data_0.json'
        with open(os.path.join('data/cleaned', batch_file_name), 'r', encoding='utf-8') as f:
            cleaned_dataset = json.load(f)
        
        features = []
        for data in tqdm(cleaned_dataset, desc=f'Tokenizing dataset'):
            tok_to_orig_index = []
            orig_to_tok_index = []
            all_doc_tokens = []
            for (i, token) in enumerate(data['doc_tokens']):
                orig_to_tok_index.append(len(all_doc_tokens))
                sub_tokens = tokenizer.tokenize(token)
                tok_to_orig_index.extend([i] * len(sub_tokens))
                all_doc_tokens.extend(sub_tokens)
        
            # QUERY
            query_tokens = tokenizer.tokenize(data['question'])
            if len(query_tokens) > MAX_QUERY_LENGTH:
                query_tokens = query_tokens[-MAX_QUERY_LENGTH:]
            
            # ANSWER
            tok_start_position = 0
            tok_end_position = 0
            
            tok_start_position = orig_to_tok_index[data['start_position']]
            if data['end_position'] < len(data['doc_tokens']) - 1:
                tok_end_position = orig_to_tok_index[data['end_position'] + 1] - 1
            else:
                tok_end_position = len(all_doc_tokens) - 1
            
            max_tokens_for_doc = MAX_SEQ_LENGTH - len(query_tokens) - 3
            
            _DocSpan = collections.namedtuple("DocSpan", ["start", "length"])
            doc_spans = []
            start_offset = 0
            while start_offset < len(all_doc_tokens):
                length = min(len(all_doc_tokens) - start_offset, max_tokens_for_doc)
                doc_spans.append(_DocSpan(start=start_offset, length=length))
                if start_offset + length == len(all_doc_tokens):
                    break
                start_offset += min(length, DOC_STRIDE)
            
            for doc_span in doc_spans:
                tokens = []
                token_type_ids = []
                tokens.append("[CLS]")
                token_type_ids.append(0)
                tokens.extend(query_tokens)
                token_type_ids.extend([0] * len(query_tokens))
                tokens.append("[SEP]")
                token_type_ids.append(0)
                
                for i in range(doc_span.length):
                    split_token_index = doc_span.start + i
                    tokens.append(all_doc_tokens[split_token_index])
                    token_type_ids.append(1)
                tokens.append("[SEP]")
                token_type_ids.append(1)
                
                input_ids = tokenizer.convert_tokens_to_ids(tokens)
                input_mask = [1] * len(input_ids)
                padding = [0] * (MAX_SEQ_LENGTH - len(input_ids))
                input_ids.extend(padding)
                input_mask.extend(padding)
                token_type_ids.extend(padding)
                
                start_position = 0
                end_position = 0
                
                doc_start = doc_span.start
                doc_end = doc_start + doc_span.length - 1
                
                contains_an_annotation = (
                    tok_start_position >= doc_start and tok_end_position <= doc_end
                )
                
                if contains_an_annotation:
                    doc_offset = len(query_tokens) + 2
                    start_position = tok_start_position - doc_start + doc_offset
                    end_position = tok_end_position - doc_start + doc_offset
                    features.append({
                        'input_ids': input_ids,
                        'input_mask': input_mask,
                        'token_type_ids': token_type_ids,
                        'start_position': start_position,
                        'end_position': end_position
                    })
                    break
                else:
                    continue
                
        if features:
            dataset_file = f"tokenized_{file_name}_{batch_idx}.json"  # 'tokenized_cleaned_train_data_1.pt"
            with open(os.path.join('data/tokenized', dataset_file), 'w', encoding='utf-8') as f:
                json.dump(features, f, ensure_ascii=False, indent=4)
            features.clear()
            

def main():
    # 데이터 저장 폴더 생성
    os.makedirs('data', exist_ok=True)
    os.makedirs('data/cleaned', exist_ok=True)
    os.makedirs('data/tokenized', exist_ok=True)
    os.makedirs('data/document', exist_ok=True)
    
    # 데이터 로드
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

    convert_dataset('cleaned_train_data', train_batch_count)
    convert_dataset('cleaned_validation_data', val_batch_count)


if __name__ == '__main__':
    main()
    