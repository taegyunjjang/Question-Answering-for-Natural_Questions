from transformers import BertTokenizerFast, BertForQuestionAnswering
import torch
import json
import numpy as np
import pickle
from tqdm import tqdm
import logging
logging.disable(logging.WARNING)


def load_model_and_tokenizer(model_path, pretrained_model_name):
    model = BertForQuestionAnswering.from_pretrained(pretrained_model_name)
    tokenizer = BertTokenizerFast.from_pretrained(pretrained_model_name)
    
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint)
    return model, tokenizer

def load_documents(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        documents = json.load(f)
    return documents

def search_top_documents(question, bm25, documents):
    question_tokens = question.split()
    scores = bm25.get_scores(question_tokens)
    
    top_index = np.argmax(scores)
    
    if top_index < len(documents):
        top_document = documents[top_index]
    else:
        top_document = ''
    
    return top_document

def predict_answer(question, context, model, tokenizer):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    inputs = tokenizer(question, context, truncation=True, max_length=512, return_tensors='pt').to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)

    start_scores = outputs.start_logits
    end_scores = outputs.end_logits

    start_index = torch.argmax(start_scores)
    end_index = torch.argmax(end_scores)

    if start_index <= end_index:
        answer = tokenizer.decode(inputs['input_ids'][0][start_index:end_index + 1])
    else:
        answer = 'no answer'

    return answer

def compute_f1(prediction, truth):
    prediction_token = prediction.split()
    truth_token = truth.split()
    
    if len(truth_token) == 0 or prediction == 'no answer':
        return int(len(truth_token) == 0 and prediction == 'no answer')
    
    common_token = set(prediction_token) & set(truth_token)
    
    if len(common_token) == 0:
        return 0
    
    precision = len(common_token) / len(prediction_token)
    recall = len(common_token) / len(truth_token)
    
    return 2 * (precision * recall) / (precision + recall)

def main():
    pretrained_model_name = 'bert-large-uncased-whole-word-masking-finetuned-squad'
    model_path = r'C:\Users\tgkim\Workspace\results\model_checkpoint_step_5.pt'
    model, tokenizer = load_model_and_tokenizer(model_path, pretrained_model_name)
    
    with open('data/bm25_index.pkl', 'rb') as f:
        bm25 = pickle.load(f)
    
    test_dataset = load_documents('data/cleaned/cleaned_test_data_0.json')  # 'question', 'context'
    documents = load_documents('data/document/cleaned_train_document.json')
    
    total_f1_score = 0
    num_questions = len(test_dataset)

    for data in tqdm(test_dataset, desc='Inference'):
        question = data['question']
        gold_answer = data['context']

        top_document = search_top_documents(question, bm25, documents)
        
        answer = predict_answer(question, top_document, model, tokenizer)
            
        f1_score = compute_f1(answer, gold_answer)
        total_f1_score += f1_score
        
        # print(f"Question: {question}")
        # print(f"Answers: {answer}")
        # print(f"F1 score: {f1_score}")
        # print('-'*100)
        
    avg_f1_score = total_f1_score / num_questions
    print(f"Average F1 score: {avg_f1_score}")

    
if __name__ == '__main__':
    main()
