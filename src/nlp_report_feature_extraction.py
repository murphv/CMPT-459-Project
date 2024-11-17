import nltk
import os
import pandas as pd
from tqdm import tqdm
import re
from transformers import BertTokenizer, BertModel
import torch
import numpy as np


nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
# nltk.download('punkt_tab')


def load_filings(filing_dir):
    """
    Loads SEC filings from the specified directory and aggregates them per company per quarter.
    
    Parameters:
        sec_filing_dir (str): Path to the SEC_Filings directory.
        
    Returns:
        pd.DataFrame: DataFrame containing Company, Year, Quarter, and Report Text.
    """
    data = []
    companies = os.listdir(filing_dir)

    # Iterating through each company in SEC_Filings folder
    for company in tqdm(companies, desc="Loading Company Filings"):
        comp_dir = os.path.join(filing_dir, company)
        
        if os.path.isdir(comp_dir):         # Check if directory is actually correct
            files = os.listdir(comp_dir)

            for file in files:              # iterating for all the files in the company dir
                if file.endswith('.txt'):
                    name_parts = file.replace('.txt', '').split('_')

                    try:    # Attempting to extract the relative information from the file name
                        ticker, rep_type, year, quarter = name_parts
                        
                        with open(os.path.join(comp_dir, file), 'r', encoding='utf-8') as f:
                            text = f.read()
                        
                        data.append({
                            'Company': ticker,
                            'Year': year,
                            'Quarter': quarter,
                            'Report_Text': text
                        })

                    except Exception as e:
                        print(f"Error processing file {file}: {e}")


    return pd.DataFrame(data)


def process_text(text):
    """
    Preprocesses the text for BERT by lowercasing and removing HTML tags.
    Does not remove stop words or perform lemmatization.
    """
    text = text.lower()
    # Remove HTML tags
    text = re.sub(r'<.*?>', ' ', text)
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def get_bert_embeddings(text_list, tokenizer, model, max_length=512, batch_size=16):
    """
    Generates BERT embeddings for a list of texts.
    
    Parameters:
        text_list (list): List of preprocessed text strings.
        tokenizer: BERT tokenizer.
        model: BERT model.
        max_length (int): Maximum token length for BERT (default is 512).
        batch_size (int): Number of samples per batch.
        
    Returns:
        np.ndarray: Array of BERT embeddings.
    """
    embeddings = []
    model.eval()
    
    with torch.no_grad():
        for i in tqdm(range(0, len(text_list), batch_size), desc="Generating BERT Embeddings"):
            batch_texts = text_list[i:i+batch_size]
            encoded_inputs = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors='pt'
            )
            input_ids = encoded_inputs['input_ids']
            attention_mask = encoded_inputs['attention_mask']
            
            outputs = model(input_ids, attention_mask=attention_mask)
            # Use the [CLS] token representation for each document
            cls_embeddings = outputs.last_hidden_state[:, 0, :].numpy()
            embeddings.extend(cls_embeddings)
    
    return np.array(embeddings)


def main():
    sec_filing_path = 'data/SEC_Filings/'

    # Load pre-trained BERT tokenizer and model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = BertModel.from_pretrained('bert-base-uncased')

    df = load_filings(sec_filing_path)
    
    df['Report_Text'] = df['Report_Text'].apply(process_text)

    bert_embeddings = get_bert_embeddings(df['Report_Text'].tolist(), tokenizer, bert_model)    
    print(f"BERT embeddings shape: {bert_embeddings.shape}")

    bert_feature_names = [f'BERT_{i}' for i in range(bert_embeddings.shape[1])]
    
    nlp_df = pd.DataFrame(bert_embeddings, columns=bert_feature_names)
    
    nlp_df['Company'] = df['Company'].values
    nlp_df['Year'] = df['Year'].values
    nlp_df['Quarter'] = df['Quarter'].values

    nlp_df.to_csv('data/nlp_features.csv', index=False)



if __name__ == '__main__':
    main()