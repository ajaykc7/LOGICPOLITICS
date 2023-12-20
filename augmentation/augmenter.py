import pandas as pd

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    AutoTokenizer,
    GenerationConfig
)

from helpers import make_prompts, emotion, dilemma, relevancy, intention

import time
import re

FALLACY_HASH = ["emotion", "dilemma", "relevancy", "intention"]
BATCH_SIZE = 5

def inference(dataset, tokenizer, model, gen_config):
    """
    Batch inference on dataset of fallacy prompts

    Parameters
    ----------
    dataset:
        list of prompts of a specific fallacy type. Each entry is a
        dictionary: {"source": <the source sentence>, "prompt": <the constructed prompt for the source sentence>}
    
    Returns
    -------
    augmented_results:
        list of augmented premises. Each entry is a dictionary:
        {"source": <the source sentence>, "premise": <augmented premises for the source sentence>}
    """
    augmented_results = []
    for i in range(0, len(dataset), BATCH_SIZE):
        start = time.time()
        end_id = min(i+BATCH_SIZE, len(dataset)-1)
        
        prompts = [d["prompt"] for d in dataset[i:end_id]]
        sources = [d["source"] for d in dataset[i:end_id]]
        try:
            model_inputs = tokenizer(prompts, return_tensors="pt", padding=True).to("cuda")
            # with torch.no_grad():
            output = model.generate(**model_inputs,
                                    max_new_tokens=200,
                                    generation_config=gen_config)
            # TODO: perplexity from `output.scores`
            # scores = output.scores  # tuple of up to `max_new_tokens` tensors each of shape(batch_size, vocab_size)
            
            generated_texts = tokenizer.batch_decode(output.sequences)

            augmented_results += [{"source": source, "premise": premise} for source, premise in zip(sources, generated_texts)]
        except:
            pass
        print(f"dataset[{i}:{end_id}] after {time.time() - start} seconds")
    print("Augmentation done!\n\n\n")
    return augmented_results

def extract_premises(augmented):
    """
    Parameters
    ----------
    augmented: list(dict())
        list of {"source": <src sentence>, "premise": <augmented premise>}
    
    Returns
    -------
    premises:
        augmented premises that's enclosed by " " in generated outputs
    premise_sources:
        source sentences of the premises
    errors:
        augmented premises that don't follow format due to uncontrollable generated
    error_sources:
        source sentences of the error premises
    """
    premises = []
    is_error = []
    sources = []
    for aug in augmented:
        try:
            s = aug["premise"].replace("\n", " ")
            g = re.findall(r'\[/INST].*', s)
            premises.append(g[0].split('"')[-2])
            is_error.append("-")
        except:
            g = g[0].replace("[/INST]", "")
            g = g.replace("<unk>", "")
            g = g.replace("</s>", "")
            premises.append(g)
            is_error.append("+")
        sources.append(aug["source"])

    assert len(premises) == len(sources) == len(is_error)
    return premises, sources, is_error

if __name__ == "__main__":
    # Pull extracted sentences from keywords
    sentences = []

    df = pd.read_csv("../Data/filtered_keyword_sentences2.csv")
    sentences += df["Extracted Sentence"].to_list()

    sentences = set(sentences)

    fallacy_prompts = [emotion, dilemma, relevancy, intention]

    # prompt_datasets: num_fallacy_type x num_sentences
    prompt_datasets = [make_prompts(sentences, f) for f in fallacy_prompts]

    #Quantization configuration
    compute_dtype = getattr(torch, "float16")
    bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.pad_token_id =  tokenizer.unk_token_id
    tokenizer.padding_side = 'left'

    model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1",
                                                quantization_config=bnb_config,
                                                load_in_4bit=True,
                                                device_map="auto")
    #Configure the pad token in the model
    model.config.pad_token_id = tokenizer.pad_token_id

    gen_config = GenerationConfig(
        pad_token_id=tokenizer.pad_token_id,
        temperature=0.7,
        top_p=1.0,
        top_k=50,
        do_sample=True,
        num_beams=1,
        output_scores=True,
        return_dict_in_generate=True
    )
    
    # Test
    # augmented_premises = [inference(dataset) for dataset in [prompt_datasets[0][:3], prompt_datasets[1][:3], prompt_datasets[2][:3], prompt_datasets[3][:3]]]
    # Augment each fallacy-specific prompt dataset
    augmented_premises = [inference(dataset, tokenizer, model, gen_config) for dataset in prompt_datasets] # num_fallacy_type x num_sentences

    # Create dataframe
    premises = []
    sources = []
    is_errors = []

    for augmented in augmented_premises: # each loop corresponds to a fallacy type
        premise, source, is_error = extract_premises(augmented)
        premises.append(premise)
        sources = source
        is_errors.append(is_error)

    data = {"source": source}
    for t in range(len(premises)):
        data[f"p_{FALLACY_HASH[t]}"] = premises[t]
        data[f"p_{FALLACY_HASH[t]}_is_error"] = is_errors[t]
        
    df = pd.DataFrame(data)

    df.to_csv("/kaggle/working/premises.csv")