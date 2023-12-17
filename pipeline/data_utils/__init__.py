from .processors.builder import build_processors
from .xgpt3_dataset import MultiModalDataset
from mplug_owl_video.processing_mplug_owl import MplugOwlImageProcessor, MplugOwlProcessor
import torch
import json

def load_jsonl(filename):
    with open(filename, "r", encoding="utf-8") as f:
        return [json.loads(l.strip("\n")) for l in f.readlines()]

def train_valid_test_datasets_provider(data_path, config, tokenizer, seq_length=1024):
    """Build train and valid datasets."""
    print('> building train and validation datasets for mPLUG-Owl ...')
    train_ds, valid_ds = build_train_valid_test_datasets(
        input_file=data_path,  
        tokenizer=tokenizer,
        max_length=seq_length, 
        config=config)  
    print("> finished creating mPLUG-Owl datasets ...")
    print(f'train_valid_test_datasets_provider: train_ds: {train_ds}, valid_ds: {valid_ds}')
    return train_ds, valid_ds

def build_train_valid_test_datasets(input_file, tokenizer, max_length=80, config=None):
    #train_processors = build_processors(config['train_processors'])
    valid_processors = build_processors(config['valid_processors'])
    pretrained_ckpt = 'MAGAer13/mplug-owl-llama-7b-video'
    image_processor = MplugOwlImageProcessor.from_pretrained(pretrained_ckpt)
    processor = MplugOwlProcessor(image_processor, tokenizer)
    print(input_file[0])
    data = load_jsonl(input_file[0])
    print(f'data: {data}')
    prompts = data[0]['text']
    image_list = data[0]['image']
    print(f'prompts: {prompts}, image_list: {image_list}')
    inputs = processor(text=prompts, videos=image_list, num_frames=4, return_tensors='pt')
    inputs = {k: v.bfloat16() if v.dtype == torch.float else v for k, v in inputs.items()}
    #inputs = {k: v.to(model.device) for k, v in inputs.items()}
    print(f'inputs: {inputs}')
    assert len(input_file) == 2 # If you have files more than 2, modify code at here or merger them into train and dev
    #print(input_file[0])
    #train_ds = MultiModalDataset(input_file[0], tokenizer, processor, max_length)
    valid_ds = MultiModalDataset(input_file[1], tokenizer, valid_processors, max_length)
    test_ds = None
    return (inputs, valid_ds)
