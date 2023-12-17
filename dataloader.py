import torch
from torch.utils.data import Dataset, DataLoader
from mplug_owl_video.tokenization_mplug_owl import MplugOwlTokenizer
from mplug_owl_video.modeling_mplug_owl import MplugOwlForConditionalGeneration
from mplug_owl_video.processing_mplug_owl import MplugOwlImageProcessor, MplugOwlProcessor
import json
from PIL import Image


#For each sample we fixed the number of panels to be 6 and generate the utterance for 7th panel
#The input should be a jsonl file with the following format:
#{"video": ["path/to/image1", "path/to/image2", ...], "text": "transcript", "label": "response"}
#Here "transcript" is the utterance for the first 6 panels and "response" is the utterance for the 7th panel


def load_jsonl(filename):
    with open(filename, "r", encoding="utf-8") as f:
        return [json.loads(l.strip("\n")) for l in f.readlines()]

class ComicDataset(Dataset):
    def __init__(self, input_files, tokenizer, processors, max_length=2048):
        self.data = load_jsonl(input_files)
        self.tokenizer = tokenizer
        self.processor = processors

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        transcript = [item['text']] #utterances up to the t th panel
        response = [item['label']]  #utterance for the (t+1) th panel
        prompts = [f'''The following is a conversation between a curious human and AI assistant.
                        Human: Here is a video and a transcript {transcript}.
                        Human: <|video|>
                        Human: Generate utterance for the last image of this video.
                        AI: {response}'''
                ]
        image_list = item['video'] #list of images up to the (t+1) th panel
        inputs = self.processor(text=prompts, videos=image_list, num_frames=len(image_list), return_tensors='pt')
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        video_pixel_values = inputs["video_pixel_values"]
        input_ids = inputs['input_ids']
        num_images_tensor = torch.tensor([len(image_list)], dtype=torch.long, device=self.device)
        num_videos_tensor = torch.tensor(1, dtype=torch.long, device=self.device)
        attention_mask = inputs['attention_mask']
        
        dtype=torch.bfloat16
        non_padding_mask = (response != self.tokenizer.pad_token_id).to(dtype)[:,:-1]
        non_media_mask = torch.ones_like(non_padding_mask).to(dtype)
        prompt_mask = torch.zeros_like(non_padding_mask).to(dtype)
        

        inputs = {
            "video_pixel_values": video_pixel_values,
            "input_ids": input_ids,
            "num_images": num_images_tensor,
            "num_videos": num_videos_tensor,
            "non_padding_mask": non_padding_mask,
            "non_media_mask": non_media_mask,
            "prompt_mask": prompt_mask,
            "attention_mask": attention_mask,
            "labels": response
        }
        return inputs
