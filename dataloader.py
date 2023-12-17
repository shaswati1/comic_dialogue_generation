import torch
from torch.utils.data import Dataset, DataLoader
from mplug_owl_video.tokenization_mplug_owl import MplugOwlTokenizer
from mplug_owl_video.modeling_mplug_owl import MplugOwlForConditionalGeneration
from mplug_owl_video.processing_mplug_owl import MplugOwlImageProcessor, MplugOwlProcessor
import json
from PIL import Image


def load_jsonl(filename):
    with open(filename, "r", encoding="utf-8") as f:
        return [json.loads(l.strip("\n")) for l in f.readlines()]

class ComicDataset(Dataset):
    def __init__(self, jsonl_file, pretrained_ckpt, device=device):
        self.data = load_jsonl(jsonl_file)
        self.image_processor = MplugOwlImageProcessor.from_pretrained(pretrained_ckpt)
        self.tokenizer = MplugOwlTokenizer.from_pretrained(pretrained_ckpt)
        self.processor = MplugOwlProcessor(self.image_processor, self.tokenizer)
        self.device = device

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        transcript = [item['text']] #utterances up to the t th image
        response = [item['label']]  #utterance for the (t+1) th image
        prompts = [f'''The following is a conversation between a curious human and AI assistant.
                        Human: Here is a video and a transcript {transcript}.
                        Human: <|video|>
                        Human: Generate utterance for the last image of this video.
                        AI: {response}'''
                ]
        image_list = item['video']
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
