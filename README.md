# Comic Dialogue Generation

## Usage
### Install
1. Clone this repository and navigate to mPLUG-Owl directory
```bash
git clone https://github.com/shaswati1/comic_dialogue_generation.git
cd mplug_owl
```

2. Setup environment
```Shell
conda env create -f mplug_owl_env.yml
conda activate mplug_owl
```

### Run Model with provided interface
#### Model initialization
Build model, toknizer and processor.
```Python
from pipeline.interface import get_model
model, tokenizer, processor = get_model(pretrained_ckpt='your checkpoint directory', use_bf16='use bf16 or not')
```

#### Model inference
Prepare model inputs.
```Python
# We use a human/AI template to organize the context.
# <|video|> denotes sequence of images placehold.
prompts = [
'''The following is a conversation between a curious human and AI assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.
                    Human: Here is a video and a transcript {transcript}.
                    Human: <|video|>
                    Human: Generate utterance for the last image of this video.
                    AI: '''
]

# The image paths should be placed in the image_list and kept in the same order as in the prompts.
# We support local file paths. You can customize the pre-processing of images by modifying the mplug_owl.modeling_mplug_owl.ImageProcessor
image_list = ['/image1.jpg', '/image2.jpg', '/image3.jpg', '/image4.jpg']
```


Get response.
```Python
# generate kwargs (the same in transformers) can be passed in the do_generate()
from pipeline.interface import do_generate
sentence = do_generate(prompts, image_list, model, tokenizer, processor, 
                       use_bf16=True, max_length=512, top_k=5, do_sample=True)
```

### Instruction Tuning
The training samples should be stored in a ```name_of_your_file.jsonl``` and orgnized in the following format:
```json
{"video": ["/image1.jpg","/image2.jpg","/image3.jpg","/image4.jpg","/image5.jpg","/image6.jpg","/image7.jpg"],"text": "The following is a conversation between a curious human and AI assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\nHuman: <video>\nHuman: Generate utterance for the last image of this video. Image1: when i wound this watch ii just took off into space now it is running down and im landing ! \nImage2: this is the craziest watch in the world ! i gotta show it to the mob ! \nImage3: weight less ! hey , fellers , i was looking for you want to show you something important !\nImage4: so you know how to wind the watch so what ? he is wasting our time ! lets blow !\nImage5: wait a minute fellers ! you aint seen it yet ---\nImage6: hey let go of me ! who do you think you are a helicopter or something ? all of a sudden i ' m getting a very terrific idea !\nAI: i just touched him and he started floating with me if i touched an armored car full of money , it might float too ! this is one experiment i gotta try !.","label": "i just touched him and he started floating with me if i touched an armored car full of money , it might float too ! this is one experiment i gotta try !","task_type": "gpt4instruct_sft"}

```
The ```task_type``` can be in one of ```{'quora_chat_sft', 'sharegpt_chat_sft', 'llava_sft', 'gpt4instruct_sft'}```.

Prepare your own train.jsonl and dev.jsonl and modify ```data_files``` in ```configs/v0.yaml```.

Execute the training script.
```
PYTHONPATH=./ bash train_it.sh # If you want to finetune LLM, replace it with train_it_wo_lora.sh
```
