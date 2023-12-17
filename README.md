# comic_dialogue_generation
This is a class project for CMSC 491/691 course.

## Usage
### Install
1. Clone this repository and navigate to mPLUG-Owl directory
```bash
git clone https://github.com/shaswati1/comic_dialogue_generation.git
cd mPLUG-Owl
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
'''The following is a conversation between a curious human and AI assistant.
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
