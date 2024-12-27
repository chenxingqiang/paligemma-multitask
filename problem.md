
1.标有7-2的代码是单独微调目标检测的代码。

2.标有multimodal是单独多模态的代码。

3.标有multitask的是联合微调的代码。

①主要问题，联合微调代码中的两个部分分别损失函数的定义，我在联合微调的代码中关于目标检测的损失先使用了均方误差，我很奇怪，也可能是我没看懂，为什么官方给的单独目标检测的代码和单独多模态的代码里面的损失函数一样呢？既然和多模态的损失函数一样，他是怎么就能把目标检测的类别和bounding box给出来呢？我查看了一下单独目标检测代码部分的process部分，它在模型的低层代码中可以处理类别和bounding box。那就是不是说我在写联合微调的代码的时候可以将目标检测和多模态部分写成相同的损失函数去分配不同的权重？

②paligemm单独的目标检测中的损失函数是不能计算置信度的，能改不？其实这问题也无妨，没有置信度也没有关系，就现在先把两个任务联合微调的代码先解决就好。


PaliGemma – Google's Cutting-Edge Open Vision Language Model
Published May 14, 2024
Merve Noyan's avatar
merve
Merve Noyan
Andreas P. Steiner's avatar
andsteing
Andreas P. Steiner
Google's avatar
google
Pedro Cuenca's avatar
pcuenq
Pedro Cuenca
This article is also available in Chinese 简体中文.


Updated on 23-05-2024: We have introduced a few changes to the transformers PaliGemma implementation around fine-tuning, which you can find in this notebook.
PaliGemma is a new family of vision language models from Google. PaliGemma can take in an image and a text and output text.

The team at Google has released three types of models: the pretrained (pt) models, the mix models, and the fine-tuned (ft) models, each with different resolutions and available in multiple precisions for convenience.

All models are released in the Hugging Face Hub model repositories with their model cards and licenses and have transformers integration.

What is PaliGemma?
PaliGemma (Github) is a family of vision-language models with an architecture consisting of SigLIP-So400m as the image encoder and Gemma-2B as text decoder. SigLIP is a state-of-the-art model that can understand both images and text. Like CLIP, it consists of an image and text encoder trained jointly. Similar to PaLI-3, the combined PaliGemma model is pre-trained on image-text data and can then easily be fine-tuned on downstream tasks, such as captioning, or referring segmentation. Gemma is a decoder-only model for text generation. Combining the image encoder of SigLIP with Gemma using a linear adapter makes PaliGemma a powerful vision language model.

Architecture

The PaliGemma release comes with three types of models:

PT checkpoints: Pretrained models that can be fine-tuned to downstream tasks.
Mix checkpoints: PT models fine-tuned to a mixture of tasks. They are suitable for general-purpose inference with free-text prompts, and can be used for research purposes only.
FT checkpoints: A set of fine-tuned models, each one specialized on a different academic benchmark. They are available in various resolutions and are intended for research purposes only.
The models come in three different resolutions (224x224, 448x448, 896x896) and three different precisions (bfloat16, float16, and float32). Each repository contains the checkpoints for a given resolution and task, with three revisions for each of the available precisions. The main branch of each repository contains float32 checkpoints, where as the bfloat16 and float16 revisions contain the corresponding precisions. There are separate repositories for models compatible with 🤗 transformers, and with the original JAX implementation.

As explained in detail further down, the high-resolution models require a lot more memory to run, because the input sequences are much longer. They may help with fine-grained tasks such as OCR, but the quality increase is small for most tasks. The 224 versions are perfectly fine for most purposes.

You can find all the models and Spaces in this collection.

Model Capabilities
PaliGemma is a single-turn vision language model not meant for conversational use, and it works best when fine-tuning to a specific use case.

You can configure which task the model will solve by conditioning it with task prefixes, such as “detect” or “segment”. The pretrained models were trained in this fashion to imbue them with a rich set of capabilities (question answering, captioning, segmentation, etc.). However, they are not designed to be used directly, but to be transferred (by fine-tuning) to specific tasks using a similar prompt structure. For interactive testing, you can use the "mix" family of models, which have been fine-tuned on a mixture of tasks.

The examples below use the mix checkpoints to demonstrate some of the capabilities.

Image Captioning
PaliGemma can caption images when prompted to. You can try various captioning prompts with the mix checkpoints to see how they respond.

Captioning

Visual Question Answering
PaliGemma can answer questions about an image, simply pass your question along with the image to do so.

VQA

Detection
PaliGemma can detect entities in an image using the detect [entity] prompt. It will output the location for the bounding box coordinates in the form of special <loc[value]> tokens, where value is a number that represents a normalized coordinate. Each detection is represented by four location coordinates in the order y_min, x_min, y_max, x_max, followed by the label that was detected in that box. To convert values to coordinates, you first need to divide the numbers by 1024, then multiply y by the image height and x by its width. This will give you the coordinates of the bounding boxes, relative to the original image size.

Detection

Referring Expression Segmentation
PaliGemma mix checkpoints can also segment entities in an image when given the segment [entity] prompt. This is called referring expression segmentation, because we refer to the entities of interest using natural language descriptions. The output is a sequence of location and segmentation tokens. The location tokens represent a bounding box as described above. The segmentation tokens can be further processed to generate segmentation masks.

Segmentation

Document Understanding
PaliGemma mix checkpoints have great document understanding and reasoning capabilities.

ocrqa

Mix Benchmarks
Below you can find the scores for mix checkpoints.

Model	MMVP Accuracy	POPE Accuracy (random/popular/adversarial)
mix-224	46.00	88.00 86.63 85.67
mix-448	45.33	89.37 88.40 87.47
Fine-tuned Checkpoints
In addition to the pretrained and mix models, Google has released models already transferred to various tasks. They correspond to academic benchmarks that can be used by the research community to compare how they perform. Below, you can find a selected few. These models also come in different resolutions. You can check out the model card of any model for all metrics.

Model Name	Dataset/Task	Score in Transferred Task
paligemma-3b-ft-vqav2-448	Diagram Understanding	85.64 Accuracy on VQAV2
paligemma-3b-ft-cococap-448	COCO Captions	144.6 CIDEr
paligemma-3b-ft-science-qa-448	Science Question Answering	95.93 Accuracy on ScienceQA Img subset with no CoT
paligemma-3b-ft-refcoco-seg-896	Understanding References to Specific Objects in Images	76.94 Mean IoU on refcoco 72.18 Mean IoU on refcoco+ 72.22 Mean IoU on refcocog
paligemma-3b-ft-rsvqa-hr-224	Remote Sensing Visual Question Answering	92.61 Accuracy on test 90.58 Accuracy on test2
Demo
As part of this release we have a demo that wraps the reference implementation in the big_vision repository and provides an easy way to play around with the mix models.

We also have a version of the demo compatible with Transformers, to show how to use the PaliGemma transformers API.

How to Run Inference
To obtain access to the PaliGemma models, you need to accept the Gemma license terms and conditions. If you already have access to other Gemma models in Hugging Face, you’re good to go. Otherwise, please visit any of the PaliGemma models, and accept the license if you agree with it. Once you have access, you need to authenticate either through notebook_login or huggingface-cli login. After logging in, you’ll be good to go!

You can also try inference in this notebook right away.

Using Transformers
You can use the PaliGemmaForConditionalGeneration class to infer with any of the released models. Simply preprocess the prompt and the image with the built-in processor, and then pass the preprocessed inputs for generation.

from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

model_id = "google/paligemma-3b-mix-224"
model = PaliGemmaForConditionalGeneration.from_pretrained(model_id)
processor = AutoProcessor.from_pretrained(model_id)

prompt = "What is on the flower?"
image_file = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg?download=true"
raw_image = Image.open(requests.get(image_file, stream=True).raw)
inputs = processor(prompt, raw_image, return_tensors="pt")
output = model.generate(**inputs, max_new_tokens=20)

print(processor.decode(output[0], skip_special_tokens=True)[len(prompt):])
# bee

You can also load the model in 4-bit as follows.

from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)
model = PaligemmaForConditionalGeneration.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map={"":0}
)

In addition to 4-bit (or 8-bit) loading, the transformers integration allows you to leverage other tools in the Hugging Face ecosystem, such as:

Training and inference scripts and examples
Serialization to safe files (safetensors)
Integrations with tools such as PEFT (parameter efficient fine-tuning)
Utilities and helpers to run generation with the model
Detailed Inference Process
If you want to write your own pre-processing or training code or would like to understand in more detail how PaliGemma works, these are the steps that the input image and text go through.

The input text is tokenized normally. A <bos> token is added at the beginning, and an additional newline token (\n) is appended. This newline token is an essential part of the input prompt the model was trained with, so adding it explicitly ensures it's always there. The tokenized text is also prefixed with a fixed number of <image> tokens. How many? It depends on the input image resolution and the patch size used by the SigLIP model. PaliGemma models are pre-trained on one of three square sizes (224x224, 448x448, or 896x896), and always use a patch size of 14. Therefore, the number of <image> tokens to prepend is 256 for the 224 models (224/14 * 224/14), 1024 for the 448 models, and 4096 for the 896 models.

Note that larger images result in much longer input sequences, and therefore require a lot more memory to go through the language portion of the model. Keep this in mind when considering what model to use. For finer-grained tasks, such as OCR, larger images may help achieve better results, but the incremental quality is small for the vast majority of tasks. Do test on your tasks before deciding to move to a larger resolution!

This complete "prompt" goes through the text embeddings layer of the language model and generates token embeddings with 2048 dimensions per token.

In parallel with this, the input image is resized, using bicubic resampling, to the required input size (224x224 for the smallest-resolution models). Then it goes through the SigLIP Image Encoder to generate image embeddings with 1152 dimensions per patch. This is where the linear projector comes into play: the image embeddings are projected to obtain representations with 2048 dimensions per patch, same as the ones obtained from the text tokens. The final image embeddings are then merged with the <image> text embeddings, and this is the final input that is used for autoregressive text generation. Generation works normally in autoregressive mode. It uses full block attention for the complete input (image + bos + prompt + \n), and a causal attention mask for the generated text.

All of these details are taken care of automatically in the processor and model classes, so inference can be performed using the familiar high-level transformers API shown in the previous examples.

Fine-tuning
Using big_vision
PaliGemma was trained in the big_vision codebase. The same codebase was already used to develop models like BiT, the original ViT, LiT, CapPa, SigLIP, and many more.

The project config folder configs/proj/paligemma/ contains a README.md. The pretrained model can be transferred by running config files in the transfers/ subfolder, and all our transfer results were obtained by running the configs provided therein. If you want to transfer your own model, fork the example config transfers/forkme.py and follow the instructions in the comments to adapt it to your usecase.

There is also a Colab finetune_paligemma.ipynb that runs a simplified fine-tuning that works on a free T4 GPU runtime. To fit on the limited host and GPU memory, the code in the Colab only updates the weights in the attention layers (170M params) and uses SGD (instead of Adam).

Using transformers
Fine-tuning PaliGemma is very easy, thanks to transformers. One can also do QLoRA or LoRA fine-tuning. In this example, we will briefly fine-tune the decoder, and then show how to switch to QLoRA fine-tuning. We will install the latest version of the transformers library.

pip install transformers

Just like on the inference section, we will authenticate to access the model using notebook_login().

from huggingface_hub import notebook_login
notebook_login()

For this example, we will use the VQAv2 dataset, and fine-tune the model to answer questions about images. Let’s load the dataset. We will only use the columns question, multiple_choice_answer and image, so let’s remove the rest of the columns as well. We will also split the dataset.

from datasets import load_dataset 
ds = load_dataset('HuggingFaceM4/VQAv2', split="train") 
cols_remove = ["question_type", "answers", "answer_type", "image_id", "question_id"] 
ds = ds.remove_columns(cols_remove)
ds = ds.train_test_split(test_size=0.1)
train_ds = ds["train"]
val_ds = ds["test"]

We will now load the processor, which contains the image processing and tokenization part, and preprocess our dataset.

from transformers import PaliGemmaProcessor 
model_id = "google/paligemma-3b-pt-224"
processor = PaliGemmaProcessor.from_pretrained(model_id)

We will create a prompt template to condition PaliGemma to answer visual questions. Since the tokenizer pads the inputs, we need to set the pads in our labels to something other than the pad token in the tokenizer, as well as the image token.

import torch
device = "cuda"

image_token = processor.tokenizer.convert_tokens_to_ids("<image>")
def collate_fn(examples):
  texts = ["answer " + example["question"] for example in examples]
  labels= [example['multiple_choice_answer'] for example in examples]
  images = [example["image"].convert("RGB") for example in examples]
  tokens = processor(text=texts, images=images, suffix=labels,
                    return_tensors="pt", padding="longest")

  tokens = tokens.to(torch.bfloat16).to(device)
  return tokens

You can either load the model directly or load the model in 4-bit for QLoRA. Below you can see how to load the model directly. We will load the model, and freeze the image encoder and the projector, and only fine-tune the decoder. If your images are within a particular domain, which might not be in the dataset the model was pre-trained with, you might want to skip freezing the image encoder.

model = PaliGemmaForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.bfloat16).to(device)

for param in model.vision_tower.parameters():
    param.requires_grad = False

for param in model.multi_modal_projector.parameters():
    param.requires_grad = True

If you want to load model in 4-bit for QLoRA, you can add the following changes below.

from transformers import BitsAndBytesConfig
from peft import get_peft_model, LoraConfig

bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_type=torch.bfloat16
)

lora_config = LoraConfig(
    r=8, 
    target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
    task_type="CAUSAL_LM",
)
model = PaliGemmaForConditionalGeneration.from_pretrained(model_id, quantization_config=bnb_config, device_map={"":0})
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
#trainable params: 11,298,816 || all params: 2,934,634,224 || trainable%: 0.38501616002417344

We will now initialize the Trainer and TrainingArguments. If you will do QLoRA fine-tuning, set the optimizer to paged_adamw_8bit instead.

from transformers import TrainingArguments
args=TrainingArguments(
            num_train_epochs=2,
            remove_unused_columns=False,
            per_device_train_batch_size=16,
            gradient_accumulation_steps=4,
            warmup_steps=2,
            learning_rate=2e-5,
            weight_decay=1e-6,
            adam_beta2=0.999,
            logging_steps=100,
            optim="adamw_hf",
            save_strategy="steps",
            save_steps=1000,
            push_to_hub=True,
            save_total_limit=1,
            bf16=True,
            report_to=["tensorboard"],
            dataloader_pin_memory=False
        )

Initialize Trainer, pass in the datasets, data collating function and training arguments, and call train() to start training.

trainer = Trainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collate_fn,
        args=args
        )
trainer.train()

Additional Resources
Vision Language Models Explained
Model docs
Notebook for inference
Big vision PaliGemma demo
🤗 transformers PaliGemma demo
Collection for all PaliGemma models
Collection for all PaliGemma fine-tuned models
Original Implementation
We would like to thank Omar Sanseviero, Lucas Beyer, Xiaohua Zhai and Matthias Minderer for their thorough reviews on this blog post. We would like to thank Peter Robicheaux for their help with fine-tuning changes in transformers.

More Articles from our Blog

