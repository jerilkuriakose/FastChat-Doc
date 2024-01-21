# FastChat-Doc

FastChat is an open-source platform designed for training, serving, and evaluating large language models (LLMs) for chatbot applications. It offers a comprehensive set of features that cater to various needs, making it a popular choice for LLM developers and enthusiasts.

Here's a breakdown of its key aspects:

**Core Features:**

* **Pre-trained Models:** FastChat comes with pre-trained weights and training/evaluation code for state-of-the-art models like Vicuna and FastChat-T5. These models are known for their impressive performance in chatbot tasks.
* **Multi-Model Serving System:** FastChat boasts a distributed multi-model serving system with a user-friendly web interface and OpenAI-compatible RESTful APIs. This allows you to easily deploy and manage multiple LLM models simultaneously.
* **OpenAI Compatibility:** The OpenAI-compatible APIs make FastChat a seamless drop-in replacement for OpenAI in existing applications. This simplifies integration and leverages familiar tools and workflows.
* **Chatbot Arena:** FastChat powers Chatbot Arena, a platform for hosting LLM battles and comparing their performance through user votes. This fosters a competitive environment for LLM development and provides valuable insights for researchers and developers.

**Benefits of using FastChat:**

* **Flexibility:** Train, serve, and evaluate diverse LLM models with a single platform.
* **Ease of Use:** The web UI and OpenAI compatibility make interaction intuitive and effortless.
* **Scalability:** The distributed architecture allows for handling large volumes of chat requests efficiently.
* **Openness:** The open-source nature fosters community contributions and continuous improvement.

## Install

### Method 1: With pip

```bash
pip3 install "fschat[model_worker,webui]"
```

### Method 2: From source

1. Clone this repository and navigate to the FastChat folder

```bash
git clone https://github.com/lm-sys/FastChat.git
cd FastChat
```

If you are running on Mac:

```bash
brew install rust cmake
```

2. Install Package

```bash
pip3 install --upgrade pip  # enable PEP 660 support
pip3 install -e ".[model_worker,webui]"
```

## Simple test to check whether the installation is working fine

#### Single GPU

The command below requires around 14GB of GPU memory for Vicuna-7B and 28GB of GPU memory for Vicuna-13B.
`--model-path` can be a local folder or a Hugging Face repo name.

```
python3 -m fastchat.serve.cli --model-path lmsys/vicuna-7b-v1.5
```

#### Multiple GPUs

You can use model parallelism to aggregate GPU memory from multiple GPUs on the same machine.

```
python3 -m fastchat.serve.cli --model-path lmsys/vicuna-7b-v1.5 --num-gpus 2
```

Tips:
Sometimes the "auto" device mapping strategy in huggingface/transformers does not perfectly balance the memory allocation across multiple GPUs.
You can use `--max-gpu-memory` to specify the maximum memory per GPU for storing model weights.
This allows it to allocate more memory for activations, so you can use longer context lengths or larger batch sizes. For example,

```
python3 -m fastchat.serve.cli --model-path lmsys/vicuna-7b-v1.5 --num-gpus 2 --max-gpu-memory 8GiB
```

#### Not Enough Memory

If you do not have enough memory, you can enable 8-bit compression by adding `--load-8bit` to commands above.
This can reduce memory usage by around half with slightly degraded model quality.
It is compatible with the CPU, GPU, and Metal backend.

Vicuna-13B with 8-bit compression can run on a single GPU with 16 GB of VRAM, like an Nvidia RTX 3090, RTX 4080, T4, V100 (16GB), or an AMD RX 6800 XT.

```
python3 -m fastchat.serve.cli --model-path lmsys/vicuna-7b-v1.5 --load-8bit
```

## FastChat Server Architecture

![server arch](assets/server_arch.png)

## Setup the server in local

### Serving with Web GUI

Here are the commands to follow in your terminal:

#### Launch the controller

```bash
python3 -m fastchat.serve.controller
```

This controller manages the distributed workers.

#### Launch the model worker(s)

```bash
python3 -m fastchat.serve.model_worker --model-path lmsys/vicuna-7b-v1.5
```

Wait until the process finishes loading the model and you see `"Uvicorn running on ..."`. The model worker will register itself to the controller .

To ensure that your model worker is connected to your controller properly, send a test message using the following command:

```bash
python3 -m fastchat.serve.test_message --model-name vicuna-7b-v1.5
```

You will see a short output.

#### Launch the Gradio web server

```bash
python3 -m fastchat.serve.gradio_web_server
```

The following will be the output GUI

<a href="/"><img src="assets/demo_narrow.gif" width="70%"></a>

> The above command will only only open `Single Tab` GUI

#### Launch the Gradio web server for multi tab (battle, arena, etc)

```bash
python3 -m fastchat.serve.gradio_web_server_multi
```

## vLLM Integration

[vLLM](https://vllm.ai/) can be used as an optimized worker implementation in FastChat.
It offers advanced continuous batching and a much higher (~10x) throughput.

### Instructions

1. Install vLLM.

   ```
   pip install vllm
   ```
2. When you launch a model worker, replace the normal worker (`fastchat.serve.model_worker`) with the vLLM worker (`fastchat.serve.vllm_worker`). All other commands such as controller, gradio web server, and OpenAI API server are kept the same.

   ```
   python3 -m fastchat.serve.vllm_worker --model-path lmsys/vicuna-7b-v1.5
   ```

   If you see tokenizer errors, try

   ```
   python3 -m fastchat.serve.vllm_worker --model-path lmsys/vicuna-7b-v1.5 --tokenizer hf-internal-testing/llama-tokenizer
   ```

   If you use an AWQ quantized model, try
   '''
   python3 -m fastchat.serve.vllm_worker --model-path TheBloke/vicuna-7B-v1.5-AWQ --quantization awq
   '''

## OpenAI API Server

First, launch the controller

```bash
python3 -m fastchat.serve.controller
```

Then, launch the model worker(s)

```bash
python3 -m fastchat.serve.model_worker --model-path lmsys/vicuna-7b-v1.5
```

Finally, launch the RESTful API server

```bash
python3 -m fastchat.serve.openai_api_server --host localhost --port 8000
```

### Testing the OpenAI API Server

First, install OpenAI python package >= 1.0:

```bash
pip install --upgrade openai
```

Then, interact with the Vicuna model:

```python
import openai

openai.api_key = "EMPTY"
openai.base_url = "http://localhost:8000/v1/"

model = "vicuna-7b-v1.5"
prompt = "Once upon a time"

# create a completion
completion = openai.completions.create(model=model, prompt=prompt, max_tokens=64)
# print the completion
print(prompt + completion.choices[0].text)

# create a chat completion
completion = openai.chat.completions.create(
  model=model,
  messages=[{"role": "user", "content": "Hello! What is your name?"}]
)
# print the completion
print(completion.choices[0].message.content)
```

## Setup FastChat in multiple nodes

### node-01

Start the controller

```bash
python3 -m fastchat.serve.controller --host 0.0.0.0 --port 10002
```

Start the vLLM model workers

```bash
CUDA_VISIBLE_DEVICES=0 python3 -m fastchat.serve.vllm_worker \
        --model-path lmsys/vicuna-13b-v1.5 \
        --model-name vicuna-13b \
        --controller http://node-01:10002 \
        --host 0.0.0.0 \
        --port 31000 \
        --worker-address http://$(hostname):31000
```

```bash
CUDA_VISIBLE_DEVICES=1 python3 -m fastchat.serve.vllm_worker \
        --model-path lmsys/vicuna-13b-v1.5 \
        --model-name vicuna-13b \
        --controller http://node-01:10002 \
        --host 0.0.0.0 \
        --port 31001 \
        --worker-address http://$(hostname):31001
```

Start Ray Head

```bash
CUDA_VISIBLE_DEVICES=2,3 ray start --head
```

```bash
python3 -m fastchat.serve.vllm_worker \
        --model-path lmsys/vicuna-33b-v1.3 \
        --model-name vicuna-33b \
        --controller http://node-01:10002 \
        --host 0.0.0.0 \
        --port 31002 \
        --worker-address http://$(hostname):31002 \
        --num-gpus 2
```

### node-02

```bash
CUDA_VISIBLE_DEVICES=0 python3 -m fastchat.serve.vllm_worker \
        --model-path meta-llama/Llama-2-13b-chat-hf \
        --model-name llama-2-13b-chat \
        --controller http://node-01:10002 \
        --host 0.0.0.0 \
        --port 31000 \
        --worker-address http://$(hostname):31000 \
        --tokenizer meta-llama/Llama-2-7b-chat-hf
```

```bash
CUDA_VISIBLE_DEVICES=1 python3 -m fastchat.serve.vllm_worker \
        --model-path meta-llama/Llama-2-13b-chat-hf \
        --model-name llama-2-13b-chat \
        --controller http://node-01:10002 \
        --host 0.0.0.0 \
        --port 31001 \
        --worker-address http://$(hostname):31001 \
        --tokenizer meta-llama/Llama-2-7b-chat-hf
```

```bash
CUDA_VISIBLE_DEVICES=2 python3 -m fastchat.serve.vllm_worker \
        --model-path meta-llama/Llama-2-7b-chat-hf \
        --model-name llama-2-7b-chat \
        --controller http://node-01:10002 \
        --host 0.0.0.0 \
        --port 31002 \
        --worker-address http://$(hostname):31002 \
        --tokenizer meta-llama/Llama-2-7b-chat-hf
```

```bash
CUDA_VISIBLE_DEVICES=3 python3 -m fastchat.serve.vllm_worker \
        --model-path WizardLM/WizardLM-13B-V1.1 \
        --model-name wizardlm-13b  \
        --controller http://node-01:10002 \
        --host 0.0.0.0 \
        --port 31003 \
        --worker-address http://$(hostname):31003
```

### node-03

```bash
python3 -m fastchat.serve.vllm_worker \
        --model-path mosaicml/mpt-30b-chat \
        --controller http://node-01:10002 \
        --host 0.0.0.0 \
        --port 31000 \
        --worker-address http://$(hostname):31000 \
        --num-gpus 2
```

```bash
python3 -m fastchat.serve.vllm_worker \
        --model-path timdettmers/guanaco-33b-merged \
        --model-name guanaco-33b  \
        --controller http://node-01:10002 \
        --host 0.0.0.0 \
        --port 31002 \
        --worker-address http://$(hostname):31002 \
        --num-gpus 2 \
        --tokenizer hf-internal-testing/llama-tokenizer
```

### node-04

```bash
CUDA_VISIBLE_DEVICES=0 python3 -m fastchat.serve.multi_model_worker \
        --model-path ~/model_weights/RWKV-4-Raven-14B-v12-Eng98%25-Other2%25-20230523-ctx8192.pth \
        --model-name RWKV-4-Raven-14B \
        --model-path lmsys/fastchat-t5-3b-v1.0 \
        --model-name fastchat-t5-3b \
        --controller http://node-01:10002 \
        --host 0.0.0.0 \
        --port 31000 \
        --worker http://$(hostname):31000 \
        --limit 4
```

```bash
CUDA_VISIBLE_DEVICES=1 python3 -m fastchat.serve.multi_model_worker \
        --model-path OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5 \
        --model-name oasst-pythia-12b \
        --model-path mosaicml/mpt-7b-chat \
        --model-name mpt-7b-chat \
        --controller http://node-01:10002 \
        --host 0.0.0.0 \
        --port 31001 \
        --worker http://$(hostname):31001 \
        --limit 4
```

```bash
CUDA_VISIBLE_DEVICES=2 python3 -m fastchat.serve.multi_model_worker \
        --model-path lmsys/vicuna-7b-v1.5 \
        --model-name vicuna-7b \
        --model-path THUDM/chatglm-6b \
        --model-name chatglm-6b \
        --controller http://node-01:10002 \
        --host 0.0.0.0 \
        --port 31002 \
        --worker http://$(hostname):31002 \
        --limit 4
```

```bash
CUDA_VISIBLE_DEVICES=3 python3 -m fastchat.serve.vllm_worker \
        --model-path ~/model_weights/alpaca-13b  \
        --controller http://node-01:10002 \
        --host 0.0.0.0 \
        --port 31003 \
        --worker-address http://$(hostname):31003
```

### Test the deployment

```bash
python3 -m fastchat.serve.test_message \
        --model vicuna-13b \
        --controller http://localhost:10002
```

## Conversations

### Chatbot Arena Conversations

1. Gather battles

```
python3 clean_battle_data.py --max-num 10 --mode conv_release
```

2. Tag OpenAI moderation

```
python3 tag_openai_moderation.py --in clean_battle_conv_20230814.json
```

3. Clean PII
4. Filter additional blocked words

```
python3 filter_bad_conv.py --in clean_battle_conv_20230630_tagged_v1_pii.json
```

5. Add additional toxicity tag

### All Conversations

1. Gather chats

```
python3 clean_chat_data.py
```

2. Sample

```
python3 conv_release_scripts/sample.py
```

## Custom Dataset

### Dataset Format

[Sample Format](https://github.com/lm-sys/FastChat/blob/main/fastchat/llm_judge/data/mt_bench/question.jsonl "sample format")

### Install
```bash
git clone https://github.com/lm-sys/FastChat.git
cd FastChat
pip install -e ".[model_worker,llm_judge]"
```

### Review Pre-Generated Model Answers and Judgments
We provide pre-generated model answers and judgments for some models.
You can view them at this [demo](https://huggingface.co/spaces/lmsys/mt-bench).

To download the pre-generated data, use
```
python3 download_mt_bench_pregenerated.py
```

After downloading the data, you can view them locally by
```
python3 qa_browser.py --share
```
You can use this QA browser to view the answers generated by you later.

## MT-Bench

### Evaluate a model on MT-bench

#### Step 1. Generate model answers to MT-bench questions
```bash
python gen_model_answer.py --model-path [MODEL-PATH] --model-id [MODEL-ID]
```
Arguments:
  - `[MODEL-PATH]` is the path to the weights, which can be a local folder or a Hugging Face repo ID.
  - `[MODEL-ID]` is a name you give to the model.

e.g.,
```bash
python gen_model_answer.py --model-path lmsys/vicuna-7b-v1.5 --model-id vicuna-7b-v1.5
```
The answers will be saved to `data/mt_bench/model_answer/[MODEL-ID].jsonl`.

You can also specify `--num-gpus-per-model` for model parallelism (needed for large 65B models) and `--num-gpus-total` to parallelize answer generation with multiple GPUs.

#### Step 2. Generate GPT-4 judgments
There are several options to use GPT-4 as a judge, such as pairwise winrate and single-answer grading.
In MT-bench, we recommend single-answer grading as the default mode.
This mode asks GPT-4 to grade and give a score to model's answer directly without pairwise comparison.
For each turn, GPT-4 will give a score on a scale of 10. We then compute the average score on all turns.

```bash
export OPENAI_API_KEY=XXXXXX  # set the OpenAI API key
python gen_judgment.py --model-list [LIST-OF-MODEL-ID] --parallel [num-concurrent-api-call]
```

e.g.,
```bash
python gen_judgment.py --model-list vicuna-13b-v1.3 alpaca-13b llama-13b claude-v1 gpt-3.5-turbo gpt-4 --parallel 2
```
The judgments will be saved to `data/mt_bench/model_judgment/gpt-4_single.jsonl`

#### Step 3. Show MT-bench scores

- Show the scores for selected models
  ```bash
  python show_result.py --model-list vicuna-13b-v1.3 alpaca-13b llama-13b claude-v1 gpt-3.5-turbo gpt-4
  ```
- Show all scores
  ```bash
  python show_result.py
  ```

### Other grading options
Besides score-based single-answer grading, we also support two additional grading options based on win rates:
- `pariwise-baseline`: run pairwise comparison against a baseline model.
- `pairwise-all`: run pairwise comparison between all model pairs on all questions.

#### Option 2: pairwise comparison against a baseline (default: gpt-3.5-turbo)

- Generate GPT-4 judgments
```bash
python gen_judgment.py --mode pairwise-baseline --model-list vicuna-13b-v1.3 alpaca-13b llama-13b --parallel 2
```
The judgments will be saved to `data/mt_bench/model_judgment/gpt-4_pair.jsonl`

- Show results
```bash
python show_result.py --mode pairwise-baseline
```

#### Option 3: Run GPT-4 judge with all pair comparisons

Another option is to run pairwise comparisons on all possible pairs.
This could be more expensive when #models increases, but it gives you a more comprehensive information.

```bash
python gen_judgment.py --mode pairwise-all --model-list [LIST-OF-MODEL-ID] --parallel [num-concurrent-api-call]
```

```bash
python show_result.py --mode pairwise-all
```

### Evaluate a custom model on custom bench

#### Step 1. Download or save the custom bench question to the following location

[Question Format](https://github.com/lm-sys/FastChat/blob/main/fastchat/llm_judge/data/mt_bench/question.jsonl "format")

```bash
data/allam_bench/question.jsonl
```

#### Step 2. Generate model answers to custom bench questions
```bash
python gen_model_answer.py \
        --model-path /models/allam-13b \
        --model-id allam-13b \
        --bench-name allam_bench
```
The answers will be saved to `data/allam_bench/model_answer/[MODEL-ID].jsonl`.
