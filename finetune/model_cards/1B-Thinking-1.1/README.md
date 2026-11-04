---
license: apache-2.0
datasets:
- OpenLLM-France/Luciole-PostTraining-Dataset-1.1
language:
- fr
- en
base_model:
- OpenLLM-France/Luciole-1B-Base
tags:
- openllm-france
---
![luciole_logo.png](luciole_logo.png)

# Model Card for Luciole-1B-Thinking-1.1

* [Model Description](#model-description)
  * [Bias, Risks, and Limitations](#bias-risks-and-limitations)
  * [Recommendations](#recommendations)
* [Training Details](#training-details)
  * [Training Data](#training-data)
  * [Instruction template](#instruction-template)
  * [Training Procedure](#training-procedure)
* [Evaluation](#evaluation)
* [Using the model](#using-the-model)
  * [With ollama](#with-ollama)
  * [With vLLM](#with-vllm)
* [Citation](#citation)
* [Acknowledgements](#acknowledgements)
* [Contact](#contact)

> [!NOTE]
> **NOTE:** Although this model has undergone safety alignment during the DPO thinking phase, it could still generate unsafe output for certain inputs. The model also has not been trained on identity prompts, so it may not correctly answer the questions about its name/origin. The Luciole models are under active development and we plan to address these issues in future versions.  


## Model Description

Luciole-1B-Thinking-1.1 is a fine-tuned and aligned version of [Luciole-1B-Base](https://huggingface.co/OpenLLM-France/Luciole-1B-Base), an open-source, multilingual causal language model created by OpenLLM-France. Luciole-1B-Thinking-1.1  was developed by [LINAGORA](https://labs.linagora.com/) and the [OpenLLM-France](https://openllm-france.fr/) consortium as a part of the OpenLLM France project, funded by [BPI France](https://www.bpifrance.fr/) through the [France 2030](https://www.info.gouv.fr/grand-dossier/france-2030) program.

Training of Luciole-1B-Thinking-1.1 was conducted on [GENCI](https://www.genci.fr/) supercomputer Jean Zay, managed by [IDRIS](http://www.idris.fr/docs/idris/missions). It was trained in two phases: (i) a supervised fine-tuning (SFT) phase on instruction data with thinking traces, and (ii) a final preference alignment phase using Direct Preference Optimization (DPO). The training data covers topics in math, science, coding, general chat, RAG and translation. 

Note that Luciole-1B-Thinking-1.1 is only the first iteration of post-trained models based on [Luciole-1B-Base](https://huggingface.co/OpenLLM-France/Luciole-1B-Base). Development of the Luciole models is an active, ongoing project and further iterations, including models trained on higher proportions of French data, are planned. In the spirit of open-source, we share the model weights, training recipes, and the training data to facilitate research and to provide open source building blocks for other training projects.   

If you are interested in contributing to the Luciole project, contact us at contact@openllm-france.fr. 

* License: [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0)
* Training repository: [Luciole-Training](https://github.com/OpenLLM-France/Luciole-Training/tree/main/finetune)
* Training data: [Luciole-PostTraining-Dataset-1.1](https://huggingface.co/datasets/OpenLLM-France/Luciole-PostTraining-Dataset-1.1)
* Technical report: coming soon.


### Bias, Risks, and Limitations
Luciole-1B-Thinking-1.1 is the result of a first phase of fine-tuning and alignment to human preferences. Efforts to improve the model are active and ongoing and the model should be thoroughly tested for target use cases before being incorporated in industrial pipelines. Training has mostly focused on general instruction following, and additional training would be necessary for tasks specifically focused on code generation and mathematical problem solving. While the model has seen some tool calling data, it has a tendency to over-refuse tool calling requests. These capacities are experimental and will be improved in future versions.

Luciole-1B-Thinking-1.1 was post-trained almost entirely on English data (in contrast to its base model, [OpenLLM-France/Luciole-1B-Base](https://huggingface.co/OpenLLM-France/Luciole-1B-Base), which was trained on roughly 30% French data). Future post-training phases will focus specifically on increasing the proportion of French data and testing the impact that this has on model performance. 

Due to its size, Luciole-1B-Thinking-1.1 is limited in the information that it can memorize. It can also hallucinate; that is, produce false answers that result from its training on massive amounts of diverse text. Its ability to produce correct answers could be improved by implementing the model in a retrieval augmented generation pipeline. 

Finally,  Luciole-1B-Thinking-1.1 was trained on sequences of 16,384 tokens (in contrast to its base model, whose context window was extended to 131,000 tokens). 


### Recommendations
* Further train Luciole-1B-Thinking-1.1 for specific use cases.
* Integrate the model in a RAG pipeline to augment its knowledge base.
* Extend training with longer sequences to increase context length.


## Training details

### Training data
The post-training data and dataset description is available [here](https://huggingface.co/datasets/OpenLLM-France/Luciole-PostTraining-Dataset-1.1).



### Training procedure

Luciole-1B-Thinking-1.1 was trained in two phases. The model was first finetuned on thinking data, followed by DPO on preference data with thinking traces. 

|                            | **SFT with thinking** |  **DPO with thinking** |  
|----------------------------|:----------:|:------------:|
| **Num training samples**   |     2.6M     |      284K   |  
| **Num epochs**             |       4      |      1      |  
| **Max learning rate**      |     2.0e-5     |       3.0e-8      | 
| **Min learning rate**      |      2.0e-6  |       0       |   
| **LR scheduler**           |      cosine    |      linear       |  
| **Max sequence length**    |     16384    |     16384        | 
| **Batch size**             |    2048      |        128       |  
| **DPO β**            |     n/a     |     5       |  

Links to the interim models: [SFT with thinking](https://dl.labs.linagora.com/files/models/OpenLLM-France/Luciole-1B-SFT-Thinking-1.1/).


## Evaluation

The two figures compare the performance of Luciole 1.1 Thinking models with other thinking models on English and French benchmarks. All the models were evaluated with ```temperature = 0``` and a thinking budget of 5000 tokens. For the Olmo models, we ran evaluation on Think-DPO variant as Olmo-Think had undergone an additional stage of RLVR training.

For Luciole-1B-Thinking-1.1, there is a dearth of thinking models of similar size for comparison. While Luciole-1B-Thinking-1.1 lags behind the other larger models on English benchmarks, it outperforms Olmo-3-7B-Think-DPO on ifeval. 

<center>
   <img src='figure_en.png' width='800px'>
</center>

For French benchmarks, Luciole-1B-Thinking-1.1 displays a strong performance on several benchmarks like EIFFEL (a French cultural benchmark), blend, and ifeval-fr.

<center>
   <img src='figure_fr.png' width='800px'>
</center>

## Using the model

Luciole-1B-Thinking-1.1 was trained on the chat template inspired from [Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B/). In our chat template, we use the default system prompt "You are a helpful AI assistant named Luciole, trained by LINAGORA and OpenLLM France.". 


As an example, when applying the chat template to the user query below: 

```
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("OpenLLM-France/Luciole-1B-Thinking-1.1")
query = [
  {
    "content": "Who was Molière?",
    "role": "user"
  }
]
print(tokenizer.apply_chat_template(query, tokenize=False, add_generation_prompt=True))
```
we get the following formatted prompt:
```
<|im_start|>system
You are a helpful AI assistant named Luciole, trained by LINAGORA and OpenLLM France.<|im_end|>
<|im_start|>user
Who was Molière?<|im_end|>
<|im_start|>assistant
```

Given this input, the model first generates a thinking trace, enclosed in thinking tags, before giving its final response:
```
<think>
Okay, the user is asking who Molière was, and they want a concise answer. Let me start by recalling what I know about Molière …
</think>

Jean-Baptiste Poquelin, known as Molière, was a 17th-century French playwright, actor, and poet. Born in Paris in 1622, … <|im_end|>
``` 

Thinking can be turned off by using ```tokenizer.apply_chat_template(…, enable_thinking=False)```. This will append empty thinking tags (```<think>\n\n</think>\n\n```) in the prompt, immediately after "`assistant\n`".
However, for testing a non-thinking model, it is better to conduct experiments on [Luciole-1B-Instruct-1.1](https://huggingface.co/OpenLLM-France/Luciole-1B-Instruct-1.1).

By default, the `<think>…</think>` reasoning of all previous assistant turns is kept when formatting a multi-turn conversation. Setting ```tokenizer.apply_chat_template(…, preserve_thinking=False)``` drops it from every assistant turn preceding the last user message (only the most recent turn keeps its reasoning), which shortens the prompt and reduces the KV-cache footprint. With vLLM, set it per request via ```extra_body={"chat_template_kwargs": {"preserve_thinking": false}}```. Ollama does not use this parameter, as it formats prompts with its own built-in renderer.

The chat template prefills the opening `<think>` tag in the generation prompt, so the model's output starts directly with the reasoning and contains only the closing `</think>` (reasoning, then `</think>`, then the answer). When serving with vLLM, add `--reasoning-parser qwen3` so the reasoning is returned separately in `reasoning_content` and the final answer in `content` (SGLang uses the same `--reasoning-parser qwen3`). This requires a recent vLLM (≥ 0.24; prefer the latest).


### With ollama


Note: for best performance, please use the [GGUF files](https://huggingface.co/OpenLLM-France/Luciole-1B-Thinking-1.1-GGUF) released officially by OpenLLM.

* Download and install [Ollama](https://ollama.com/download)
* Download the [GGUF model](https://huggingface.co/OpenLLM-France/Luciole-1B-Thinking-1.1-GGUF)
* Copy the [`Modelfile`](https://huggingface.co/OpenLLM-France/Luciole-1B-Thinking-1.1-GGUF/blob/main/Modelfile), adpating if necessary the path to the GGUF file (line starting with `FROM`).
* Run in a shell:
    * `ollama create -f Modelfile Luciole`
    * `ollama run Luciole`
* Once ">>>" appears, type your prompt(s) and press Enter.
* Optionally, restart a conversation by typing "`/clear`"
* End the session by typing "`/bye`".

Useful for debug:
* [How to print input requests and output responses in Ollama server?](https://stackoverflow.com/a/78831840)
* [Documentation on Modelfile](https://github.com/ollama/ollama/blob/main/docs/modelfile.mdx#parameter)
   * Examples: [Ollama model library](https://github.com/ollama/ollama#model-library)
      * Llama 3 example: https://ollama.com/library/llama3.1
* Add GUI : https://docs.openwebui.com/

### With vLLM

#### 1. Run vLLM Docker Container

Use the following command to deploy the model,
replacing `INSERT_YOUR_HF_TOKEN` with your Hugging Face Hub token.

```bash
docker run --runtime nvidia --gpus=all \
    --env "HUGGING_FACE_HUB_TOKEN=INSERT_YOUR_HF_TOKEN" \
    -p 8000:8000 \
    --ipc=host \
    vllm/vllm-openai:latest \
    --model OpenLLM-France/Luciole-1B-Thinking-1.1 \
    --reasoning-parser qwen3 \
    --enable-auto-tool-choice --tool-call-parser hermes
```

#### 2. Query the model with the OpenAI Client in Python

To query the deployed model, use the OpenAI Python client as follows:

```python
from openai import OpenAI

# Initialize the client
client = OpenAI(base_url='http://localhost:8000/v1', api_key='empty')

# Define the input content
content = "Hello Luciole"

# Generate a response
chat_response = client.chat.completions.create(
    model="OpenLLM-France/Luciole-1B-Thinking-1.1",
    messages=[
        {"role": "user", "content": content}
    ],
)
print(chat_response.choices[0].message.content)
```

#### 3. Tool calling

With the `--enable-auto-tool-choice --tool-call-parser hermes` flags used above, the model can call functions:
pass a list of `tools` as with the OpenAI API, and the calls chosen by the model are returned in `message.tool_calls`.
Requests without `tools` are not affected.

```python
# Define the tools available to the model
tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string", "description": "City name"}},
            "required": ["city"],
        },
    },
}]

# Generate a response that may contain tool calls
chat_response = client.chat.completions.create(
    model="OpenLLM-France/Luciole-1B-Thinking-1.1",
    messages=[
        {"role": "user", "content": "What is the weather like in Paris?"}
    ],
    tools=tools,
)
print(chat_response.choices[0].message.tool_calls)
```

**Warning:** Tool usage is still a work in progress, see [BFCL v4](https://gorilla.cs.berkeley.edu/leaderboard.html) results for [live](https://huggingface.co/OpenLLM-France/Luciole-1B-Instruct-1.1/blob/main/bfcl_live.png) and [non-live](https://huggingface.co/OpenLLM-France/Luciole-1B-Instruct-1.1/blob/main/bfcl_non_live.png) categories.

## Citation

✍ Paper coming soon!


## Acknowledgements

We gratefully acknowledge BPI France for funding the OpenLLM France project under the [call](https://www.bpifrance.fr/nos-appels-a-projets-concours/appel-a-projets-communs-numeriques-pour-lintelligence-artificielle-generative) "Communs numériques pour l’intelligence artificielle générative" ("Digital commons for generative artificial intelligence") as a part of the [France 2030](https://www.info.gouv.fr/grand-dossier/france-2030) program.

Training of Luciole-1B-Thinking-1.1 was made possible by computing AI and storage resources by [GENCI](https://www.genci.fr/) at [IDRIS](http://www.idris.fr/) thanks to the project A0181016189 on the supercomputer Jean Zay’s H100 partition. We gratefully acknowledge their support. 


Luciole-1B-Thinking-1.1 was created by members of [LINAGORA](https://labs.linagora.com/) and the [OpenLLM-France](https://www.openllm-france.fr/) community, including in alphabetical order:

Akshay Chaturvedi (LINAGORA)  
Liam Duignan (CEA List)  
Olivier Ferret (CEA List)  
Olivier Gouvert (LINAGORA)  
Émile Hazard (Opsci)  
Julie Hunter (LINAGORA)   
Jean-Pierre Lorré (LINAGORA)  
Jérôme Louradour (LINAGORA)   
Michel-Marie Maudet (LINAGORA)  
Kate Thompson (LINAGORA)  
Matteo Van Ypersele de Strihou (LINAGORA) 

Particular thanks to the following OpenLLM France members for their valuable input: Clément Bénesse (Opsci), Christophe Cerisara (LORIA) and Gabriel Lauzzana (LORIA).  
  
We thank the support teams from IDRIS and NVIDIA for technical guidance throughout the project.

We would also like to thank members of the [Gaperon](https://huggingface.co/collections/almanach/gaperon) and [Salamandra](https://huggingface.co/collections/BSC-LT/salamandra)  projects for sharing their insights with us. We also acknowledge the numerous open source actors whose resources have guided us throughout the training process, with particular thanks to [Nvidia](https://www.nvidia.com/en-eu/),  [HuggingFace](https://huggingface.co/) and [Allen AI](https://allenai.org/). 


Finally, we thank the entire OpenLLM-France community, whose members have helped in diverse ways.

## Contact

contact@openllm-france.fr