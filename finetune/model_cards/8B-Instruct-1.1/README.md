---
license: apache-2.0
language:
- fr
- en
pipeline_tag: text-generation
tags:
- openllm-france
base_model:
- OpenLLM-France/Luciole-8B-Base
datasets:
- OpenLLM-France/Luciole-PostTraining-Dataset-1.1
---
![luciole_logo.png](luciole_logo.png)

# Model Card for Luciole-8B-Instruct-1.1

* [Model Description](#model-description)
  * [Bias, Risks, and Limitations](#bias-risks-and-limitations)
  * [Recommendations](#recommendations)
* [Training Details](#training-details)
  * [Training Data](#training-data)
  * [Instruction template](#instruction-template)
  * [Training Procedure](#training-procedure)
* [Evaluation](#evaluation)
* [Testing the model](#testing-the-model)
  * [Test with ollama](#test-with-ollama)
  * [Test with vLLM](#test-with-vllm)
* [Citation](#citation)
* [Acknowledgements](#acknowledgements)
* [Contact](#contact)

## Model Description

Luciole-8B-Instruct-1.1 is a fine-tuned and aligned version of [Luciole-8B-Base](https://huggingface.co/OpenLLM-France/Luciole-8B-Base), an open-source, multilingual causal language model created by OpenLLM-France. Luciole-8B-Instruct-1.1  was developed by [LINAGORA](https://labs.linagora.com/) and the [OpenLLM-France](https://openllm-france.fr/) consortium as a part of the OpenLLM France project, funded by [BPI France](https://www.bpifrance.fr/) through the [France 2030](https://www.info.gouv.fr/grand-dossier/france-2030) program.

Training of Luciole-8B-Instruct-1.1 was conducted on Jean Zay in three phases: (i) a supervised fine-tuning (SFT) phase on instruction data with thinking traces, (ii) an SFT phase on instruction data without thinking traces, and (iii) a final preference alignment phase using Direct Preference Optimization (DPO). The training data covers topics in math, science, coding, general chat, RAG, and translation. 

Note that Luciole-8B-Instruct-1.1 is only the first iteration of post-trained models based on [Luciole-8B-Base](https://huggingface.co/OpenLLM-France/Luciole-8B-Base). Development of the Luciole models is an active, ongoing project and further iterations, including models trained on higher proportions of French data, are planned. In the spirit of open-source, we share the model weights and training recipes to facilitate research and to provide open source building blocks for other training projects.   

If you are interested in contributing to the Luciole project, contact us at contact@openllm-france.fr. 

* License: [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0)
* Training repository: [Luciole-Training](https://github.com/OpenLLM-France/Luciole-Training/tree/main/finetune)
* Training data: [Luciole-PostTraining-Dataset-1.1](https://huggingface.co/datasets/OpenLLM-France/Luciole-PostTraining-Dataset-1.1)
* Technical report: coming soon.


### Bias, Risks, and Limitations
Luciole-8B-Instruct-1.1 is the result of a first phase of fine-tuning and alignment to human preferences. Efforts to improve the model are active and ongoing and the model should be thoroughly tested for target use cases before being incorporated in industrial pipelines. Training has mostly focused on instruction following, and additional training would be necessary for tasks specifically focused on code generation or mathematical problem solving. It is also susceptible to hallucinations; that is, producing false answers that result from its training on massive amounts of diverse text. 

Luciole-8B-Instruct-1.1 was post-trained almost entirely on English data (in contrast to its base model, [OpenLLM-France/Luciole-8B-Base](https://huggingface.co/OpenLLM-France/Luciole-8B-Base), which was trained on roughly 30% French data). Future post-training phases will focus specifically on increasing the proportion of French data and testing the impact that this has on model performance. 

Due to its size, Luciole-8B-Instruct-1.1 is limited in the information that it can memorize; its ability to produce correct answers could be improved by implementing the model in a retrieval augmented generation pipeline.

Finally,  Luciole-8B-Instruct-1.1 was trained on sequences of 16,384 tokens (in contrast to its base model, whose context window was extended to 131,000 tokens). 

### Recommendations
* Further train Luciole-8B-Instruct-1.1 for specific use cases.
* Integrate the model in a RAG pipeline to augment its knowledge base.
* Extend training with longer sequences to increase context length.


## Training details

### Training data
The post-training data and dataset description is available [here](https://huggingface.co/datasets/OpenLLM-France/Luciole-PostTraining-Dataset-1.1).

### Instruction template
Luciole-8B-Instruct-1.1 was trained on the chat template inspired from [Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B/). In our chat template, we also use a default system prompt.


As an example, when applying the chat template to the conversation below:

```
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("OpenLLM-France/Luciole-8B-Instruct-1.1")
chat = [
  {
    "content": "Who was Molière?",
    "role": "user"
  },
  {
    "content": "Molière was a 17th-century French playwright, actor, and comedian.",
    "role": "assistant"
  },
  {
    "content": "What are his best-known works?",
    "role": "user"
  }
]
print(tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True))
```
we get the following output
```
<|im_start|>system
You are a helpful AI assistant named Luciole, trained by LINAGORA and OpenLLM France.<|im_end|>
<|im_start|>user
Who was Molière?<|im_end|>
<|im_start|>assistant
Molière was a 17th-century French playwright, actor, and comedian.<|im_end|>
<|im_start|>user
What are his best-known works?<|im_end|>
<|im_start|>assistant
```
### Training procedure

Luciole-8B-Instruct-1.1 was trained in three phases. The model was first finetuned on thinking data, and then on data without thinking traces. The resulting instruct model was aligned during a final DPO finetuning phase.


|                            | **SFT with thinking** | **SFT without thinking**  | **DPO without thinking** |  
|----------------------------|:----------:|:-----------:|:------------:|
| **Num training samples**   |     2.6M     |     2.1M   |      284K   |  
| **Num epochs**             |       4      |      4     |      1      |  
| **Max learning rate**      |     2.0e-5     |    2.0e-5       |      1.0e-6      |
| **Min learning rate**      |      2.0e-6  |       2.0e-6    |     0       |    
| **LR scheduler**           |      cosine    |      cosine     |     linear       |  
| **Max sequence length**    |     16384    |     16384      |    16384        | 
| **Batch size**             |    2048      |      2048     |     128       |  
| **DPO β**            |     n/a     |     n/a      |     5       |  

Links to the interim models: [SFT with thinking](https://dl.labs.linagora.com/files/models/OpenLLM-France/Luciole-8B-SFT-Thinking-1.1/), [SFT without thinking](https://dl.labs.linagora.com/files/models/OpenLLM-France/Luciole-8B-SFT-1.1/)

<!--<sup>*</sup>As noted above, while Luciole-8B-SFT-1.1 is trained on sequences of 16384 tokens, it maintains the capacity of the original base model, [OpenLLM-France/Luciole-8B-Base](https://huggingface.co/OpenLLM-France/Luciole-8B-Base), to handle context sizes of up to 131K tokens.-->

## Evaluation
Overall, Luciole-8B-Instruct-1.1 performs comparably with other open source instruct models of the same size. The model approaches Qwen3 in instruction following and outperforms the other models on several benchmarks. It is also notable that Luciole-8B-Instruct-1.1 outperforms most other open source multilingual models on French language instruction despite being finetuned on mostly English language data. This is likely the effect of seeing a larger proportion of French data during pretraining.  


<center>
   <img src='benchmark_eval.png' width='800px'>
</center>

## Testing the model

### Test with ollama
Note: for best performance, please use the [GGUF files](https://huggingface.co/OpenLLM-France/Luciole-8B-Instruct-1.1-GGUF) released officially by OpenLLM.

* Download and install [Ollama](https://ollama.com/download)
* Download the [GGUF model](https://huggingface.co/OpenLLM-France/Luciole-8B-Instruct-1.1-GGUF)
* Copy the [`Modelfile`](https://huggingface.co/OpenLLM-France/Luciole-8B-Instruct-1.1-GGUF/blob/main/Modelfile), adpating if necessary the path to the GGUF file (line starting with `FROM`).
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

### Test with vLLM

#### 1. Run vLLM Docker Container

Use the following command to deploy the model,
replacing `INSERT_YOUR_HF_TOKEN` with your Hugging Face Hub token.

```bash
docker run --runtime nvidia --gpus=all \
    --env "HUGGING_FACE_HUB_TOKEN=INSERT_YOUR_HF_TOKEN" \
    -p 8000:8000 \
    --ipc=host \
    vllm/vllm-openai:latest \
    --model OpenLLM-France/Luciole-8B-Instruct-1.1
```

#### 2. Test using OpenAI Client in Python

To test the deployed model, use the OpenAI Python client as follows:

```python
from openai import OpenAI

# Initialize the client
client = OpenAI(base_url='http://localhost:8000/v1', api_key='empty')

# Define the input content
content = "Hello Luciole"

# Generate a response
chat_response = client.chat.completions.create(
    model="OpenLLM-France/Luciole-8B-Instruct-1.1",
    messages=[
        {"role": "user", "content": content}
    ],
)
print(chat_response.choices[0].message.content)
```

## Citation

✍ Paper coming soon!


## Acknowledgements

We gratefully acknowledge BPI France for funding the OpenLLM France project under the [call](https://www.bpifrance.fr/nos-appels-a-projets-concours/appel-a-projets-communs-numeriques-pour-lintelligence-artificielle-generative) "Communs numériques pour l’intelligence artificielle générative" ("Digital commons for generative artificial intelligence") as a part of the [France 2030](https://www.info.gouv.fr/grand-dossier/france-2030) program.

Training of Luciole-8B-Instruct-1.1 was made possible by computing AI and storage resources by [GENCI](https://www.genci.fr/) at [IDRIS](http://www.idris.fr/) thanks to the project A0181016189 on the supercomputer Jean Zay’s H100 partition. We gratefully acknowledge their support. 


Luciole-8B-Instruct-1.1 was created by members of [LINAGORA](https://labs.linagora.com/) and the [OpenLLM-France](https://www.openllm-france.fr/) community, including in alphabetical order:

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