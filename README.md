<div align="center">
  <img src = "https://github.com/user-attachments/assets/93ff2412-777c-4619-812b-0134eb327cf3" width="600"/>
  <img src="https://github.com/InternLM/lmdeploy/assets/36994684/0cf8d00f-e86b-40ba-9b54-dc8f1bc6c8d8" width="600"/>
  <br /><br />
  

[![GitHub Repo stars](https://img.shields.io/github/stars/InternLM/xtuner?style=social)](https://github.com/InternLM/xtuner/stargazers)
[![license](https://img.shields.io/github/license/InternLM/xtuner.svg)](https://github.com/InternLM/xtuner/blob/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/xtuner)](https://pypi.org/project/xtuner/)
[![Downloads](https://static.pepy.tech/badge/xtuner)](https://pypi.org/project/xtuner/)
[![issue resolution](https://img.shields.io/github/issues-closed-raw/InternLM/xtuner)](https://github.com/InternLM/xtuner/issues)
[![open issues](https://img.shields.io/github/issues-raw/InternLM/xtuner)](https://github.com/InternLM/xtuner/issues)

👋 join us on [![Static Badge](https://img.shields.io/badge/-grey?style=social&logo=wechat&label=WeChat)](https://cdn.vansin.top/internlm/xtuner.jpg)
[![Static Badge](https://img.shields.io/badge/-grey?style=social&logo=twitter&label=Twitter)](https://twitter.com/intern_lm)
[![Static Badge](https://img.shields.io/badge/-grey?style=social&logo=discord&label=Discord)](https://discord.gg/xa29JuW87d)

🔍 Explore our models on
[![Static Badge](https://img.shields.io/badge/-gery?style=social&label=🤗%20Huggingface)](https://huggingface.co/xtuner)
[![Static Badge](https://img.shields.io/badge/-gery?style=social&label=🤖%20ModelScope)](https://www.modelscope.cn/organization/xtuner)
[![Static Badge](https://img.shields.io/badge/-gery?style=social&label=🧰%20OpenXLab)](https://openxlab.org.cn/usercenter/xtuner)
[![Static Badge](https://img.shields.io/badge/-gery?style=social&label=🧠%20WiseModel)](https://www.wisemodel.cn/organization/xtuner)

English | [简体中文](README_zh-CN.md)

</div>

## 🧠 MindMap

![](https://picx.zhimg.com/80/v2-a852a4f3d4b8993e57ad0812cd07fc97_1440w.jpeg)

## 🚀 Training Plan

| Task | Time | State |
| --- | --- | --- |
| crawl raw data | 7.23-7.25 | √   |
| wash the data | 7.25-7.27 | √   |
| convert data to json | 7.28 | √   |
| finish camp3-L1 | 8.1-8.8 | √   |
| finetune | 8.9-8.14 | √   |
| test results | 8.15 | √   |
| finish camp3-L2 | 8.16-8.22 | √   |
| re-finetune | 8.23-8.28 | waiting |



## 📖 Introduction

XTuner is an efficient, flexible and full-featured toolkit for fine-tuning large models.

**Efficient**

- Support LLM, VLM pre-training / fine-tuning on almost all GPUs. XTuner is capable of fine-tuning 7B LLM on a single 8GB GPU, as well as multi-node fine-tuning of models exceeding 70B.
- Automatically dispatch high-performance operators such as FlashAttention and Triton kernels to increase training throughput.
- Compatible with [DeepSpeed](https://github.com/microsoft/DeepSpeed) 🚀, easily utilizing a variety of ZeRO optimization techniques.

**Flexible**

- Support various LLMs ([InternLM](https://huggingface.co/internlm), [Mixtral-8x7B](https://huggingface.co/mistralai), [Llama 2](https://huggingface.co/meta-llama), [ChatGLM](https://huggingface.co/THUDM), [Qwen](https://huggingface.co/Qwen), [Baichuan](https://huggingface.co/baichuan-inc), ...).
- Support VLM ([LLaVA](https://github.com/haotian-liu/LLaVA)). The performance of [LLaVA-InternLM2-20B](https://huggingface.co/xtuner/llava-internlm2-20b) is outstanding.
- Well-designed data pipeline, accommodating datasets in any format, including but not limited to open-source and custom formats.
- Support various training algorithms ([QLoRA](http://arxiv.org/abs/2305.14314), [LoRA](http://arxiv.org/abs/2106.09685), full-parameter fune-tune), allowing users to choose the most suitable solution for their requirements.

**Full-featured**

- Support continuous pre-training, instruction fine-tuning, and agent fine-tuning.
- Support chatting with large models with pre-defined templates.
- The output models can seamlessly integrate with deployment and server toolkit ([LMDeploy](https://github.com/InternLM/lmdeploy)), and large-scale evaluation toolkit ([OpenCompass](https://github.com/open-compass/opencompass), [VLMEvalKit](https://github.com/open-compass/VLMEvalKit)).

## 🔥 Supports

<table>
<tbody>
<tr align="center" valign="middle">
<td>
  <b>Models</b>
</td>
<td>
  <b>SFT Datasets</b>
</td>
<td>
  <b>Data Pipelines</b>
</td>
 <td>
  <b>Algorithms</b>
</td>
</tr>
<tr valign="top">
<td align="left" valign="top">
<ul>
  <li><a href="https://huggingface.co/internlm">InternLM2 / 2.5</a></li>
  <li><a href="https://huggingface.co/meta-llama">Llama 2 / 3</a></li>
  <li><a href="https://huggingface.co/collections/microsoft/phi-3-6626e15e9585a200d2d761e3">Phi-3</a></li>
  <li><a href="https://huggingface.co/THUDM/chatglm2-6b">ChatGLM2</a></li>
  <li><a href="https://huggingface.co/THUDM/chatglm3-6b">ChatGLM3</a></li>
  <li><a href="https://huggingface.co/Qwen/Qwen-7B">Qwen</a></li>
  <li><a href="https://huggingface.co/baichuan-inc/Baichuan2-7B-Base">Baichuan2</a></li>
  <li><a href="https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1">Mixtral</a></li>
  <li><a href="https://huggingface.co/deepseek-ai/DeepSeek-V2-Chat">DeepSeek V2</a></li>
  <li><a href="https://huggingface.co/google">Gemma</a></li>
  <li>...</li>
</ul>
</td>
<td>
<ul>
  <li><a href="https://modelscope.cn/datasets/damo/MSAgent-Bench">MSAgent-Bench</a></li>
  <li><a href="https://huggingface.co/datasets/fnlp/moss-003-sft-data">MOSS-003-SFT</a> 🔧</li>
  <li><a href="https://huggingface.co/datasets/tatsu-lab/alpaca">Alpaca en</a> / <a href="https://huggingface.co/datasets/silk-road/alpaca-data-gpt4-chinese">zh</a></li>
  <li><a href="https://huggingface.co/datasets/WizardLM/WizardLM_evol_instruct_V2_196k">WizardLM</a></li>
  <li><a href="https://huggingface.co/datasets/timdettmers/openassistant-guanaco">oasst1</a></li>
  <li><a href="https://huggingface.co/datasets/garage-bAInd/Open-Platypus">Open-Platypus</a></li>
  <li><a href="https://huggingface.co/datasets/HuggingFaceH4/CodeAlpaca_20K">Code Alpaca</a></li>
  <li><a href="https://huggingface.co/datasets/burkelibbey/colors">Colorist</a> 🎨</li>
  <li><a href="https://github.com/WangRongsheng/ChatGenTitle">Arxiv GenTitle</a></li>
  <li><a href="https://github.com/LiuHC0428/LAW-GPT">Chinese Law</a></li>
  <li><a href="https://huggingface.co/datasets/Open-Orca/OpenOrca">OpenOrca</a></li>
  <li><a href="https://huggingface.co/datasets/shibing624/medical">Medical Dialogue</a></li>
  <li>...</li>
</ul>
</td>
<td>
<ul>
  <li><a href="docs/zh_cn/user_guides/incremental_pretraining.md">Incremental Pre-training</a> </li>
  <li><a href="docs/zh_cn/user_guides/single_turn_conversation.md">Single-turn Conversation SFT</a> </li>
  <li><a href="docs/zh_cn/user_guides/multi_turn_conversation.md">Multi-turn Conversation SFT</a> </li>
</ul>
</td>
<td>
<ul>
  <li><a href="http://arxiv.org/abs/2305.14314">QLoRA</a></li>
  <li><a href="http://arxiv.org/abs/2106.09685">LoRA</a></li>
  <li>Full parameter fine-tune</li>
  <li><a href="https://arxiv.org/abs/2305.18290">DPO</a></li>
  <li><a href="https://arxiv.org/abs/2403.07691">ORPO</a></li>
  <li>Reward Model</a></li>
</ul>
</td>
</tr>
</tbody>
</table>

## 🛠️ Quick Start

### Installation

- It is recommended to build a Python-3.10 virtual environment using conda

  ```bash
  conda create --name xtuner-env python=3.10 -y
  conda activate xtuner-env
  ```

- Install XTuner via pip

  ```shell
  pip install -U xtuner
  ```

  or with DeepSpeed integration

  ```shell
  pip install -U 'xtuner[deepspeed]'
  ```

- Install XTuner from source

  ```shell
  git clone https://github.com/InternLM/xtuner.git
  cd xtuner
  pip install -e '.[all]'
  ```

### Fine-tune

XTuner supports the efficient fine-tune (*e.g.*, QLoRA) for LLMs. Dataset prepare guides can be found on [dataset_prepare.md](./docs/en/user_guides/dataset_prepare.md).

- **Step 0**, prepare the config. XTuner provides many ready-to-use configs and we can view all configs by

  ```shell
  xtuner list-cfg
  ```

  Or, if the provided configs cannot meet the requirements, please copy the provided config to the specified directory and make specific modifications by

  ```shell
  xtuner copy-cfg ${CONFIG_NAME} ${SAVE_PATH}
  vi ${SAVE_PATH}/${CONFIG_NAME}_copy.py
  ```

- **Step 1**, start fine-tuning.

  ```shell
  xtuner train ${CONFIG_NAME_OR_PATH}
  ```

  For example, we can start the QLoRA fine-tuning of InternLM2.5-Chat-7B with oasst1 dataset by

  ```shell
  # On a single GPU
  xtuner train internlm2_5_chat_7b_qlora_oasst1_e3 --deepspeed deepspeed_zero2
  # On multiple GPUs
  (DIST) NPROC_PER_NODE=${GPU_NUM} xtuner train internlm2_5_chat_7b_qlora_oasst1_e3 --deepspeed deepspeed_zero2
  (SLURM) srun ${SRUN_ARGS} xtuner train internlm2_5_chat_7b_qlora_oasst1_e3 --launcher slurm --deepspeed deepspeed_zero2
  ```

  - `--deepspeed` means using [DeepSpeed](https://github.com/microsoft/DeepSpeed) 🚀 to optimize the training. XTuner comes with several integrated strategies including ZeRO-1, ZeRO-2, and ZeRO-3. If you wish to disable this feature, simply remove this argument.

  - For more examples, please see [finetune.md](./docs/en/user_guides/finetune.md).

- **Step 2**, convert the saved PTH model (if using DeepSpeed, it will be a directory) to Hugging Face model, by

  ```shell
  xtuner convert pth_to_hf ${CONFIG_NAME_OR_PATH} ${PTH} ${SAVE_PATH}
  ```

### Chat

XTuner provides tools to chat with pretrained / fine-tuned LLMs.

```shell
xtuner chat ${NAME_OR_PATH_TO_LLM} --adapter {NAME_OR_PATH_TO_ADAPTER} [optional arguments]
```

For example, we can start the chat with InternLM2.5-Chat-7B :

```shell
xtuner chat internlm/internlm2_5-chat-7b --prompt-template internlm2_chat
```

For more examples, please see [chat.md](./docs/en/user_guides/chat.md).

### Deployment

- **Step 0**, merge the Hugging Face adapter to pretrained LLM, by

  ```shell
  xtuner convert merge \
      ${NAME_OR_PATH_TO_LLM} \
      ${NAME_OR_PATH_TO_ADAPTER} \
      ${SAVE_PATH} \
      --max-shard-size 2GB
  ```

- **Step 1**, deploy fine-tuned LLM with any other framework, such as [LMDeploy](https://github.com/InternLM/lmdeploy) 🚀.

  ```shell
  pip install lmdeploy
  python -m lmdeploy.pytorch.chat ${NAME_OR_PATH_TO_LLM} \
      --max_new_tokens 256 \
      --temperture 0.8 \
      --top_p 0.95 \
      --seed 0
  ```

  🔥 Seeking efficient inference with less GPU memory? Try 4-bit quantization from [LMDeploy](https://github.com/InternLM/lmdeploy)! For more details, see [here](https://github.com/InternLM/lmdeploy/tree/main#quantization).

### Evaluation

- We recommend using [OpenCompass](https://github.com/InternLM/opencompass), a comprehensive and systematic LLM evaluation library, which currently supports 50+ datasets with about 300,000 questions.

## 🤝 Contributing

We appreciate all contributions to XTuner. Please refer to [CONTRIBUTING.md](.github/CONTRIBUTING.md) for the contributing guideline.

## 🎖️ Acknowledgement

- [Llama 2](https://github.com/facebookresearch/llama)
- [DeepSpeed](https://github.com/microsoft/DeepSpeed)
- [QLoRA](https://github.com/artidoro/qlora)
- [LMDeploy](https://github.com/InternLM/lmdeploy)
- [LLaVA](https://github.com/haotian-liu/LLaVA)

## 🖊️ Citation

```bibtex
@misc{2023xtuner,
    title={XTuner: A Toolkit for Efficiently Fine-tuning LLM},
    author={XTuner Contributors},
    howpublished = {\url{https://github.com/InternLM/xtuner}},
    year={2023}
}
```

## License

This project is released under the [Apache License 2.0](LICENSE). Please also adhere to the Licenses of models and datasets being used.
