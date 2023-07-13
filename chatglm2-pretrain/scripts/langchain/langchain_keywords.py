import argparse
import os
parser = argparse.ArgumentParser()
parser.add_argument('--file_path',required=True,type=str)
parser.add_argument('--model_path',required=True,type=str)
parser.add_argument('--gpus', default="0", type=str)
parser.add_argument('--chain_type', default="stuff", type=str)
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
# os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION']='python'
file_path = args.file_path
model_path = args.model_path

import torch
from langchain import HuggingFacePipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter, SpacyTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain

#prompt_template = "{text}\n\n请为以上内容写一段摘要：\n"
#关键字
#prompt_template = "请从以下内容中提取五个词组或者短语作为关键词：\n\n{text}\n\n### 关键字："
#prompt_template = "{text}\n\n提取5个词组或者短语作为关键词并以列表形式输出：\n"
prompt_template = "{text}\n\n提取5个词组或者短语作为关键词并且以带序号的列表格式输出: \n"  # use
#prompt_template = "问：{text}\n提取5个词组或者短语作为关键词并且以带序号的列表格式输出。\n答："
#prompt_template = "{text}\n\n提取5个关键词, 字数必须大于1，并且输出带序号的列表格式"


refine_template = (
    "Below is an instruction that describes a task."
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n"
    "已有一段摘要：{existing_answer}\n"
    "现在还有一些文字，（如果有需要）你可以根据它们完善现有的摘要。"
    "\n"
    "{text}\n"
    "\n"
    "如果这段文字没有用，返回原来的摘要即可。请你生成一个最终的摘要。"
    "\n\n### Response: "
)


if __name__ == '__main__':
    load_type = torch.float16
    
    if torch.cuda.is_available():
        device = torch.device(0)
    else:
        device = torch.device('cpu')
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0, length_function=len)
    # text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=0)
    #text_splitter = SpacyTextSplitter(pipeline='zh_core_web_sm',chunk_size=1000,chunk_overlap=100)
    with open(file_path) as f:
        text = f.read()
    pattern = "^\d*\[\d+\]"
    import re
    text = re.sub(pattern, "", text)
    docs = text_splitter.create_documents([text])
    #print(docs)
    print("loading LLM...")
    model = HuggingFacePipeline.from_model_id(model_id=model_path,
            task="summarization",
            model_kwargs={
                          "torch_dtype" : load_type,
                          "low_cpu_mem_usage" : True,
                          "trust_remote_code": True,
                          "temperature": 0.2,
                          "max_length": 32768,
                          "device_map": "auto",
                          "repetition_penalty":1.1}
            )

    PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"])
    REFINE_PROMPT = PromptTemplate(
        template=refine_template,input_variables=["existing_answer", "text"],
    )

    if args.chain_type == "stuff":
        chain = load_summarize_chain(model, chain_type="stuff", prompt=PROMPT)
    elif args.chain_type == "refine":
        chain = load_summarize_chain(model, chain_type="refine", question_prompt=PROMPT, refine_prompt=REFINE_PROMPT)
    print(chain.run(docs))
