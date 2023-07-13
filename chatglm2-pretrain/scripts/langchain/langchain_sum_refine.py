import argparse
import os
parser = argparse.ArgumentParser()
parser.add_argument('--file_path',required=True,type=str)
parser.add_argument('--model_path',required=True,type=str)
parser.add_argument('--gpus', default="0", type=str)
parser.add_argument('--chain_type', default="refine", type=str)
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
# os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION']='python'
file_path = args.file_path
model_path = args.model_path

import torch
from langchain import HuggingFacePipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter, SpacyTextSplitter
from langchain.document_loaders import UnstructuredFileLoader
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain

#摘要
'''
prompt_template = ("Below is an instruction that describes a task. "
                   "Write a response that appropriately completes the request.\n\n"
                   "### Instruction:\n请为以下文字写一段摘要:\n{text}\n\n### Response: ")
'''
prompt_template = ("下面是一段关于任务要求的说明， "
                   "请根据任务要求做出响应的回答。\n\n"
                   "### 任务:\n请为以下文字写一段摘要:\n{text}\n\n### 摘要: ")

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

    #text_splitter = RecursiveCharacterTextSplitter(chunk_size=20000, chunk_overlap=200, length_function=len)
    # text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=0)
    text_splitter = SpacyTextSplitter(pipeline='zh_core_web_sm',chunk_size=2000,chunk_overlap=200)
    '''
    with open(file_path) as f:
        text = f.read()
    pattern = "^\d*\[\d+\]"
    import re
    text = re.sub(pattern, "", text)
    docs = text_splitter.create_documents([text])
    '''
    # 导入文本
    loader = UnstructuredFileLoader(file_path)
    # 将文本转成 Document 对象
    document = loader.load()
    docs = text_splitter.split_documents(document)

    print("loading LLM...")
    model = HuggingFacePipeline.from_model_id(model_id=model_path,
            task="summarization",
            model_kwargs={
                          "torch_dtype" : load_type,
                          "low_cpu_mem_usage" : True,
                          "trust_remote_code": True,
                          "temperature": 0.2,
                          "max_length": 16384,
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
        chain = load_summarize_chain(model, chain_type="refine", refine_prompt=PROMPT)
    print(chain.run(docs))
