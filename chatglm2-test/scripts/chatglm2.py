from transformers import AutoTokenizer, AutoModel
import sys
import re

tokenizer = AutoTokenizer.from_pretrained("/data9/NFS/patent/model_hub/chatglm2-6b/", trust_remote_code=True)
model = AutoModel.from_pretrained("/data9/NFS/patent/model_hub/chatglm2-6b/", trust_remote_code=True, device='cuda')
model = model.eval()

file_path = "patent_without_abstract_clean/{}.txt".format(sys.argv[1])
with open(file_path) as f:
        text = f.read()
pattern = "^\d*\[\d+\]"
text = re.sub(pattern, "", text)

prompt = "{}\n\n提取5个短语作为关键词并且以带序号的列表格式输出: \n".format(text)
response, _ = model.chat(tokenizer, prompt, history=[])
print(response)
