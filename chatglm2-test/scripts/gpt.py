import os
import openai
import re
import sys

file_path = "patent_without_abstract_clean/{}.txt".format(sys.argv[1])
with open(file_path) as f:
        text = f.read()
pattern = "^\d*\[\d+\]"
text = re.sub(pattern, "", text)

openai.api_key = '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'
prompt = "{}\n\n提取5个词组或者短语作为关键词并且以带序号的列表格式输出: \n".format(text)
print(prompt)
completion = openai.ChatCompletion.create(
  # model="gpt-3.5-turbo",
  temperature=0.1,
  frequency_penalty=1.1,
  model="gpt-3.5-turbo-16k-0613",
  messages=[
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": prompt}
  ]
)

print(completion.choices[0].message['content'])
