from langchain_community.llms import HuggingFacePipeline
from langchain_community.embeddings import HuggingFaceEmbeddings
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA

################################################
# Set embeddings models
################################################
EMBEDDINGS = HuggingFaceEmbeddings(model_name='intfloat/multilingual-e5-large')

################################################
# Set vector db
################################################
VECTOR_DB = Chroma(persist_directory="./chroma_db/", embedding_function=EMBEDDINGS)

################################################
# Set the model_id and set prompt.
################################################
model_id = "elyza/ELYZA-japanese-Llama-2-7b-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
DEFAULT_SYSTEM_PROMPT = "あなたは日本人のプロのソフトウェアエンジニアです。ソフトウェアエンジニア目線で答えでください"
# DEFAULT_SYSTEM_PROMPT = ""
text = """下記のドキュメントを参照して、日本語で質問の回答のみ返してください。答えがわからない場合は、わからないと言ってください。

ドキュメント: 
{context}

質問:
{question}
"""
prompt_template = "{bos_token}{b_inst} {system}{prompt} {e_inst} ".format(
    bos_token=tokenizer.bos_token,
    b_inst=B_INST,
    system=f"{B_SYS}{DEFAULT_SYSTEM_PROMPT}{E_SYS}",
    prompt=text,
    e_inst=E_INST,
)
PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context","question"])


################################################
# Set the model
# The model will be downloaded on the first run and cached in # modle cache folder "~/.cache/huggingface/hub" (under wsl shell)
################################################
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto", quantization_config=bnb_config)
generator = pipeline("text-generation", model=model, tokenizer=tokenizer, max_length=3000)

LLM = HuggingFacePipeline(pipeline=generator)
print("llm loaded!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

def run_llm(query: str) -> any:
    torch.cuda.empty_cache()

    # for RetrievalQA
    qa = RetrievalQA.from_chain_type(
        llm=LLM,
        chain_type="stuff",
        retriever=VECTOR_DB.as_retriever(),
        return_source_documents=True,
        chain_type_kwargs={"prompt":PROMPT},
        verbose=True
    )
    result = qa.invoke({"query": query})
    print(result)

    return result

if __name__ == "__main__":
    print(run_llm(query="Payjpはどんな言語やプラットフォームのライブラリをサポートしていますか"))
