"""按平台Qwen实际XML工具模板路由；冻结本地基底，无外部API。"""
import json,re,time,uuid

def parse_tool_calls(text):
    blocks=re.findall(r'<tool_call>\s*(.*?)\s*</tool_call>',text,re.S)
    if '<tool_call>' in text and not blocks:raise ValueError('工具调用标签不完整')
    calls=[]
    for block in blocks:
        match=re.fullmatch(r'<function=([A-Za-z_][A-Za-z_0-9]*)>\s*(.*?)\s*</function>',block,re.S)
        if not match:raise ValueError('工具函数格式无效')
        body=match.group(2);args={};position=0
        for parameter in re.finditer(r'<parameter=([A-Za-z_][A-Za-z_0-9]*)>\s*(.*?)\s*</parameter>',body,re.S):
            if body[position:parameter.start()].strip():raise ValueError('工具参数含未解析文本')
            key,value=parameter.group(1),parameter.group(2)
            if key in args:raise ValueError('重复工具参数')
            # 所有其他参数按文本保留，避免把数字形样本ID或检索词擅自转型。
            if key in ['top_k','limit']:
                if not re.fullmatch(r'\d+',value):raise ValueError('工具整数参数无效')
                value=int(value)
            args[key]=value;position=parameter.end()
        if body[position:].strip():raise ValueError('工具参数尾部未解析')
        calls.append({'id':uuid.uuid4().hex,'type':'function','function':{'name':match.group(1),'arguments':args}})
    return calls

class QwenToolRouter:
    def __init__(self,gateway,tools,*,max_calls=32,max_context_tokens=24000):
        self.gateway=gateway;self.tools=tools;self.max_calls=max_calls;self.max_context_tokens=max_context_tokens
        self.model=None;self.tokenizer=None
    def load_frozen(self):
        import torch
        from transformers import AutoTokenizer,Qwen3_5ForConditionalGeneration
        if torch.cuda.mem_get_info()[0]<64*1024**3:raise RuntimeError('等待至少64GiB空闲PPU后加载冻结Qwen，不抢占训练')
        self.tokenizer=AutoTokenizer.from_pretrained('/model/Qwen3.6-27B',local_files_only=True)
        self.model=Qwen3_5ForConditionalGeneration.from_pretrained('/model/Qwen3.6-27B',local_files_only=True,dtype=torch.bfloat16,attn_implementation='sdpa').to('cuda').eval()
        self.model.requires_grad_(False)
    def run(self,messages):
        import torch
        if self.model is None:raise RuntimeError('冻结Qwen尚未加载')
        history=list(messages);events=[];count=0;start=time.monotonic()
        while True:
            prompt=self.tokenizer.apply_chat_template(history,tools=self.tools,tokenize=False,add_generation_prompt=True,enable_thinking=False)
            encoded=self.tokenizer(prompt,return_tensors='pt').to('cuda')
            if encoded.input_ids.shape[1]>self.max_context_tokens:return {'status':'context_budget_reached','events':events,'final':None}
            with torch.inference_mode():
                tokens=self.model.generate(**encoded,max_new_tokens=2048,do_sample=False,temperature=None,top_p=None,top_k=None)
            text=self.tokenizer.decode(tokens[0,encoded.input_ids.shape[1]:],skip_special_tokens=True)
            del tokens,encoded
            try:calls=parse_tool_calls(text)
            except ValueError as e:return {'status':'invalid_tool_syntax','error':str(e),'events':events,'final':None,'raw_text':text}
            if not calls:
                try:final=json.loads(text.strip().removeprefix('```json').removesuffix('```').strip())
                except json.JSONDecodeError:return {'status':'invalid_final_json','events':events,'final':None,'raw_text':text}
                return {'status':'draft_requires_output_validation','final':final,'events':events,'calls':count,'seconds':time.monotonic()-start}
            if count+len(calls)>self.max_calls:return {'status':'call_budget_reached','events':events,'final':None}
            history.append({'role':'assistant','content':'','tool_calls':calls})
            for call in calls:
                begin=time.monotonic();function=call['function'];count+=1
                try:result=self.gateway.dispatch(function['name'],function['arguments'])
                except (ValueError,FileNotFoundError) as e:result={'status':'tool_input_error','error':str(e)}
                events.append({'name':function['name'],'arguments':function['arguments'],'status':result['status'],'seconds':time.monotonic()-begin})
                history.append({'role':'tool','tool_call_id':call['id'],'name':function['name'],'content':json.dumps(result,ensure_ascii=False)})
