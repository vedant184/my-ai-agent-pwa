# My AI Agent - Chat API v2.0
import json, os, math, time, io, sys
from http.server import BaseHTTPRequestHandler
from anthropic import Anthropic
from urllib.request import urlopen, Request
from urllib.parse import quote_plus

MODELS = ["claude-sonnet-4-6", "claude-haiku-4-5-20251001"]
MAX_RETRIES = 3
RETRY_DELAY = 2

SYSTEM = """You are My AI Agent - a powerful, helpful AI assistant.
ALWAYS respond in the SAME language the user writes in.
You have these tools: calculator, get_datetime, run_python, wikipedia_search, generate_file.
Use tools proactively. Be warm, friendly. Use markdown formatting."""

TOOLS = [
    {"name":"calculator","description":"Evaluate math expressions (Python syntax). Use for any calculations.","input_schema":{"type":"object","properties":{"expression":{"type":"string","description":"Math expression"}},"required":["expression"]}},
    {"name":"get_datetime","description":"Get current date, time, day of week.","input_schema":{"type":"object","properties":{"format":{"type":"string","description":"full/date/time/day","default":"full"}},"required":[]}},
    {"name":"run_python","description":"Execute Python code and return output. Use for data processing, algorithms, text manipulation, lists, etc.","input_schema":{"type":"object","properties":{"code":{"type":"string","description":"Python code to execute"}},"required":["code"]}},
    {"name":"wikipedia_search","description":"Search Wikipedia for information on any topic. Returns summary.","input_schema":{"type":"object","properties":{"query":{"type":"string","description":"Search query"}},"required":["query"]}},
    {"name":"generate_file","description":"Generate a downloadable file (code, text, html, csv, json, etc).","input_schema":{"type":"object","properties":{"filename":{"type":"string","description":"Filename with extension (e.g. script.py, data.csv)"},"content":{"type":"string","description":"File content"}},"required":["filename","content"]}}
]

def execute_tool(name, inp):
    if name == "calculator":
        safe = {'abs':abs,'round':round,'min':min,'max':max,'pow':pow,'int':int,'float':float,'sin':math.sin,'cos':math.cos,'tan':math.tan,'sqrt':math.sqrt,'log':math.log,'log10':math.log10,'pi':math.pi,'e':math.e,'ceil':math.ceil,'floor':math.floor,'factorial':math.factorial}
        try: return f"{inp['expression']} = {eval(inp['expression'],{'__builtins__':{}},safe)}"
        except Exception as e: return f"Error: {e}"
    if name == "get_datetime":
        from datetime import datetime
        now = datetime.now()
        fmt = inp.get("format","full")
        if fmt=="date": return now.strftime("%Y-%m-%d")
        elif fmt=="time": return now.strftime("%H:%M:%S")
        elif fmt=="day": return now.strftime("%A")
        return now.strftime("%A, %B %d, %Y at %H:%M:%S")
    if name == "run_python":
        old = sys.stdout
        sys.stdout = io.StringIO()
        try:
            exec(inp["code"], {"__builtins__":{"print":print,"range":range,"len":len,"str":str,"int":int,"float":float,"list":list,"dict":dict,"tuple":tuple,"set":set,"sorted":sorted,"enumerate":enumerate,"zip":zip,"map":map,"filter":filter,"sum":sum,"min":min,"max":max,"abs":abs,"round":round,"type":type,"isinstance":isinstance,"bool":bool,"True":True,"False":False,"None":None,"math":math}})
            output = sys.stdout.getvalue()
            return output if output else "Code executed successfully (no output)"
        except Exception as e: return f"Error: {e}"
        finally: sys.stdout = old
    if name == "wikipedia_search":
        try:
            url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{quote_plus(inp['query'])}"
            req = Request(url, headers={"User-Agent":"MyAIAgent/1.0"})
            data = json.loads(urlopen(req, timeout=5).read())
            return f"**{data.get('title','')}**\n{data.get('extract','No info found.')}"
        except: return "Could not find Wikipedia article. Try different keywords."
    if name == "generate_file":
        return json.dumps({"type":"file","filename":inp["filename"],"content":inp["content"]})
    return "Unknown tool"

class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length",0))))
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            self._json(500,{"error":"API key not configured"})
            return
        messages = body.get("messages",[])
        client = Anthropic(api_key=api_key)
        def call_retry(msgs):
            for model in MODELS:
                for attempt in range(MAX_RETRIES):
                    try: return client.messages.create(model=model,max_tokens=4096,system=SYSTEM,tools=TOOLS,messages=msgs),model
                    except Exception as e:
                        es=str(e).lower()
                        if "overloaded" in es or "529" in es or "rate_limit" in es or "429" in es:
                            time.sleep(RETRY_DELAY*(attempt+1)); continue
                        raise e
            raise Exception("All models busy. Try again soon.")
        try:
            all_text,tool_calls,files,used_model=[],[],[],None
            for _ in range(5):
                response,used_model=call_retry(messages)
                ser=[]
                for b in response.content:
                    if b.type=="text": ser.append({"type":"text","text":b.text})
                    elif b.type=="tool_use": ser.append({"type":"tool_use","id":b.id,"name":b.name,"input":b.input})
                messages.append({"role":"assistant","content":ser})
                tus=[b for b in response.content if b.type=="tool_use"]
                for b in response.content:
                    if hasattr(b,"text") and b.text: all_text.append(b.text)
                if not tus: break
                tr=[]
                for tu in tus:
                    r=execute_tool(tu.name,tu.input)
                    tool_calls.append({"name":tu.name,"input":tu.input,"result":r})
                    if tu.name=="generate_file":
                        try: files.append(json.loads(r))
                        except: pass
                    tr.append({"type":"tool_result","tool_use_id":tu.id,"content":r})
                messages.append({"role":"user","content":tr})
                if response.stop_reason=="end_turn": break
            self._json(200,{"text":"\n".join(all_text),"tools_used":tool_calls,"files":files,"model_used":used_model or "unknown","messages":messages})
        except Exception as e:
            self._json(500,{"error":str(e)})
    def do_OPTIONS(self):
        self.send_response(200); self._cors(); self.end_headers()
    def _json(self,code,data):
        self.send_response(code)
        self.send_header("Content-Type","application/json")
        self._cors(); self.end_headers()
        self.wfile.write(json.dumps(data,ensure_ascii=False).encode())
    def _cors(self):
        self.send_header("Access-Control-Allow-Origin","*")
        self.send_header("Access-Control-Allow-Methods","POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers","Content-Type")
