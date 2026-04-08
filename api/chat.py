import json
import os
import math
import time
from http.server import BaseHTTPRequestHandler
from anthropic import Anthropic

MODELS = ["claude-sonnet-4-6", "claude-haiku-4-5-20251001"]
MAX_RETRIES = 3
RETRY_DELAY = 2

SYSTEM_PROMPT = """You are a highly capable, helpful, honest, and harmless AI assistant called "My AI Agent".
ALWAYS respond in the SAME language the user writes in.
If user writes Hindi reply in Hindi, Spanish reply in Spanish, etc.
Support ALL languages worldwide.
Be warm, friendly, respectful. Use markdown formatting.
You have a calculator tool and a get_datetime tool."""

TOOLS = [
    {"name": "calculator", "description": "Evaluate a mathematical expression.", "input_schema": {"type": "object", "properties": {"expression": {"type": "string", "description": "Math expression (Python syntax)"}}, "required": ["expression"]}},
    {"name": "get_datetime", "description": "Get current date, time, day of week.", "input_schema": {"type": "object", "properties": {"format": {"type": "string", "description": "full/date/time/day", "default": "full"}}, "required": []}}
]

def execute_tool(name, inp):
    if name == "calculator":
        allowed = {'abs':abs,'round':round,'min':min,'max':max,'pow':pow,'int':int,'float':float,'sin':math.sin,'cos':math.cos,'tan':math.tan,'sqrt':math.sqrt,'log':math.log,'log10':math.log10,'pi':math.pi,'e':math.e,'ceil':math.ceil,'floor':math.floor,'factorial':math.factorial}
        try:
            return f"{inp['expression']} = {eval(inp['expression'], {'__builtins__': {}}, allowed)}"
        except Exception as e:
            return f"Math error: {e}"
    if name == "get_datetime":
        from datetime import datetime
        now = datetime.now()
        fmt = inp.get("format", "full")
        if fmt == "date": return now.strftime("%Y-%m-%d")
        elif fmt == "time": return now.strftime("%H:%M:%S")
        elif fmt == "day": return now.strftime("%A")
        return now.strftime("%A, %B %d, %Y at %H:%M:%S")
    return "Unknown tool"

class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            self._json(500, {"error": "API key not configured"})
            return
        messages = body.get("messages", [])
        client = Anthropic(api_key=api_key)

        def call_retry(msgs):
            for model in MODELS:
                for attempt in range(MAX_RETRIES):
                    try:
                        return client.messages.create(model=model, max_tokens=4096, system=SYSTEM_PROMPT, tools=TOOLS, messages=msgs), model
                    except Exception as e:
                        es = str(e).lower()
                        if "overloaded" in es or "529" in es or "rate_limit" in es or "429" in es:
                            time.sleep(RETRY_DELAY * (attempt + 1))
                            continue
                        raise e
            raise Exception("All models busy. Try again in a few minutes.")

        try:
            all_text, tool_calls, used_model = [], [], None
            for _ in range(5):
                response, used_model = call_retry(messages)
                serialized = []
                for b in response.content:
                    if b.type == "text": serialized.append({"type":"text","text":b.text})
                    elif b.type == "tool_use": serialized.append({"type":"tool_use","id":b.id,"name":b.name,"input":b.input})
                messages.append({"role":"assistant","content":serialized})
                tool_uses = [b for b in response.content if b.type == "tool_use"]
                for b in response.content:
                    if hasattr(b,"text") and b.text: all_text.append(b.text)
                if not tool_uses: break
                tr = []
                for tu in tool_uses:
                    r = execute_tool(tu.name, tu.input)
                    tool_calls.append({"name":tu.name,"input":tu.input,"result":r})
                    tr.append({"type":"tool_result","tool_use_id":tu.id,"content":r})
                messages.append({"role":"user","content":tr})
                if response.stop_reason == "end_turn": break
            self._json(200, {"text":"\n".join(all_text),"tools_used":tool_calls,"model_used":used_model or "unknown","messages":messages})
        except Exception as e:
            self._json(500, {"error": str(e)})

    def do_OPTIONS(self):
        self.send_response(200)
        self._cors()
        self.end_headers()

    def _json(self, code, data):
        self.send_response(code)
        self.send_header("Content-Type","application/json")
        self._cors()
        self.end_headers()
        self.wfile.write(json.dumps(data, ensure_ascii=False).encode())

    def _cors(self):
        self.send_header("Access-Control-Allow-Origin","*")
        self.send_header("Access-Control-Allow-Methods","POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers","Content-Type")
