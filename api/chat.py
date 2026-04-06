import json
import os
import math
from http.server import BaseHTTPRequestHandler
from anthropic import Anthropic

SYSTEM_PROMPT = """You are a highly capable, helpful, honest, and harmless AI assistant called "My AI Agent".
ALWAYS respond in the SAME language the user writes in.
If user writes Hindi reply in Hindi, Spanish reply in Spanish, etc.
Support ALL languages worldwide.
Be warm, friendly, respectful. Use markdown formatting.
You have a calculator tool and a get_datetime tool."""

TOOLS = [
    {
        "name": "calculator",
        "description": "Evaluate a mathematical expression.",
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "Math expression (Python syntax)"}
            },
            "required": ["expression"]
        }
    },
    {
        "name": "get_datetime",
        "description": "Get current date, time, day of week.",
        "input_schema": {
            "type": "object",
            "properties": {
                "format": {"type": "string", "description": "full, date, time, or day", "default": "full"}
            },
            "required": []
        }
    }
]

def execute_tool(name, tool_input):
    if name == "calculator":
        allowed = {
            'abs': abs, 'round': round, 'min': min, 'max': max,
            'pow': pow, 'int': int, 'float': float,
            'sin': math.sin, 'cos': math.cos, 'tan': math.tan,
            'sqrt': math.sqrt, 'log': math.log, 'log10': math.log10,
            'pi': math.pi, 'e': math.e, 'ceil': math.ceil, 'floor': math.floor,
            'factorial': math.factorial,
        }
        try:
            result = eval(tool_input["expression"], {"__builtins__": {}}, allowed)
            return str(tool_input["expression"]) + " = " + str(result)
        except Exception as e:
            return "Math error: " + str(e)
    if name == "get_datetime":
        from datetime import datetime
        now = datetime.now()
        fmt = tool_input.get("format", "full")
        if fmt == "date": return now.strftime("%Y-%m-%d")
        elif fmt == "time": return now.strftime("%H:%M:%S")
        elif fmt == "day": return now.strftime("%A")
        return now.strftime("%A, %B %d, %Y at %H:%M:%S")
    return "Unknown tool"

class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(content_length))
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            self._send_json(500, {"error": "API key not configured on server"})
            return
        messages = body.get("messages", [])
        model = body.get("model", "claude-sonnet-4-6")
        client = Anthropic(api_key=api_key)
        try:
            max_iterations = 5
            all_text = []
            tool_calls = []
            for _ in range(max_iterations):
                response = client.messages.create(
                    model=model, max_tokens=4096,
                    system=SYSTEM_PROMPT, tools=TOOLS, messages=messages,
                )
                assistant_content = response.content
                serialized = []
                for b in assistant_content:
                    if b.type == "text":
                        serialized.append({"type": "text", "text": b.text})
                    elif b.type == "tool_use":
                        serialized.append({"type": "tool_use", "id": b.id, "name": b.name, "input": b.input})
                messages.append({"role": "assistant", "content": serialized})
                tool_uses = [b for b in assistant_content if b.type == "tool_use"]
                for b in assistant_content:
                    if hasattr(b, "text") and b.text:
                        all_text.append(b.text)
                if not tool_uses:
                    break
                tool_results = []
                for tu in tool_uses:
                    result = execute_tool(tu.name, tu.input)
                    tool_calls.append({"name": tu.name, "input": tu.input, "result": result})
                    tool_results.append({"type": "tool_result", "tool_use_id": tu.id, "content": result})
                messages.append({"role": "user", "content": tool_results})
                if response.stop_reason == "end_turn":
                    break
            self._send_json(200, {"text": "\n".join(all_text), "tools_used": tool_calls, "messages": messages})
        except Exception as e:
            self._send_json(500, {"error": str(e)})

    def do_OPTIONS(self):
        self.send_response(200)
        self._cors_headers()
        self.end_headers()

    def _send_json(self, code, data):
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self._cors_headers()
        self.end_headers()
        self.wfile.write(json.dumps(data, ensure_ascii=False).encode())

    def _cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
