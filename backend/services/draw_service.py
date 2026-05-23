import io
import re
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from backend.services.tracker import track_api_call
from backend.services.ai_service import gemini_flash

def generate_graph_base64(graph_code: str, figsize=(6, 4), dpi=150) -> str:
    """Executes the generated Matplotlib code inside a sandbox environment
    and returns a base64 encoded PNG data URI.
    """
    try:
        plt.figure(figsize=figsize)
        safe_globals = {
            "__builtins__": {}, "plt": plt, "np": np,
            "range": range, "len": len, "zip": zip,
            "list": list, "tuple": tuple, "int": int, "float": float,
            "str": str, "abs": abs, "min": min, "max": max,
            "sum": sum, "round": round, "enumerate": enumerate,
            "math": __import__('math'),
        }
        # Run in sandboxed exec
        exec(graph_code, safe_globals)
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=dpi)
        buf.seek(0)
        encoded = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
        return f"data:image/png;base64,{encoded}"
    except Exception as e:
        print(f"Sandbox Matplotlib execution failed: {e}")
        plt.close()
        return None

def execute_matplotlib_diagram(prompt: str, subject: str = "General") -> tuple:
    """Invokes Gemini flash to compose the Matplotlib code for the educational diagram,
    runs it inside the safe sandbox, and returns (description, base64_image_url).
    """
    img_prompt = f"""Create a clear educational diagram for: "{prompt}"
Subject: {subject}
Provide matplotlib code between [GRAPH_START] and [GRAPH_END] tags.
Do NOT use plt.savefig()."""
    try:
        response = gemini_flash.generate_content(img_prompt)
        track_api_call("gemini-2.5-flash-lite", "image_generate")
        answer = response.text
        
        graph_url = None
        graph_match = re.search(r'\[GRAPH_START\](.*?)\[GRAPH_END\]', answer, re.DOTALL)
        description = re.sub(r'\[GRAPH_START\].*?\[GRAPH_END\]', '', answer, flags=re.DOTALL).strip()
        
        if graph_match:
            graph_code = graph_match.group(1).strip()
            graph_url = generate_graph_base64(graph_code, figsize=(8, 6), dpi=150)
            
        return description[:500], graph_url
    except Exception as e:
        print(f"execute_matplotlib_diagram error: {e}")
        return str(e), None
