import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aca.config import OPENROUTER_MODELS, get_settings

load_dotenv()

# ============================================================
# CONFIGURATION
# ============================================================

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
MODEL_ID = get_settings().default_openrouter_model
# Available shared registry entries:
# OPENROUTER_MODELS["minimax_m2_7"]
# OPENROUTER_MODELS["kimi_k2_5"]


# ============================================================
# TOOL DEFINITIONS
# ============================================================

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate mathematical expressions. Supports +, -, *, /, ^, (), %.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "The math expression to evaluate, e.g., '25 * 4 + 10'"
                    }
                },
                "required": ["expression"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "text_analyzer",
            "description": "Analyze text and return statistics like word count, character count, and case conversions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The text to analyze"
                    }
                },
                "required": ["text"]
            }
        }
    }
]


# ============================================================
# TOOL IMPLEMENTATIONS
# ============================================================

def calculator(expression: str) -> str:
    """Safely evaluate a mathematical expression."""
    # Whitelist only safe characters
    allowed_chars = set("0123456789+-*/().^% ")
    if not all(c in allowed_chars for c in expression):
        return json.dumps({"error": "Invalid characters in expression"})
    
    try:
        # Safe eval with no builtins
        result = eval(expression, {"__builtins__": {}}, {"^": "**"})
        return json.dumps({"expression": expression, "result": result})
    except Exception as e:
        return json.dumps({"error": f"Evaluation error: {str(e)}"})


def text_analyzer(text: str) -> str:
    """Analyze text and return statistics."""
    words = text.split()
    return json.dumps({
        "original": text,
        "word_count": len(words),
        "char_count": len(text),
        "char_count_no_spaces": len(text.replace(" ", "")),
        "uppercase": text.upper(),
        "lowercase": text.lower(),
        "words_list": words
    })


TOOL_REGISTRY = {
    "calculator": calculator,
    "text_analyzer": text_analyzer,
}


# ============================================================
# AGENT IMPLEMENTATION - OpenAI SDK
# ============================================================

def create_openai_client() -> OpenAI:
    """Create OpenAI client configured for OpenRouter."""
    if not OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is not set in the environment.")

    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
        default_headers={
            "HTTP-Referer": "http://localhost:8000",  # Optional: for rankings
            "X-Title": "GLM5-Thinking-Agent"          # Optional: for rankings
        }
    )


def run_agent_openai_sdk(
    query: str,
    model: str = MODEL_ID,
    max_iterations: int = 10,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Run agent with interleaved thinking using OpenAI SDK.
    
    The thinking/reasoning tokens appear in separate fields in the response,
    allowing true interleaved thinking between tool calls.
    """
    client = create_openai_client()
    
    messages: List[Dict] = [{"role": "user", "content": query}]
    reasoning_trace: List[str] = []
    tool_calls_log: List[Dict] = []
    
    def log(msg: str) -> None:
        if verbose:
            print(msg)
    
    for iteration in range(1, max_iterations + 1):
        log(f"\n{'='*70}")
        log(f"ITERATION {iteration}")
        log(f"{'='*70}")
        
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
                # Extra body for OpenRouter-specific features
                extra_body={
                    "include_reasoning": True
                }
            )
        except Exception as e:
            log(f"API Error: {e}")
            # Fallback without reasoning params
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto"
            )
        
        choice = response.choices[0]
        message = choice.message
        
        # ---- EXTRACT REASONING/THINKING ----
        # OpenRouter returns reasoning in different ways depending on model
        thinking_text = None
        
        # Method 1: message.reasoning (some models)
        if hasattr(message, 'reasoning') and message.reasoning:
            thinking_text = message.reasoning
        
        # Method 2: Check response body for reasoning field
        elif hasattr(response, 'reasoning') and response.reasoning:
            thinking_text = response.reasoning
        
        # Method 3: Check in message content with special markers
        elif message.content and "<think" in message.content:
            # Some models embed thinking in content with XML tags
            import re
            think_match = re.search(r'<think[^>]*>(.*?)</think', message.content, re.DOTALL)
            if think_match:
                thinking_text = think_match.group(1).strip()
        
        if thinking_text:
            log(f"\n🧠 THINKING:\n{thinking_text}")
            reasoning_trace.append(thinking_text)
        else:
            log("\n🧠 (No thinking content returned)")
        
        # ---- CHECK FOR TOOL CALLS ----
        if not message.tool_calls:
            log(f"\n💬 FINAL ANSWER:\n{message.content}")
            return {
                "status": "complete",
                "answer": message.content,
                "reasoning_steps": reasoning_trace,
                "tool_calls": tool_calls_log,
                "iterations": iteration
            }
        
        # ---- PROCESS TOOL CALLS ----
        log(f"\n🔧 TOOL CALLS ({len(message.tool_calls)}):")
        
        # Add assistant message with tool calls to conversation
        messages.append(message.model_dump(exclude_none=True))
        
        for tool_call in message.tool_calls:
            func_name = tool_call.function.name
            func_args = json.loads(tool_call.function.arguments)
            
            log(f"  → {func_name}({json.dumps(func_args)})")
            
            # Execute the tool
            if func_name in TOOL_REGISTRY:
                result = TOOL_REGISTRY[func_name](**func_args)
            else:
                result = json.dumps({"error": f"Unknown tool: {func_name}"})
            
            log(f"  ← {result[:200]}{'...' if len(result) > 200 else ''}")
            
            # Log for return value
            tool_calls_log.append({
                "iteration": iteration,
                "tool": func_name,
                "args": func_args,
                "result": result
            })
            
            # Add tool result to messages (required for conversation continuity)
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result
            })
    
    return {
        "status": "max_iterations",
        "answer": None,
        "reasoning_steps": reasoning_trace,
        "tool_calls": tool_calls_log,
        "iterations": max_iterations
    }


# ============================================================
# AGENT IMPLEMENTATION - Direct HTTP (for debugging)
# ============================================================

def run_agent_direct_http(
    query: str,
    model: str = MODEL_ID,
    max_iterations: int = 10,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Run agent using direct HTTP requests to OpenRouter API.
    Useful for debugging the exact request/response format.
    """
    
    messages: List[Dict] = [{"role": "user", "content": query}]
    reasoning_trace: List[str] = []
    tool_calls_log: List[Dict] = []
    
    def log(msg: str) -> None:
        if verbose:
            print(msg)
    
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:8000",
        "X-Title": "GLM5-Thinking-Agent"
    }
    
    for iteration in range(1, max_iterations + 1):
        log(f"\n{'='*70}")
        log(f"ITERATION {iteration}")
        log(f"{'='*70}")
        
        payload = {
            "model": model,
            "messages": messages,
            "tools": TOOLS,
            "tool_choice": "auto",
            "include_reasoning": True
        }
        
        if verbose:
            log(f"\n📤 REQUEST (messages count: {len(messages)})")
        
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload
        )
        
        if response.status_code != 200:
            log(f"\n❌ HTTP Error {response.status_code}: {response.text}")
            return {
                "status": "error",
                "error": response.text,
                "reasoning_steps": reasoning_trace,
                "tool_calls": tool_calls_log,
                "iterations": iteration
            }
        
        data = response.json()
        
        if verbose:
            # Print full response for debugging (truncate large fields)
            debug_data = json.dumps(data, indent=2)
            if len(debug_data) > 2000:
                log(f"\n📥 RESPONSE (truncated):\n{debug_data[:2000]}...")
            else:
                log(f"\n📥 RESPONSE:\n{debug_data}")
        
        choice = data["choices"][0]
        message = choice["message"]
        
        # ---- EXTRACT REASONING ----
        thinking_text = None
        
        # Check various locations for reasoning
        if "reasoning" in message and message["reasoning"]:
            thinking_text = message["reasoning"]
        elif "reasoning" in data and data["reasoning"]:
            thinking_text = data["reasoning"]
        elif "reasoning_content" in message and message["reasoning_content"]:
            thinking_text = message["reasoning_content"]
        
        if thinking_text:
            log(f"\n🧠 THINKING:\n{thinking_text}")
            reasoning_trace.append(thinking_text)
        
        # ---- CHECK FOR TOOL CALLS ----
        if "tool_calls" not in message or not message["tool_calls"]:
            log(f"\n💬 FINAL ANSWER:\n{message.get('content', 'No content')}")
            return {
                "status": "complete",
                "answer": message.get("content"),
                "reasoning_steps": reasoning_trace,
                "tool_calls": tool_calls_log,
                "iterations": iteration,
                "raw_response": data  # Include for debugging
            }
        
        # ---- PROCESS TOOL CALLS ----
        log(f"\n🔧 TOOL CALLS ({len(message['tool_calls'])}):")
        
        # Add assistant message to conversation
        messages.append(message)
        
        for tool_call in message["tool_calls"]:
            func_name = tool_call["function"]["name"]
            func_args = json.loads(tool_call["function"]["arguments"])
            
            log(f"  → {func_name}({json.dumps(func_args)})")
            
            if func_name in TOOL_REGISTRY:
                result = TOOL_REGISTRY[func_name](**func_args)
            else:
                result = json.dumps({"error": f"Unknown tool: {func_name}"})
            
            log(f"  ← {result[:200]}{'...' if len(result) > 200 else ''}")
            
            tool_calls_log.append({
                "iteration": iteration,
                "tool": func_name,
                "args": func_args,
                "result": result
            })
            
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call["id"],
                "content": result
            })
    
    return {
        "status": "max_iterations",
        "answer": None,
        "reasoning_steps": reasoning_trace,
        "tool_calls": tool_calls_log,
        "iterations": max_iterations
    }


# ============================================================
# TESTING
# ============================================================

if __name__ == "__main__":
    # Test queries that require thinking and tool use
    
    test_queries = [
        # Simple tool use with thinking
        "What is 345 multiplied by 89, then add 1234 to the result?",
        
        # Multi-step requiring multiple tool calls
        "Calculate (500 - 125) * 8, then analyze the text 'The result is X' where X is your calculated value.",
        
        # Complex reasoning
        "I have a rectangle with length 47 and width 23. Calculate the area and perimeter. Then analyze the word 'rectangle'."
    ]
    
    # Run with OpenAI SDK
    print("\n" + "#"*70)
    print("# USING OPENAI SDK")
    print("#"*70)
    
    result = run_agent_openai_sdk(
        query=test_queries[1],
        verbose=True
    )
    
    print("\n" + "#"*70)
    print("# SUMMARY")
    print("#"*70)
    print(f"Status: {result['status']}")
    print(f"Iterations: {result['iterations']}")
    print(f"Reasoning steps: {len(result['reasoning_steps'])}")
    print(f"Tool calls made: {len(result['tool_calls'])}")
