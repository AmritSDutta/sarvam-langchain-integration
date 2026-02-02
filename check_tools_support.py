"""Check if Sarvam API supports tools/function calling."""

import os
from sarvamai import SarvamAI

api_key = os.environ.get("SARVAM_API_KEY")
if not api_key:
    print("SARVAM_API_KEY not set")
    exit(1)

client = SarvamAI(api_subscription_key=api_key)

# Test 1: Try calling with tools parameter
tool_definition = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g. San Francisco, CA"
                }
            },
            "required": ["location"]
        }
    }
}

try:
    print("Test 1: Calling API with tools parameter...")
    response = client.chat.completions(
        messages=[{"role": "user", "content": "What's the weather in Delhi?"}],
        tools=[tool_definition]
    )
    print(f"✓ API accepted tools parameter")
    print(f"Response: {response.choices[0].message.content}")

    # Check for tool_calls
    if hasattr(response.choices[0].message, "tool_calls"):
        print(f"Tool calls: {response.choices[0].message.tool_calls}")
    else:
        print("No tool_calls in response")

except Exception as e:
    print(f"✗ API rejected tools parameter: {e}")

print("\n" + "="*50)

# Test 2: Check client method signature
import inspect
print("Test 2: Checking chat.completions method signature...")
sig = inspect.signature(client.chat.completions)
print(f"Parameters: {list(sig.parameters.keys())}")

if "tools" in sig.parameters:
    print("✓ 'tools' parameter exists in method signature")
else:
    print("✗ 'tools' parameter NOT found in method signature")
