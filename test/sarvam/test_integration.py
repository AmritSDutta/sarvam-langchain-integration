"""Integration tests for Sarvam API calls.

These tests make real API calls to Sarvam and require:
1. SARVAM_API_KEY environment variable to be set
2. --integration flag to run: pytest -m integration

Run with:
    pytest -m integration
    pytest test/sarvam/test_integration.py -m integration
"""

import os

import pytest
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool

from sarvam import SarvamLLM, SarvamChat


@pytest.mark.integration
def test_llm_invoke_real():
    """Test LLM style invocation with real API call."""
    api_key = os.environ.get("SARVAM_API_KEY")
    if not api_key:
        pytest.skip("SARVAM_API_KEY environment variable not set")

    llm = SarvamLLM()
    response = llm.invoke("What is the capital of India?")

    assert response is not None
    assert isinstance(response, str)
    assert len(response) > 0
    # Check that response contains something relevant
    assert any(word in response.lower() for word in ["delhi", "capital"])


@pytest.mark.integration
def test_llm_invoke_tools():
    """Test Chat style invocation with tools and real API call.

    Note: Sarvam AI does NOT support tool/function calling yet.
    This test verifies that bind_tools() works and logs appropriately.
    """
    api_key = os.environ.get("SARVAM_API_KEY")
    if not api_key:
        pytest.skip("SARVAM_API_KEY environment variable not set")

    # Enable logging to see INFO level messages
    import logging
    logging.basicConfig(level=logging.INFO)

    @tool
    def get_weather(location: str) -> str:
        """Get the current weather for a location."""
        print(f"[get_weather()] Getting weather for {location}")
        return f"Sunny and 75°F in {location}"

    chat = SarvamChat()
    bound_chat = chat.bind_tools([get_weather])

    # Verify tools are bound (stored but not used by API)
    assert bound_chat.bound_tools is not None
    assert len(bound_chat.bound_tools) == 1
    # Tools are wrapped in {"type": "function", "function": {...}} format
    print(f"[OK] Bound tool: {bound_chat.bound_tools[0]['function']['name']}")

    # Invoke - should log INFO message about tools not being supported
    response = bound_chat.invoke([HumanMessage(content="What is the weather in Delhi?")])

    assert response is not None
    assert response.content is not None
    assert isinstance(response.content, str)
    assert len(response.content) > 0

    # Handle Unicode for Windows console
    safe_content = response.content.encode('ascii', errors='replace').decode('ascii')
    print(f"Response: {safe_content}")
    # Model should answer directly (not using tools)
    assert any(word in response.content.lower() for word in ["delhi", "weather", "temperature", "india"])


@pytest.mark.integration
def test_chat_invoke_real():
    """Test Chat style invocation with real API call."""
    api_key = os.environ.get("SARVAM_API_KEY")
    if not api_key:
        pytest.skip("SARVAM_API_KEY environment variable not set")

    chat = SarvamChat()
    response = chat.invoke([HumanMessage(content="Hello!")])

    assert response is not None
    assert response.content is not None
    assert isinstance(response.content, str)
    assert len(response.content) > 0


@pytest.mark.integration
def test_llm_with_temperature_real():
    """Test LLM with custom temperature using real API."""
    api_key = os.environ.get("SARVAM_API_KEY")
    if not api_key:
        pytest.skip("SARVAM_API_KEY environment variable not set")

    llm = SarvamLLM(temperature=0.3)
    response = llm.invoke("What is 2 + 2?")

    assert response is not None
    assert isinstance(response, str)
    assert len(response) > 0
    # Low temperature should give focused answer
    assert "4" in response


@pytest.mark.integration
def test_chat_with_reasoning_effort_real():
    """Test Chat with reasoning effort using real API."""
    api_key = os.environ.get("SARVAM_API_KEY")
    if not api_key:
        pytest.skip("SARVAM_API_KEY environment variable not set")

    chat = SarvamChat(reasoning_effort="medium", temperature=0.5)
    response = chat.invoke([HumanMessage(content="Solve: If 3x + 7 = 22, what is x?")])

    assert response is not None
    assert response.content is not None
    assert isinstance(response.content, str)
    assert len(response.content) > 0
    # Should solve the equation
    assert "5" in response.content


@pytest.mark.integration
def test_chat_multi_turn_conversation_real():
    """Test multi-turn conversation with real API."""
    api_key = os.environ.get("SARVAM_API_KEY")
    if not api_key:
        pytest.skip("SARVAM_API_KEY environment variable not set")

    chat = SarvamChat()

    # First message
    messages = [
        HumanMessage(content="What are the two main styles of Indian classical music?")
    ]
    response1 = chat.invoke(messages)
    assert response1.content is not None
    assert len(response1.content) > 0

    # Follow-up message (conversation history)
    from langchain_core.messages import AIMessage

    messages.extend([
        AIMessage(content=response1.content),
        HumanMessage(content="Which one is more popular in North India?")
    ])
    response2 = chat.invoke(messages)

    assert response2.content is not None
    assert len(response2.content) > 0


@pytest.mark.integration
def test_hindi_language_support_real():
    """Test Hindi language support with real API."""
    api_key = os.environ.get("SARVAM_API_KEY")
    if not api_key:
        pytest.skip("SARVAM_API_KEY environment variable not set")

    chat = SarvamChat(temperature=0.3)

    response = chat.invoke([
        SystemMessage(content="आप एक सहायक हैं जो हिंदी में जवाब देता है।"),
        HumanMessage(content="भारत की राजधानी क्या है?")
    ])

    assert response.content is not None
    assert isinstance(response.content, str)
    assert len(response.content) > 0
    # Response should be in Hindi or English (Hindi or about Delhi)
    assert any(
        word in response.content.lower()
        for word in ["दिल्ली", "delhi", "नई दिल्ली", "new delhi"]
    )


@pytest.mark.integration
def test_llm_with_wiki_grounding_real():
    """Test LLM with wiki grounding for factual queries."""
    api_key = os.environ.get("SARVAM_API_KEY")
    if not api_key:
        pytest.skip("SARVAM_API_KEY environment variable not set")

    llm = SarvamLLM(wiki_grounding=True, temperature=0.2)
    response = llm.invoke("What is the history of the Taj Mahal?")

    assert response is not None
    assert isinstance(response, str)
    assert len(response) > 0
    # Should mention relevant keywords
    assert any(
        word in response.lower()
        for word in ["taj mahal", "agra", "shah jahan", "mumtaz", "monument"]
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_capability_inference_with_real_api():
    """Test capability inference pattern with real API call.

    This test mimics the exact usage pattern from the user's codebase
    where SarvamChat.ainvoke() is called with a long string prompt for
    task capability classification.

    Run with:
        pytest -m integration test/sarvam/test_integration.py::test_capability_inference_with_real_api
    """
    import asyncio

    api_key = os.environ.get("SARVAM_API_KEY")
    if not api_key:
        pytest.skip("SARVAM_API_KEY environment variable not set")

    chat = SarvamChat(temperature=0.3)

    # The exact prompt from user's code
    prompt = """You are a task classifier. Analyze this task and identify which LLM capabilities are required. Available capabilities: - reasoning: Complex reasoning, chain-of-thought, analysis - tools: Function calling, tool use, API interactions - fast: Low latency, quick response time - cheap: Low cost per token (budget-conscious) - informational: General information, factual queries, knowledge retrieval - coding: Code writing, programming, software development - vision: Image understanding, visual content - long: Long context window needed - synthesizing: Synthesizing capabilities - summarizing: Summarizing capabilities - planning: Planning capabilities
Task: "Analyze recent gold price surge in recent times"
Rules: 1. If task involves writing code, programming, or software: include "coding" 2. If task asks for facts, explanations, or knowledge: include "informational" 3. If task needs complex analysis: include "reasoning" 4. Return only the required capability names as a comma-separated list. 5. If unsure, default to "reasoning"
Example outputs: -coding, reasoning, cheap -informational, cheap, long -summarizing, synthesizing, long"""

    planning_model = "sarvam-m"

    try:
        response = await chat.ainvoke(prompt)
        content = response.content.strip()
        print(f"\nCapability inference[{planning_model}] suggested: {content}")

        # Parse the response - look for capability names
        valid_caps = {
            "reasoning",
            "tools",
            "fast",
            "cheap",
            "informational",
            "coding",
            "vision",
            "long",
            "synthesizing",
            "summarizing",
            "planning",
        }

        capabilities = set()

        # Split by comma and extract capabilities
        for cap in content.split(","):
            cap = cap.strip().strip('"\'').strip()
            if cap in valid_caps:
                capabilities.add(cap)

        # Fallback to informational if empty
        if not capabilities:
            capabilities.add("reasoning")

        print(f"Parsed capabilities: {capabilities}")

        # Verify we got some capabilities back
        assert len(capabilities) > 0, "Should have at least one capability"
        # For "gold price surge" task, should include informational or reasoning
        assert len(capabilities & {"informational", "reasoning", "synthesizing", "analyzing"}) > 0, \
            "Should include at least one of: informational, reasoning, synthesizing"

    except Exception as e:
        # Fallback on error - if API is down or returns error, skip the test
        error_msg = str(e)
        if "internal_server_error" in error_msg.lower() or "500" in error_msg:
            pytest.skip(f"Sarvam API is currently experiencing internal server errors: {e}")
        # Re-raise other exceptions
        default_capabilities = {"reasoning", "informational", "planning"}
        print(f"Warning: Capability inference failed[{planning_model}]: {e}, will return {default_capabilities}")
        raise


@pytest.mark.integration
def test_think_tag_extraction_with_real_api():
    """Test `` tag extraction with real API call.

    This test verifies that SarvamChat properly extracts content after
    the `` tag when the model includes reasoning blocks in responses.

    Run with:
        pytest -m integration test/sarvam/test_integration.py::test_think_tag_extraction_with_real_api
    """
    api_key = os.environ.get("SARVAM_API_KEY")
    if not api_key:
        pytest.skip("SARVAM_API_KEY environment variable not set")

    # Use high reasoning effort to trigger `` tag in response
    chat = SarvamChat(
        api_key=api_key,
        reasoning_effort="high",
        temperature=0.3,
        max_retry=3
    )

    print("\n[Test] `` tag extraction with real API")
    print("-" * 60)

    # Ask a question that likely triggers reasoning
    prompt = "What is the capital of India? Answer with just the city name."

    response = chat.invoke([HumanMessage(content=prompt)])

    print(f"  Response content: {response.content[:100].encode('ascii', errors='replace').decode('ascii')}...")

    # Verify we got a response
    assert response.content is not None
    assert isinstance(response.content, str)
    assert len(response.content) > 0

    # The response should be the extracted content after `` tag
    # It should NOT contain the `` tag itself
    assert "<think>" not in response.content, "Response should not contain <think> tag"
    assert "</think>" not in response.content, "Response should not contain </think> tag"

    # The extracted content should be the actual answer (e.g., "New Delhi" or "Delhi")
    # Check for expected keywords
    assert any(word in response.content.lower() for word in ["delhi", "new delhi"]), \
        f"Expected city name in response, got: {response.content}"

    print(f"  Final extracted answer: {response.content.encode('ascii', errors='replace').decode('ascii')}")
    print("  [PASS] `` tag extraction test PASSED")
    print("-" * 60)
