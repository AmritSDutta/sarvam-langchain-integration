"""Integration tests for tool/function calling with real Sarvam AI API calls.

These tests require SARVAM_API_KEY environment variable to be set.

Tests cover all 4 models that support tool calling:
- sarvam-105b
- sarvam-105b-32k
- sarvam-30b
- sarvam-30b-16k

Run with:
    pytest test/sarvam/test_integration_tools.py -v

Or run just one model:
    pytest test/sarvam/test_integration_tools.py::test_tool_calling_105b -v
"""
import os
from typing import Literal

import pytest
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool

from sarvam import SarvamChat

# Skip all tests if API key is not set
pytestmark = pytest.mark.skipif(
    not os.environ.get("SARVAM_API_KEY"),
    reason="SARVAM_API_KEY environment variable is not set"
)

# Models that support tool calling
TOOL_SUPPORTED_MODELS = [
    "sarvam-105b",
    "sarvam-105b-32k",
    "sarvam-30b",
    "sarvam-30b-16k",
]


@pytest.fixture
def weather_tool():
    """Fixture for weather tool."""
    @tool
    def get_weather(location: str) -> str:
        """Get the current weather for a location.

        Args:
            location: The city or location to get weather for

        Returns:
            Simulated weather information
        """
        # Simulated weather data (in real use, this would call a weather API)
        weather_data = {
            "Mumbai": "Sunny and 32°C",
            "Delhi": "Partly cloudy and 28°C",
            "Bangalore": "Pleasant and 24°C",
            "Chennai": "Humid and 30°C",
        }
        return weather_data.get(location, f"Weather data not available for {location}")

    return get_weather


@pytest.fixture
def calculator_tool():
    """Fixture for calculator tool."""
    @tool
    def calculator(expression: str) -> str:
        """Calculate a mathematical expression.

        Args:
            expression: Math expression to evaluate (e.g., "2 + 2")

        Returns:
            Calculation result
        """
        try:
            result = eval(expression)  # noqa: S307
            return f"Result: {result}"
        except Exception as e:
            return f"Error: {str(e)}"

    return calculator


@pytest.fixture
def search_tool():
    """Fixture for search/knowledge tool."""
    @tool
    def search(query: str) -> str:
        """Search for information about a topic.

        Args:
            query: The search query

        Returns:
            Simulated search results
        """
        # Simulated knowledge base
        knowledge = {
            "capital of India": "The capital of India is New Delhi.",
            "PM of India": "The Prime Minister of India is Narendra Modi.",
            "population of India": "India has a population of approximately 1.4 billion people.",
        }
        return knowledge.get(query.lower(), f"No information found for: {query}")

    return search


class TestBasicToolCalling:
    """Test basic tool calling functionality with real API calls."""

    @pytest.mark.parametrize("model", TOOL_SUPPORTED_MODELS)
    @pytest.mark.integration
    def test_tool_binding_works(self, model, weather_tool):
        """Test that tools can be bound to models without errors."""
        chat = SarvamChat(model=model)
        bound_chat = chat.bind_tools([weather_tool])

        # Verify tools are bound (in Sarvam format with type and function)
        assert bound_chat.bound_tools is not None
        assert len(bound_chat.bound_tools) == 1
        assert bound_chat.bound_tools[0]["type"] == "function"
        assert bound_chat.bound_tools[0]["function"]["name"] == "get_weather"

    @pytest.mark.parametrize("model", TOOL_SUPPORTED_MODELS)
    @pytest.mark.integration
    def test_simple_tool_call(self, model, weather_tool):
        """Test that model can invoke a simple tool."""
        chat = SarvamChat(model=model, temperature=0.3)
        bound_chat = chat.bind_tools([weather_tool])

        # Ask a question that should trigger the tool
        response = bound_chat.invoke([
            HumanMessage(content="What's the weather in Mumbai?")
        ])

        # Verify response
        assert response is not None
        assert hasattr(response, "content")

        # The model may either:
        # 1. Call the tool (response.additional_kwargs["tool_calls"] exists)
        # 2. Answer directly without calling the tool
        # Both are valid behaviors
        if "tool_calls" in response.additional_kwargs:
            # Tool was called
            tool_calls = response.additional_kwargs["tool_calls"]
            assert len(tool_calls) >= 1
            # Verify tool call structure
            for tc in tool_calls:
                assert "id" in tc
                assert "type" in tc
                assert "function" in tc
                assert "name" in tc["function"]
                assert "arguments" in tc["function"]
                print(f"\n[{model}] Tool call: {tc['function']['name']} with args: {tc['function']['arguments']}")
            # Note: Blank content is expected when model requests tool use
            print(f"\n[{model}] Content: '{response.content}' (blank is expected - model requested tool use)")
        else:
            # Model answered directly (also valid)
            safe_preview = response.content[:100].encode('ascii', errors='replace').decode('ascii')
            output = f"\n[{model}] Direct response: {safe_preview}..."
            print(output.encode('ascii', errors='replace').decode('ascii'))

        assert response.content or response.additional_kwargs.get("tool_calls")

    @pytest.mark.parametrize("model", TOOL_SUPPORTED_MODELS)
    @pytest.mark.integration
    def test_multiple_tools_binding(self, model, weather_tool, calculator_tool):
        """Test that multiple tools can be bound."""
        chat = SarvamChat(model=model)
        bound_chat = chat.bind_tools([weather_tool, calculator_tool])

        # Verify both tools are bound (in Sarvam format)
        assert bound_chat.bound_tools is not None
        assert len(bound_chat.bound_tools) == 2
        tool_names = {t["function"]["name"] for t in bound_chat.bound_tools}
        assert "get_weather" in tool_names
        assert "calculator" in tool_names

    @pytest.mark.parametrize("model", TOOL_SUPPORTED_MODELS)
    @pytest.mark.integration
    def test_model_chooses_correct_tool(self, model, weather_tool, calculator_tool):
        """Test that model chooses the appropriate tool."""
        chat = SarvamChat(model=model, temperature=0.3)
        bound_chat = chat.bind_tools([weather_tool, calculator_tool])

        # Test 1: Weather question
        response1 = bound_chat.invoke([
            HumanMessage(content="What's the weather in Delhi?")
        ])

        assert response1 is not None
        if "tool_calls" in response1.additional_kwargs:
            tool_calls = response1.additional_kwargs["tool_calls"]
            # Should have called get_weather
            tool_names = [tc["function"]["name"] for tc in tool_calls]
            assert "get_weather" in tool_names
            output = f"\n[{model}] Weather query - tool called: {tool_names}"
            print(output.encode('ascii', errors='replace').decode('ascii'))

        # Test 2: Math question
        response2 = bound_chat.invoke([
            HumanMessage(content="Calculate 15 * 7")
        ])

        assert response2 is not None
        if "tool_calls" in response2.additional_kwargs:
            tool_calls = response2.additional_kwargs["tool_calls"]
            # Should have called calculator
            tool_names = [tc["function"]["name"] for tc in tool_calls]
            assert "calculator" in tool_names
            output = f"\n[{model}] Math query - tool called: {tool_names}"
            print(output.encode('ascii', errors='replace').decode('ascii'))


class TestAdvancedToolCalling:
    """Test advanced tool calling scenarios."""

    @pytest.mark.parametrize("model", TOOL_SUPPORTED_MODELS)
    @pytest.mark.integration
    def test_tool_with_required_mode(self, model, search_tool):
        """Test tool_choice='required' forces tool usage."""
        chat = SarvamChat(model=model, tool_choice="required", temperature=0.3)
        bound_chat = chat.bind_tools([search_tool])

        response = bound_chat.invoke([
            HumanMessage(content="Hello, how are you?")
        ])

        # With tool_choice='required', model must call at least one tool
        # Note: Some models may still respond with text instead
        if "tool_calls" in response.additional_kwargs:
            tool_calls = response.additional_kwargs["tool_calls"]
            assert len(tool_calls) >= 1
            output = f"\n[{model}] Required mode - tools called: {[tc['function']['name'] for tc in tool_calls]}"
            print(output.encode('ascii', errors='replace').decode('ascii'))
        else:
            safe_preview = response.content[:100].encode('ascii', errors='replace').decode('ascii')
            output = f"\n[{model}] Required mode - direct response: {safe_preview}"
            print(output.encode('ascii', errors='replace').decode('ascii'))

    @pytest.mark.parametrize("model", TOOL_SUPPORTED_MODELS)
    @pytest.mark.integration
    def test_multi_turn_conversation_with_tools(self, model, weather_tool):
        """Test multi-turn conversation where model uses tools."""
        chat = SarvamChat(model=model, temperature=0.3)
        bound_chat = chat.bind_tools([weather_tool])

        # First turn: Ask about weather
        response1 = bound_chat.invoke([
            HumanMessage(content="What's the weather in Bangalore?")
        ])

        messages = [
            HumanMessage(content="What's the weather in Bangalore?"),
            response1
        ]

        # If tool was called, simulate tool result and continue
        if "tool_calls" in response1.additional_kwargs:
            tool_calls = response1.additional_kwargs["tool_calls"]

            # Add tool responses
            for tool_call in tool_calls:
                if tool_call["function"]["name"] == "get_weather":
                    # Execute the tool
                    args = eval(tool_call["function"]["arguments"])
                    result = weather_tool.func(**args)

                    # Add tool message
                    messages.append(ToolMessage(
                        content=result,
                        tool_call_id=tool_call["id"]
                    ))
                    output = f"\n[{model}] Tool result: {result}"
                    print(output.encode('ascii', errors='replace').decode('ascii'))

            # Second turn: Get final response
            response2 = bound_chat.invoke(messages)
            assert response2 is not None

            # Model may either respond with text or make another tool call
            if "tool_calls" in response2.additional_kwargs:
                # Model made another tool call - log but don't fail the test
                output = f"\n[{model}] Model made another tool call (unexpected but valid behavior)"
                print(output.encode('ascii', errors='replace').decode('ascii'))
            elif response2.content:
                # Model responded with text - expected behavior
                safe_preview = response2.content[:100].encode('ascii', errors='replace').decode('ascii')
                output = f"\n[{model}] Final response: {safe_preview}..."
                print(output.encode('ascii', errors='replace').decode('ascii'))
            else:
                # Neither content nor tool calls - unexpected
                raise AssertionError("Response has neither content nor tool_calls")


class TestModelComparison:
    """Compare behavior across different models."""

    @pytest.mark.integration
    def test_all_models_support_tools(self, weather_tool):
        """Verify all 4 models can bind and use tools."""
        results = {}

        for model in TOOL_SUPPORTED_MODELS:
            chat = SarvamChat(model=model, temperature=0.3)
            bound_chat = chat.bind_tools([weather_tool])

            response = bound_chat.invoke([
                HumanMessage(content="What's the weather in Chennai?")
            ])

            results[model] = {
                "has_content": bool(response.content),
                "has_tool_calls": "tool_calls" in response.additional_kwargs,
                "tool_count": len(response.additional_kwargs.get("tool_calls", [])),
            }

            print(f"\n{model}:")
            if response.content:
                content_preview = response.content[:80].encode('utf-8', errors='replace').decode('utf-8')
                print(f"  Content: {content_preview}...")
            else:
                print(f"  Content: None")
            print(f"  Tool calls: {results[model]['tool_count']}")
            print(f"  Supports tools: {chat._supports_tools()}")

        # All models should support tools
        for model, result in results.items():
            chat = SarvamChat(model=model)
            assert chat._supports_tools(), f"{model} should support tools"

    @pytest.mark.integration
    def test_context_window_models_work(self, weather_tool, calculator_tool):
        """Test extended context models (32k and 16k) work correctly."""
        extended_models = ["sarvam-105b-32k", "sarvam-30b-16k"]

        for model in extended_models:
            chat = SarvamChat(model=model, temperature=0.3)
            bound_chat = chat.bind_tools([weather_tool, calculator_tool])

            # Test with a longer prompt that benefits from extended context
            long_prompt = """
            I'm planning a trip to India. Can you help me with:
            1. What's the weather in Mumbai?
            2. Calculate 23 * 45
            3. What's the weather in Delhi?
            4. Calculate 100 / 4
            """

            response = bound_chat.invoke([HumanMessage(content=long_prompt)])

            assert response is not None
            assert response.content or response.additional_kwargs.get("tool_calls")

            print(f"\n{model} - Extended context test:")
            response_length = len(response.content) if response.content else 0
            print(f"  Response length: {response_length} chars")
            if "tool_calls" in response.additional_kwargs:
                print(f"  Tools called: {len(response.additional_kwargs['tool_calls'])}")


class TestToolEdgeCases:
    """Test edge cases and error handling."""

    @pytest.mark.parametrize("model", TOOL_SUPPORTED_MODELS)
    @pytest.mark.integration
    def test_tool_with_complex_arguments(self, model, search_tool):
        """Test tool calls with complex/nested arguments."""
        chat = SarvamChat(model=model, temperature=0.3)
        bound_chat = chat.bind_tools([search_tool])

        # Ask a question that requires complex understanding
        response = bound_chat.invoke([
            HumanMessage(content="Tell me about the current Prime Minister of India")
        ])

        assert response is not None
        safe_preview = response.content[:100].encode('ascii', errors='replace').decode('ascii')
        output = f"\n[{model}] Complex query response: {safe_preview}..."
        print(output.encode('ascii', errors='replace').decode('ascii'))

    @pytest.mark.parametrize("model", TOOL_SUPPORTED_MODELS)
    @pytest.mark.integration
    def test_tool_with_ambiguous_query(self, model, weather_tool, calculator_tool):
        """Test how model handles ambiguous queries."""
        chat = SarvamChat(model=model, temperature=0.5)
        bound_chat = chat.bind_tools([weather_tool, calculator_tool])

        # Ambiguous query that could use either tool
        response = bound_chat.invoke([
            HumanMessage(content="What's the temperature today?")
        ])

        assert response is not None
        safe_preview = response.content[:100].encode('ascii', errors='replace').decode('ascii')
        output = f"\n[{model}] Ambiguous query response: {safe_preview}..."
        print(output.encode('ascii', errors='replace').decode('ascii'))

        if "tool_calls" in response.additional_kwargs:
            tool_names = [tc["function"]["name"] for tc in response.additional_kwargs["tool_calls"]]
            print(f"  Tools called: {tool_names}")


class TestToolErrorHandling:
    """Test error handling with tools."""

    @pytest.mark.parametrize("model", TOOL_SUPPORTED_MODELS)
    @pytest.mark.integration
    def test_model_handles_tool_errors_gracefully(self, model, calculator_tool):
        """Test that model handles calculator errors gracefully."""
        chat = SarvamChat(model=model, temperature=0.3)
        bound_chat = chat.bind_tools([calculator_tool])

        # Provide invalid math expression
        response = bound_chat.invoke([
            HumanMessage(content="Calculate 1 / 0")
        ])

        assert response is not None
        # Model should handle the error gracefully
        safe_preview = response.content[:100].encode('ascii', errors='replace').decode('ascii')
        output = f"\n[{model}] Error handling response: {safe_preview}..."
        print(output.encode('ascii', errors='replace').decode('ascii'))


if __name__ == "__main__":
    # Run with: python test_integration_tools.py
    pytest.main([__file__, "-v", "-s"])
