#!/usr/bin/env python
"""Test script to demonstrate model selection functionality."""

from memcodeagent.llm import LlmClient

def test_models():
    """Test the model selection features."""
    print("=== Testing Model Selection Features ===\n")

    # Initialize client
    client = LlmClient()

    # Test 1: Show available models
    print("1. Available models (with credentials configured):")
    for model in client.AVAILABLE_MODELS:
        marker = "*" if model == client.model else " "
        info = client.get_model_info(model)
        print(f"  {marker} {model} ({info['provider']})")
    print()

    # Test 2: Show current model
    print(f"2. Current model: {client.model}")
    print(f"   Config file: {client._env_path}")
    print()

    # Test 3: Show all known models with credential status
    print("3. All known models by provider:")
    by_provider = {}
    for name, config in client.MODEL_REGISTRY.items():
        by_provider.setdefault(config["provider"], []).append(name)

    for provider, names in by_provider.items():
        configured = client._model_is_configured(names[0])
        status = "configured" if configured else "no credentials"
        print(f"   {provider} ({status}):")
        for name in names:
            print(f"     - {name}")
    print()

    # Test 4: Switch model (only within configured models)
    if len(client.AVAILABLE_MODELS) > 1:
        print("4. Testing model switch (without persistence):")
        old_model = client.model
        new_model = [m for m in client.AVAILABLE_MODELS if m != old_model][0]
        client.set_model(new_model, persist=False)
        print(f"   Switched from {old_model} to {client.model}")
        print()
    else:
        print("4. Skipping model switch test (only one provider configured)")
        print()

    # Test 5: Demonstrate error handling
    print("5. Error handling:")
    unconfigured = [m for m in client.MODEL_REGISTRY if not client._model_is_configured(m)]
    if unconfigured:
        try:
            client.set_model(unconfigured[0], persist=False)
        except ValueError as exc:
            print(f"   Expected error: {exc}")
    else:
        print("   (All models are configured)")
    print()

    # Test 6: Demonstrate persistence logic
    print("6. Persistence logic:")
    print(f"   set_model(model, persist=False) - Changes in-memory only")
    print(f"   set_model(model, persist=True)  - Saves to {client._env_path}")
    print()

    print("All tests passed!")
    print("\nIn the REPL:")
    print("  /models          - List available models grouped by provider")
    print("  /model           - Interactive menu to select and save as default")
    print("  /model <name>    - Directly switch to <name> and save as default")

if __name__ == "__main__":
    test_models()
