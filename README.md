# MemCodeAgent

A lightweight CLI coding agent with project memory and code retrieval.

MemCodeAgent is built for the software engineering admission project: it talks to a large language model, calls local tools, reads and edits files, runs commands, observes results, and iterates toward a programming task. The core agent loop, tool execution, context management, output parsing, and memory layer are implemented in this repository rather than delegated to an agent framework.

## Goals

- Provide a clear command-line coding-agent workflow.
- Keep local file and command execution explicit and inspectable.
- Add project memory inspired by layered agent memory.
- Add grouped code retrieval inspired by entity-centric vector search.
- Keep the implementation small enough to explain during an interview.

## Quick Start

```powershell
python -m pip install -e .[dev]
$env:OPENAI_API_KEY="your-key"
$env:MEMCODE_MODEL="gpt-4o-mini"
mca run "Read this project and summarize the next implementation step."
```

Optional environment variables:

```text
OPENAI_API_KEY      required for real model calls
OPENAI_BASE_URL     optional OpenAI-compatible endpoint
MEMCODE_MODEL       model name, defaults to gpt-4o-mini
```

## Current Architecture

```text
user task
  -> retrieve project memory and related code
  -> ask LLM for the next action
  -> execute a local tool
  -> append observation to context
  -> modify / test / retry
  -> summarize result
  -> write long-term memory
```

The initial framework contains the CLI, agent loop skeleton, local tool layer, workspace guard, LLM client wrapper, and memory/retrieval interfaces. Implementation will be added step by step with small commits.

## Safety Notes

API keys must be provided through environment variables or local untracked config files. Do not commit credentials, terminal recordings with visible keys, or generated secret files.

## License

MIT
