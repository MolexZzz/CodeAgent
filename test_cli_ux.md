# CLI UX Improvements Test Plan

## Test the three improvements:

### 1. Simplified step display (verbose mode toggle)
- Default mode should show: `→ calling tool1, tool2...` then `✓ tool_name` or `✗ tool_name`
- Verbose mode should show: full `Step N` headers with detailed tool arguments and observation content
- Use `/verbose` to toggle between modes

### 2. Model switching
- `/model` should show current model
- `/model deepseek-chat` should switch to deepseek-chat (or any other model name)
- Should show confirmation message when switching

### 3. Comprehensive help
- `/help` should show all 8 commands:
  - /help, /exit, /quit, /clear, /history, /model, /verbose, /workspace

## Quick verification commands:

```bash
cd "D:\Coding Agent"
mca chat
```

Then in the REPL:
1. Type `/help` - should see all 8 commands nicely formatted
2. Type `/model` - should show current model (e.g., gpt-4o-mini or deepseek-chat)
3. Type `/model test-model` - should show "Model switched: old -> test-model"
4. Type `/verbose` - should toggle verbose mode and show "Verbose mode: on/off"
5. Type `/workspace` - should show workspace path
6. Ask a simple question to see the simplified output format
7. Toggle `/verbose` again and ask another question to see full detail
8. Type `/exit` to quit

## Expected behavior:
- Startup shows: model, workspace, and help hint
- Step display is much clearer (no confusing "Step 1" spam in default mode)
- Can switch models mid-session
- Help shows comprehensive command list
