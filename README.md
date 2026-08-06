# 🧠 Knowledge Base (`kb`)

> **Local AI Knowledge Base & Context Engine for Antigravity AI**  
> *Seamlessly indexes your codebase, tracks git state, remembers architecture decisions, and generates super-charged prompts for Antigravity.*

---

## ⚡ Features

- **⚡ Automatic Codebase Scanning**: Automatically detects and incrementally re-indexes added, modified, or deleted files without manual `kb scan` commands.
- **🚀 Antigravity Super-Prompt Skill (`/prompt`)**: Generates structured, context-dense prompts containing hybrid-retrieved source code, file maps, git diffs, and architecture notes, formatted for Antigravity.
- **🔍 Hybrid Retrieval Engine**: Fuses offline semantic search (FAISS + `all-MiniLM-L6-v2`) with keyword search (SQLite FTS5) using Reciprocal Rank Fusion (RRF).
- **👁️ Real-Time File Watcher**: Automatically updates vector and keyword indexes as you edit code during interactive chat sessions.
- **🧠 Long-Term Architecture Memory**: Save persistent architectural decisions (`/note`) and task items (`/task`) attached to your project.
- **🌿 Native Git Integration**: Captures branch state, uncommitted diffs, and recent commits for AI context awareness.

---

## 📦 Installation

### One-Line CLI Installation

```bash
pip install git+https://github.com/Sweekar-m/Knowledge_base.git
```

*(Alternatively, for local development: `git clone https://github.com/Sweekar-m/Knowledge_base.git && pip install -e .`)*

### One-Line Uninstallation

```bash
pip uninstall knowldge-base -y
```

---

## ⚙️ Configuration

Set your NVIDIA API key in environment variables or edit `kb_config.yaml`:

```bash
set KB_NVIDIA_API_KEY=nvapi-YOUR_NVIDIA_API_KEY_HERE
```

Or configure `kb_config.yaml`:
```yaml
nvidia:
  api_key: "nvapi-YOUR_KEY"
  model: "nvidia/nemotron-3-ultra-550b-a55b"
  temperature: 0.3
```

---

## 🚀 Quick Start Guide

### 1. Launch Interactive Chat (Auto-Scans Automatically)

Simply pass your project path (or run inside your project directory):

```bash
kb chat C:\path\to\your\project
```

### 2. Generate Antigravity Super-Prompt

Use the `/prompt` command inside `kb chat` or via CLI:

```bash
kb prompt "Fix authentication state persistence in dashboard island"
```

The generated prompt will be rendered in your terminal and automatically copied to your clipboard ready to paste into Antigravity!

---

## 🛠️ CLI Reference

| Command | Description |
|---|---|
| `kb chat [path]` | Start interactive chat with auto-scanning and real-time file watcher |
| `kb prompt <issue>` | Generate Antigravity super-prompt & auto-copy to clipboard |
| `kb scan [path]` | Manually trigger incremental indexing |
| `kb search <query>` | Perform hybrid search (semantic + keyword) without calling LLM |
| `kb status [path]` | Display project stats, indexed files, and vector count |
| `kb memory [path]` | View or add architecture notes (`--add`) and tasks (`--tasks`) |
| `kb git [path]` | View current git status, branch, and uncommitted diffs |
| `kb rebuild [path]` | Drop database and perform a clean index rebuild |
| `kb watch [path]` | Standalone live file watcher daemon |

---

## 💬 Interactive Chat Slash Commands

Inside an active `kb chat` session, use the following commands:

```text
/prompt [issue]   Generate & auto-copy an Antigravity Super-Prompt for an issue
/note <text>      Save an architecture note to long-term project memory
/task <text>      Add a pending project task item
/search <query>   Perform quick hybrid search directly in chat
/history          Show recent conversation message history
exit              Quit chat session
```

---

## 📄 License

MIT License © [Sweekar-m](https://github.com/Sweekar-m)
