# 💻 Set Up Your Laptop: A Step-by-Step Manual

<!-- markdownlint-disable MD013 -->

> **In plain English:** install the tools once, **the week before the event**, so the first hour of the hackathon goes on building, not installing. Follow the section for your computer, then do **"Check it all works"** at the end.
>
> **Time needed:** 20–40 minutes.

---

## What you're installing

| Tool | What it's for |
| --- | --- |
| **Node.js (LTS)** | Runs JavaScript apps like Next.js. Includes `npm` and `npx` |
| **Git** | Saves versions of your code and uploads them to GitHub |
| **VS Code** | The code editor (or Cursor, which is based on it) |
| **GitHub CLI** (`gh`) | Logs git into GitHub without passwords. Optional but easier |
| **Claude Code** (for BMad) | The AI coding assistant this toolkit's BMad workflow runs in. Installed in [`setup-bmad.md`](setup-bmad.md). No Claude Code? Pick a fallback from [`../../05-tools-and-mcp/docs/free-ai-dev-tools.md`](../../05-tools-and-mcp/docs/free-ai-dev-tools.md) |

---

## 🪟 Windows

**Step 1. Open PowerShell** (Start menu → type "PowerShell" → open it).

**Step 2. Install everything with one command each** (`winget` comes with Windows 10 and 11):

```powershell
winget install OpenJS.NodeJS.LTS
winget install Git.Git
winget install Microsoft.VisualStudioCode
winget install GitHub.cli
```

**Step 3. Close PowerShell and open a new window**, so it picks up the new tools.

**Step 4. Fix the "running scripts is disabled" error before you hit it:**

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Without this, commands like `npm` or `npx` can fail in PowerShell with *"cannot be loaded because running scripts is disabled on this system"*.

**Step 5. Put your projects in a plain folder, not OneDrive, Desktop or Documents.**

```powershell
mkdir C:\dev
cd C:\dev
```

⚠️ OneDrive tries to sync `node_modules` (tens of thousands of files). That makes installs slow and can lock files mid-build, causing random `EPERM` or `EBUSY` errors. If your Desktop or Documents folder syncs to OneDrive, keep code out of it.

**Windows tips:**

- **Git Bash** is installed with Git. Use it when a guide shows Mac/Linux commands (`export`, `cp`, `rm -rf`). Some tools, such as Claude Code hooks, expect bash ([`../../05-tools-and-mcp/docs/claude-code-power-tools.md`](../../05-tools-and-mcp/docs/claude-code-power-tools.md)).
- Setting an environment variable for one session: bash uses `export KEY=value`, PowerShell uses `$env:KEY="value"`.

---

## 🍎 Mac

**Step 1. Open Terminal** (Cmd + Space → type "Terminal").

**Step 2. Install Apple's developer tools** (this includes git):

```bash
xcode-select --install
```

**Step 3. Install Homebrew** by following the one-line command on [brew.sh](https://brew.sh), then:

```bash
brew install node gh
brew install --cask visual-studio-code
```

**Step 4. Make a projects folder:**

```bash
mkdir ~/dev && cd ~/dev
```

---

## 🐧 Linux

Install Node LTS with [nvm](https://github.com/nvm-sh/nvm) (avoids permission problems with the system package), and `git` and `gh` with your package manager.

---

## Everyone: connect git to GitHub

**Step 1. Tell git who you are** (this name and email show on your commits):

```bash
git config --global user.name "Your Name"
git config --global user.email "you@example.com"
```

**Step 2. Log in to GitHub:**

```bash
gh auth login
```

Choose **GitHub.com** → **HTTPS** → **Yes** (authenticate git) → **Login with a web browser**, then follow the prompts.

---

## ✅ Check it all works

Run each line. Every one should print a version number or "Logged in", not an error.

```bash
node -v          # v22 or newer
npm -v
git --version
gh auth status
```

**Final test: make and run a real app** (this also downloads packages into your cache, which helps on slow venue Wi-Fi):

```bash
npx create-next-app@latest test-app --yes
cd test-app
npm run dev
```

Open <http://localhost:3000>. If you see the Next.js page, you're ready. Press `Ctrl + C` to stop it, then delete the `test-app` folder.

---

## ✅ Accounts to create before the event

Tick these off using [`../first-hackathon.md`](../first-hackathon.md) → "Before the event": **GitHub**, **Vercel** (sign up *with* GitHub), **Supabase**, and your AI tool. Students: claim the [GitHub Student Pack](free-credits.md).

## Next: set up BMad

Your laptop is ready. Now install Claude Code and practise BMad once: [`setup-bmad.md`](setup-bmad.md) → Part A.

## Common setup errors

| Error | Fix |
| --- | --- |
| `'node'` / `'git'` is not recognized | Close **all** terminal windows and open a new one. Still broken? Restart the computer |
| `running scripts is disabled on this system` | Windows Step 4 above |
| `EACCES` permission error on `npm install -g` (Mac/Linux) | Don't use `sudo`. Reinstall Node with nvm or Homebrew |
| `EPERM` / `EBUSY` during install (Windows) | Move the project out of OneDrive (Windows Step 5) and close VS Code's other windows |
| `gh: command not found` right after install | Open a new terminal window |
