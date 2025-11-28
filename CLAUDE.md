# Claude Code Project Instructions - MANDATORY

## 1. Superpowers Workflow - MANDATORY FIRST RESPONSE

### What is Superpowers?

Superpowers is a plugin providing 20+ battle-tested skills for common development patterns. It's loaded automatically via SessionStart hook and represents your **primary workflow** for all development tasks.

**Skills are MANDATORY when they match your task - not optional, not suggestions. If a skill exists for your task, you MUST use it or you will fail.**

### Mandatory First Response Protocol

**Before responding to ANY user message, complete this checklist:**

1. ☐ List available skills in your mind
2. ☐ Ask yourself: "Does ANY skill match this request?"
3. ☐ If yes → Use the Skill tool to read and run the skill file
4. ☐ Announce which skill you're using
5. ☐ Follow the skill exactly

**Responding WITHOUT completing this checklist = automatic failure.**

### Common Rationalizations to Avoid

If you catch yourself thinking ANY of these thoughts, STOP. You are rationalizing. Check for and use the skill.

- **"This is just a simple question"** → WRONG. Questions are tasks. Check for skills.
- **"I can check git/files quickly"** → WRONG. Files don't have conversation context. Check for skills.
- **"Let me gather information first"** → WRONG. Skills tell you HOW to gather information. Check for skills.
- **"This doesn't need a formal skill"** → WRONG. If a skill exists for it, use it.
- **"I remember this skill"** → WRONG. Skills evolve. Run the current version.
- **"This doesn't count as a task"** → WRONG. If you're taking action, it's a task. Check for skills.
- **"The skill is overkill for this"** → WRONG. Skills exist because simple things become complex. Use it.
- **"I'll just do this one thing first"** → WRONG. Check for skills BEFORE doing anything.

**Why:** Skills document proven techniques that save time and prevent mistakes. Not using available skills means repeating solved problems and making known errors.

### Key Superpowers Skills

**Core Workflow Skills:**
- `superpowers:brainstorming` - Refine rough ideas into designs before coding (ALWAYS use before implementation)
- `superpowers:writing-plans` - Create detailed implementation plans with bite-sized tasks
- `superpowers:executing-plans` - Execute plans in batches with review checkpoints
- `superpowers:subagent-driven-development` - Dispatch subagents for independent tasks

**Development Skills:**
- `superpowers:test-driven-development` - RED-GREEN-REFACTOR cycle (write tests first, watch fail, implement)
- `superpowers:systematic-debugging` - Four-phase debugging (investigation, pattern, hypothesis, implementation)
- `superpowers:root-cause-tracing` - Trace bugs backward to find original trigger
- `superpowers:defense-in-depth` - Validate at every layer to make bugs structurally impossible

**Quality & Review Skills:**
- `superpowers:verification-before-completion` - Verify before claiming done (evidence before assertions)
- `superpowers:requesting-code-review` - Dispatch code-reviewer subagent
- `superpowers:receiving-code-review` - Handle review feedback with rigor
- `superpowers:testing-anti-patterns` - Prevent mocking behavior, test-only methods

**Workflow Management Skills:**
- `superpowers:using-git-worktrees` - Create isolated git worktrees for feature work
- `superpowers:finishing-a-development-branch` - Complete work with merge/PR/cleanup options
- `superpowers:dispatching-parallel-agents` - Investigate independent failures concurrently
- `superpowers:condition-based-waiting` - Replace timeouts with condition polling for tests

### Discovery Commands

**See all skills:**
```bash
ls .claude/skills/
cat .claude/skills/README.md
```

**See all slash commands:**
Type `/` and press Tab in Claude Code, or:
```bash
ls .claude/commands/
```

**Slash command patterns:**
- `/sp-*` → Superpowers skills (e.g., `/sp-brainstorm`, `/sp-tdd`)
- `/[domain]` → Domain skills (e.g., `/python`, `/react`, `/fastapi`)
- `/[action]` → Actions (e.g., `/start`, `/clean`, `/verify-complete`)

---

## 2. Discovery Patterns - Learning Your Tools

### The Pattern: Explore Before Asking

Before asking "how do I X", check if a skill or command exists for X. Before asking "where is Y", use ls/find/grep to locate Y. **Exploration before questions = faster, more independent work.**

### Finding Skills

```bash
# List all available skills
ls -la .claude/skills/

# Read a specific skill
cat .claude/skills/[skill-name].md

# Skills are also accessible via slash commands (/sp-* pattern)
```

**Project-Specific Skills:**
- `/backend-expert` - Interactive backend architecture expert
- `/strategy-sme` - Strategy Subject Matter Expert
- `/create-plugin` - Strategy plugin creation guide
- `/ui-verify` - Playwright UI testing standards
- And 40+ more...

### Finding Slash Commands

```bash
# In Claude Code, type / and press Tab to see all commands
# Or list them manually:
ls .claude/commands/

# Naming patterns:
# /sp-* → Superpowers skills (e.g., /sp-brainstorm, /sp-tdd)
# /[domain] → Domain skills (e.g., /python, /react, /fastapi)
# /[action] → Actions (e.g., /start, /clean, /verify-complete)
```

### Understanding Active Hooks

```bash
# See configured hooks
cat .claude/settings.local.json

# Read hook documentation
cat .claude/hooks/README.md

# List all available hooks
ls -la .claude/hooks/
```

---

## 3. Workflow Hooks - Automated Quality Gates

### What Hooks Do

Hooks are shell scripts that run automatically at specific Claude Code events. They provide automated quality gates, reminders, and workspace sync without manual intervention.

**Hook Types:**
- **SessionStart** - Runs when Claude Code starts (loads superpowers, backs up CLAUDE.md)
- **UserPromptSubmit** - Runs before processing user input (prompt evaluation/enrichment)
- **BeforeWrite** - Runs before Write tool (protects CLAUDE.md/AGENTS.md from overwrites)
- **SessionEnd** - Runs when session closes (cleanup, final sync)

### Active Hooks in Use

**Configured in** `.claude/settings.local.json`:
- **session-start.sh** - Loads superpowers plugin, backs up CLAUDE.md
- **before-write.sh** - Protects critical files from accidental modification
- See `.claude/hooks/README.md` for complete list

### Hook + Skill Integration

**How They Work Together:**
- **Hooks automate mechanics** - File sync, backups, quality checks, environment verification
- **Skills guide decisions** - When to TDD, how to debug, design patterns, workflow choices
- **Together: automation + intelligence** - Hooks handle the repetitive, skills handle the strategic

### Git Hooks vs Claude Hooks

**Git Hooks** (`.git/hooks/`):
- pre-commit, pre-push, post-commit, etc.
- Version control automation
- Enforce code quality before commits

**Claude Hooks** (`.claude/hooks/`):
- SessionStart, BeforeWrite, UserPromptSubmit, etc.
- IDE workflow automation
- Enhance development experience

**Both work together** for comprehensive quality gates and workflow automation.

---

## 4. Core Principles - Non-Negotiables

### Never Simulate - Always Execute

- **NEVER** simulate commands - execute everything
- **NEVER** create fake/mock/demo files - only production code
- **ALWAYS** provide proof of execution with real output
- **ALWAYS** work in `src/`, `tests/`, `config/` directories

**Why:** User needs WORKING PRODUCTION CODE, not demonstrations.

### Production Code Only

- Work on **REAL production files** only
- Test **REAL production code**, not examples
- Debug **REAL production code** directly
- No `fake_*.py`, `mock_*.js`, `demo_*.py`, `example_*.js`

### File Naming Conventions

| Category | Pattern | Git Status | Example |
|----------|---------|------------|---------|
| **Production Code** | `src/*`, `tests/*`, `config/*` | ✅ COMMIT | `src/api.py`, `tests/test_api.py` |
| **Claude Temporary** | `claude_*` | ❌ NEVER | `claude_temp_debug.py` |
| **Sprint Tests** | `claude_test_*` | ❌ NEVER | `claude_test_api_1234.py` |
| **QC Reports** | `QC_*` | Optional | `QC_Sprint_Report_20240315.md` |
| **Forbidden** | `fake_*`, `mock_*`, `demo_*` | ❌ DELETE | Never create these |

### Verification Protocols

**After File Changes:**
```bash
ls -la [file] && wc -l [file] && head -5 [file]
```

**Phase Completion Checklist:**
```bash
# 1. Production code exists
find src/ -type f ! -name "claude_*" | head -5

# 2. No TODOs in production
grep -r "TODO\|FIXME" src/ --exclude="claude_*" || echo "✓ Clean"

# 3. Tests pass on production
npm test || python -m pytest

# 4. No Claude files staged
git status --porcelain | grep -v "claude_"

# 5. Ready for commit
./scripts/qc_preflight.bat && echo "✓ Ready"
```

### Port Management - CRITICAL SAFETY

**NEVER kill all processes. Only kill specific port's process:**

```bash
# Windows - Find and kill specific port (e.g., 3000)
netstat -ano | findstr :3000  # Get PID (last number)
taskkill /PID [specific_PID] /F  # Kill ONLY that PID

# FORBIDDEN: taskkill /F /IM node.exe (kills ALL node processes)
```

### Code Standards

- **NO** emojis in code/comments/commits (unless explicitly requested)
- **NO** `console.log`/`print()` in production code
- **USE** proper naming: `snake_case` (Python), `camelCase` (JS/TS)
- **REMOVE** all debug statements before completion

### Directory Structure

```
project/
├── src/          # Production code (COMMIT)
├── tests/        # Production tests (COMMIT)
├── config/       # Configuration (COMMIT)
├── scripts/      # Utilities & QC scripts (COMMIT)
├── .claude/      # Claude Code config & skills (COMMIT)
├── .ian/         # Personal notes & docs (GIT-IGNORED)
└── .gitignore    # Must exclude: claude_*, .ian/
```

---

## 5. Standards & Procedures - Detail References

Detailed procedures have been extracted to `.ian/standards/` (git-ignored) for easier maintenance. Quick summaries and links below.

### Sprint Lifecycle

**Summary:** Sprint management includes sprint start procedures, development rules, QC report generation, and git commit workflows. All sprints must start with cleanup, end with QC checks, and include git commit hashes for traceability.

**Key Commands:**
```bash
# Sprint start
echo "SPRINT_$(date +%Y%m%d_%H%M%S)" > .current_sprint

# Sprint end QC
./scripts/qc_preflight.bat  # Windows
bash ./scripts/qc_preflight.sh  # Linux/Mac
```

**Full Details:** `.ian/standards/sprint-lifecycle.md`

### QC Reports & Preflight Checks

**Summary:** QC preflight scripts perform automated checks on code quality, file hygiene, git status, testing, and documentation. All sprints must pass QC before commit/push. Reports include git commit hashes for full traceability.

**QC Script Locations:**
- Windows: `scripts/qc_preflight.bat`
- Linux/Mac: `scripts/qc_preflight.sh`
- Sprint reports: `scripts/generate_sprint_report.ps1`

**Full Details:** `.ian/standards/qc-reports.md`

### GitHub Workflow

**Summary:** GitHub CLI must be authenticated in all Claude Code instances. Issue closing follows a mandatory template with resolution documents, verification scripts, code metrics, and impact analysis. All resolutions stored in `.ian/` directory.

**Quick Auth:**
```bash
/home/buckstrdr/quick_github_auth.sh
```

**Issue Closing:**
```bash
# Create resolution doc
touch .ian/issue_{NUMBER}_resolution.md

# Create verification script
touch tests/verify_issue_{NUMBER}_fix.py

# Close issue
gh issue close {NUMBER} --comment "$(cat .ian/issue_{NUMBER}_resolution.md)"
```

**Full Details:** `.ian/standards/github-workflow.md`

### Serena Integration

**Summary:** Serena provides advanced IDE assistance with language-server-based tools for symbol navigation, file operations, memory management, and workflow automation. Prefer Serena symbol tools over basic grep/find for code navigation.

**Key Capabilities:**
- Symbol tools for finding definitions/references/implementations
- Memory tools for storing project decisions and patterns
- Workflow tools for automating repetitive tasks
- Dashboard: http://localhost:24282/dashboard/

**Full Details:** `.ian/standards/serena-integration.md`

### Local Documentation

**Location:** `./97-Local-documentation/`

Before coding, check relevant framework docs:
- JavaScript, Node.js, React, TypeScript
- Python, FastAPI, Pydantic
- Testing, WebSocket, AsyncIO patterns
- And more...

**Use documentation to create PRODUCTION code, not examples.**

---

## 6. Agent Orchestration - Secondary Workflow (Optional)

### When to Use Agents

**Agents are OPTIONAL and secondary to superpowers skills.** Use agents only when:
- Superpowers skills explicitly recommend them (e.g., `/sp-brainstorm` suggests `/architect`)
- Complex projects benefit from specialized agent expertise
- Large-scale refactoring requires focused agent attention

**Default workflow:** Superpowers skills + hooks
**Optional workflow:** Agents (when superpowers recommends)

### Available Agents

Four specialized agents are available as tools when needed:

**1. Senior Architect Planner Agent** (Opus/Red):
- Creates PRDs, breaks down features into sprints
- TDD/SDD planning with 90%+ coverage baseline
- Use: BEFORE complex development work begins
- **Trigger:** `/architect` or when `/sp-brainstorm` recommends

**2. Senior Full-Stack Engineer Agent** (Sonnet/Blue):
- End-to-end implementation with strict TDD
- Aims for 80%+ test coverage, iterative refinement
- Use: For hands-on development requiring deep expertise
- **Trigger:** `/engineer` or when implementing complex features

**3. QA Gatekeeper Agent** (Sonnet/Green):
- Comprehensive code review, production readiness assessment
- CRITICAL/HIGH/MEDIUM priorities, test gap analysis
- Use: AFTER implementation, BEFORE commits
- **Trigger:** `/qa` or when `/sp-request-review` used

**4. Documentation Agent** (Sonnet/Yellow):
- Multi-layered docs, git excellence, conventional commits
- Use: AFTER changes for comprehensive documentation
- **Trigger:** `/docs` or when major documentation needed

### Agent Workflow Sequence (When Used)

```
1. PLANNING (Optional)
   └─> Senior Architect Planner
       - Create implementation plan
       - Break down requirements

2. DEVELOPMENT
   └─> Senior Full-Stack Engineer (Optional)
       - Implement with TDD
       - Comprehensive testing

3. QUALITY (Recommended)
   └─> QA Gatekeeper
       - Code review
       - Production readiness

4. DOCUMENTATION (Optional)
   └─> Documentation Agent
       - Update docs
       - Git commits
```

**Full agent details:** `.claude/agents/[agent-name].md`

**Remember:** Superpowers skills are your primary workflow. Agents are available as specialized tools when needed.

---

## 7. Quick Reference - Daily Commands

### Verification Commands

```bash
# Verify file changes
ls -la [file] && wc -l [file] && head -5 [file]

# Check production code exists
find src/ -type f ! -name "claude_*" | head -5

# Verify no TODOs in production
grep -r "TODO\|FIXME" src/ --exclude="claude_*" || echo "✓ Clean"

# Test production code
npm test || python -m pytest

# Check git staging
git status --porcelain | grep -v "claude_"
```

### Git Workflow Shortcuts

```bash
# Stage production files only
git add src/ tests/ config/ docs/ scripts/
git reset -- claude_*

# Commit with conventional format
git commit -m "feat: description" && git push origin main

# Verify staging before commit
git status --porcelain | grep -v "claude_"
```

### QC & Cleanup

```bash
# Run QC preflight
./scripts/qc_preflight.bat  # Windows
bash ./scripts/qc_preflight.sh  # Linux/Mac

# List production files
find src/ -type f ! -name "claude_*"

# Remove Claude temporary files
find . -name "claude_*" -exec rm -v {} \;
```

### GitHub Operations

```bash
# Authenticate quickly
/home/buckstrdr/quick_github_auth.sh

# List issues
gh issue list

# Close issue with resolution
gh issue close {NUMBER} --comment "$(cat .ian/issue_{NUMBER}_resolution.md)"

# Create PR
gh pr create --title "Title" --body "Description"
```

### Discovery Commands

```bash
# Find skills
ls .claude/skills/

# Find slash commands
ls .claude/commands/

# Check active hooks
cat .claude/settings.local.json

# View hook docs
cat .claude/hooks/README.md
```

### Error Handling

```bash
# Capture and log errors
command 2>&1 | tee error.log
[ $? -ne 0 ] && cat error.log && echo "Fixing production code..."
```

---

## Forbidden Actions

❌ Creating `fake_test.py`, `mock_api.js`, `demo_server.py`
❌ Simulating output without execution
❌ Skipping verification steps
❌ Committing `claude_*` files to git
❌ Working outside `src/`, `tests/`, `config/`
❌ Skipping superpowers skill checks before tasks
❌ Using agents as primary workflow (they're secondary)

---

**Remember:** Superpowers skills are your primary workflow. Use them FIRST for every task. Agents are available as specialized tools when superpowers recommends them or when complex work requires focused expertise.
