---
name: ts-resolve-mia-fork-before-compare
description: "Resolve 'mia'/'local inference lab' to their actual checkouts before comparing; don't diff the local fork's own deepseek_v4 code"
condition: "git (log|show).*deepseek_v4|deepseek_v4.*git (log|show)"
scope: "tool:bash"
---

When the user asks to verify against 'mia' or 'local inference lab', resolve those names to their real checkouts FIRST and diff against them, never substitute `git log`/`git show` on the local fork's own `vllm/models/deepseek_v4/...` files (that only inspects the tree you already have). 'mia' = repos under `/home/naadir/go/src/github.com/MiaAI-Lab` (confirm with `ghq list | grep -iE 'mia|inference'`). 'local inference lab' = the `local-inference-lab/vllm` remote on THIS repo — check it out as a worktree (`git worktree add <path> <remote-branch>` after `git fetch origin`/`git fetch <remote>`) for comparison. Then read or diff the relevant file in that checkout. Only after locating the external checkout may you run git commands against it.