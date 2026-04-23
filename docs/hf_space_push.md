# MediBill-Env — HuggingFace Space Push (one-shot)

Paste-ready. Takes ~5 minutes when you have your HF token.

## Prerequisites (one-time)

1. **HuggingFace account** — https://huggingface.co/join (free)
2. **Access token** — https://huggingface.co/settings/tokens
   - Click **"New token"**
   - Name: `medibill-push`
   - Role: **Write**
   - Copy the token (starts `hf_...`)

## One-command push (run locally on your Mac)

```bash
cd /Users/aks/METAHackthon2026

# 1. Install the OpenEnv CLI locally if it's not already there
pip install 'openenv-core[core]' -q

# 2. Log in to HuggingFace (paste token when prompted)
huggingface-cli login

# 3. Push the env as a HF Space
openenv push . --space-name <your-hf-username>/medibill-env
```

Replace `<your-hf-username>` with your actual HF username.

## What that does

- Packages this repo as a HF Space
- Uses our `openenv.yaml` (`name: medibill`, `app: medibill.server.app:app`, `port: 8000`)
- Uses our `Dockerfile` (already validated with `openenv validate .`)
- Starts a free-tier CPU Space at `https://huggingface.co/spaces/<your-username>/medibill-env`
- Build takes 5–10 minutes on HF's side
- Space auto-starts; `/health` endpoint reachable once build completes

## Verify after push

```bash
# Wait until HF finishes building (check the Space's "Logs" tab)
# Then:
curl -sf https://huggingface.co/spaces/<your-username>/medibill-env/health
# Should return: {"status":"healthy"}

# Metadata endpoint:
curl -s https://huggingface.co/spaces/<your-username>/medibill-env/metadata | python3 -m json.tool
```

## If the build fails

Common causes and fixes:

| Error in logs | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'medibill'` | Check `pyproject.toml` — should declare `openenv-medibill-env`. Already patched in commit 019ac6b. |
| `Port 8000 not bound` | Verify `Dockerfile`'s `CMD` ends with `--host 0.0.0.0 --port 8000`. Already patched. |
| `app_port mismatch` | Check YAML frontmatter in README.md — should say `app_port: 8000`. Already patched. |
| HF Space sleeps | Free-tier Spaces idle after 48h inactive — poke `/health` before the demo |

## Fallback — if HF push doesn't work

Local Docker image already built and verified:

```bash
docker run --rm -p 8000:8000 medibill:local
curl http://127.0.0.1:8000/health
# Returns: {"status":"healthy"}
```

Bring your laptop to the venue. That's the demo-reliability backstop.

## After HF Space is live

Add the URL to:

1. `README.md` — replace `[URL after push]` in the Round 2 header
2. `docs/hf_blog_draft.md` — replace `[URL]` in the HF blog
3. `docs/pitch_v1.md` — replace `[URL]` in slide 6

Commit and re-push the repo so the Space README on HuggingFace reflects the updated content.
