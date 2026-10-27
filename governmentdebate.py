"""
Government vs Opposition: a two-LLM political debate (Gradio).

- LLM #1 plays the GOVERNMENT, LLM #2 plays the OPPOSITION.
- The country is given by the user in the UI.
- Free LLMs: Groq free tier (needs a free API key) or Ollama (fully local).
- Maximum 10 rounds (1 round = one government turn + one opposition turn).
"""
import os
import re

import gradio as gr
from openai import OpenAI

MAX_ROUNDS = 10

PROVIDERS = {
    "Groq (free API key)": {
        "base_url": "https://api.groq.com/openai/v1",
        "env_key": "GROQ_API_KEY",
        "gov_model": "openai/gpt-oss-20b",
        "opp_model": "openai/gpt-oss-120b",
    },
    "Ollama (local, free)": {
        "base_url": "http://localhost:11434/v1",
        "env_key": "OLLAMA_API_KEY",  # not needed, any string works
        "gov_model": "llama3.2",
        "opp_model": "qwen2.5",
    },
}

RULES = """
RULES:
- Support every argument with specific, real data and statistics (GDP growth, inflation,
  unemployment, debt, poverty, education/health spending, etc.) and name the source and
  year (e.g. "World Bank, 2023", "IMF", "national statistics office").
- NEVER invent numbers. If you are not sure of an exact figure, say "approximately" or
  use a figure you are confident about. Accuracy matters more than volume.
- Directly respond to the other side's latest points, then add one new argument.
- Be respectful and professional; no personal attacks, no insults.
- Keep each turn under 150 words. Do not use headings or bullet lists.
"""

# --------------------------------------------------------------------------- #
# Debate logic
# --------------------------------------------------------------------------- #


def system_prompt(side: str, country: str, topic: str) -> str:
    focus = f" Focus especially on: {topic}." if topic.strip() else ""
    if side == "gov":
        role = (
            f"You are a senior spokesperson for the GOVERNMENT of {country} in a televised "
            f"political debate. Defend the government's record, policies and future plans."
        )
    else:
        role = (
            f"You are the leader of the OPPOSITION in {country} in a televised political "
            f"debate. Criticise the government's record and policies and present credible "
            f"alternatives."
        )
    return role + focus + "\n" + RULES


def clean(text: str) -> str:
    """Remove hidden reasoning blocks some models emit."""
    return re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL).strip()


def build_messages(side, country, topic, transcript, round_no, total_rounds):
    """Build the chat history from the point of view of the current speaker."""
    msgs = [{"role": "system", "content": system_prompt(side, country, topic)}]
    if side == "gov":
        # Government always opens, so give it a starting user message.
        msgs.append({"role": "user", "content": f"The debate about {country} begins now."})
    for who, text in transcript:
        msgs.append({"role": "assistant" if who == side else "user", "content": text})

    if round_no == 1 and side == "gov":
        instr = "Give your opening statement."
    elif round_no == total_rounds and side == "opp":
        instr = "This is the final turn. Respond briefly, then give your closing statement."
    elif round_no == total_rounds:
        instr = "Respond to the opposition, then give your closing statement."
    else:
        instr = "Respond to the latest points, then add one new argument."
    msgs[-1] = {"role": msgs[-1]["role"], "content": msgs[-1]["content"] + "\n\n" + instr}
    return msgs


def ask(client, model, messages):
    resp = client.chat.completions.create(
        model=model, messages=messages, temperature=0.7, max_tokens=500
    )
    return clean(resp.choices[0].message.content)


# --------------------------------------------------------------------------- #
# UI helpers
# --------------------------------------------------------------------------- #

NAMES = {"gov": "Government", "opp": "Opposition"}


def status_html(text: str, side: str = "idle") -> str:
    return f'<div class="status {side}"><span class="dot"></span><span>{text}</span></div>'


IDLE_STATUS = status_html("Enter a country on the left, then start the debate.")

HERO = """
<div class="hero">
  <div class="hero-bar"><span class="bar-gov"></span><span class="bar-opp"></span></div>
  <h1>Government versus Opposition</h1>
  <p>Two AI models debate the politics of any country you choose. Each side is told to
  back its claims with real statistics and name the source.</p>
  <div class="hero-sides">
    <span class="side gov">Government</span>
    <span class="side opp">Opposition</span>
  </div>
</div>
"""

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,400;8..60,600;8..60,700&display=swap');

:root {
  --ink: #14213d;
  --muted: #5b6475;
  --gov: #1d4e89;
  --gov-soft: #e8f0fa;
  --gov-edge: #bcd1ee;
  --opp: #a4243b;
  --opp-soft: #fbebed;
  --opp-edge: #eebfc7;
  --serif: 'Source Serif 4', Georgia, 'Times New Roman', serif;
}

.gradio-container { max-width: 1180px !important; margin: 0 auto !important; }

/* ---------- hero ---------- */
.hero { padding: 8px 4px 4px; }
.hero-bar { display: flex; height: 6px; border-radius: 3px; overflow: hidden; margin-bottom: 22px; }
.hero-bar .bar-gov { flex: 1; background: var(--gov); }
.hero-bar .bar-opp { flex: 1; background: var(--opp); }
.hero h1 {
  font-family: var(--serif); font-weight: 700; font-size: 2.5rem; line-height: 1.1;
  letter-spacing: -0.015em; color: var(--ink); margin: 0 0 10px;
}
.hero p { color: var(--muted); font-size: 1.02rem; max-width: 62ch; margin: 0 0 16px; line-height: 1.55; }
.hero-sides { display: flex; gap: 10px; }
.hero .side {
  font-size: 0.9rem; font-weight: 600; padding: 5px 14px; border-radius: 999px; border: 1px solid;
}
.hero .side.gov { color: var(--gov); background: var(--gov-soft); border-color: var(--gov-edge); }
.hero .side.opp { color: var(--opp); background: var(--opp-soft); border-color: var(--opp-edge); }

/* ---------- status line ---------- */
.status {
  display: flex; align-items: center; gap: 10px; padding: 11px 16px; border-radius: 10px;
  font-weight: 600; font-size: 0.95rem; border: 1px solid #d5dae3; background: #fff; color: var(--ink);
}
.status .dot { width: 10px; height: 10px; border-radius: 50%; background: #9aa3b5; flex: none; }
.status.gov { background: var(--gov-soft); border-color: var(--gov-edge); color: var(--gov); }
.status.gov .dot { background: var(--gov); animation: pulse 1.2s ease-in-out infinite; }
.status.opp { background: var(--opp-soft); border-color: var(--opp-edge); color: var(--opp); }
.status.opp .dot { background: var(--opp); animation: pulse 1.2s ease-in-out infinite; }
.status.done .dot { background: #2f7d4f; }
.status.done { border-color: #bfdcca; background: #eef8f1; color: #24633e; }
.status.error { background: #fff3e6; border-color: #f2cfa5; color: #8a4b08; }
.status.error .dot { background: #d9822b; }
@keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.35; } }
@media (prefers-reduced-motion: reduce) { .status .dot { animation: none !important; } }

/* ---------- debate transcript ---------- */
#debate { border-radius: 12px; }
#debate .message { font-family: var(--serif); font-size: 1.05rem; line-height: 1.62; }
#debate .message.bot {
  background: var(--gov-soft) !important; border: 1px solid var(--gov-edge) !important;
  border-left: 5px solid var(--gov) !important; color: var(--ink) !important;
}
#debate .message.user {
  background: var(--opp-soft) !important; border: 1px solid var(--opp-edge) !important;
  border-right: 5px solid var(--opp) !important; color: var(--ink) !important;
}
#debate .message strong { font-family: system-ui, sans-serif; font-size: 0.92rem; }
#debate .message.bot strong { color: var(--gov); }
#debate .message.user strong { color: var(--opp); }

/* ---------- control panel ---------- */
#controls { padding: 4px; }
#start-btn { font-weight: 600; }
.note { color: var(--muted); font-size: 0.85rem; line-height: 1.5; margin-top: 4px; }
"""

THEME = gr.themes.Base(
    primary_hue="blue",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Public Sans"), "system-ui", "sans-serif"],
).set(
    body_background_fill="#eef1f5",
    block_background_fill="#ffffff",
    block_border_width="1px",
    block_radius="12px",
    input_radius="8px",
    button_large_radius="8px",
    button_primary_background_fill="#14213d",
    button_primary_background_fill_hover="#1f3159",
    button_primary_text_color="#ffffff",
)

# Keep the page in light mode so the debate colours always look right.
FORCE_LIGHT_JS = "() => { document.body.classList.remove('dark'); }"


def run_debate(country, topic, rounds, provider, api_key, gov_model, opp_model):
    country = (country or "").strip()
    if not country:
        raise gr.Error("Please enter a country.")

    rounds = max(1, min(int(rounds), MAX_ROUNDS))  # hard cap at 10
    cfg = PROVIDERS[provider]
    key = (api_key or "").strip() or os.getenv(cfg["env_key"], "") or "ollama"
    if provider.startswith("Groq") and key == "ollama":
        raise gr.Error("Please enter your free Groq API key (or set GROQ_API_KEY).")

    client = OpenAI(base_url=cfg["base_url"], api_key=key)
    models = {"gov": gov_model.strip(), "opp": opp_model.strip()}

    transcript = []  # [(side, text)]
    chat = []  # what the Gradio Chatbot displays
    yield chat, status_html(f"Starting a {rounds}-round debate about {country}.")

    for r in range(1, rounds + 1):
        for side in ("gov", "opp"):
            yield chat, status_html(f"Round {r} of {rounds}: {NAMES[side]} is speaking.", side)
            try:
                msgs = build_messages(side, country, topic, transcript, r, rounds)
                text = ask(client, models[side], msgs)
            except Exception as e:  # network / auth / model errors
                chat.append({"role": "assistant", "content": f"**Error:** {e}"})
                yield chat, status_html("The debate stopped because of an error.", "error")
                return
            transcript.append((side, text))
            # Government on the left (assistant), opposition on the right (user)
            chat.append({
                "role": "assistant" if side == "gov" else "user",
                "content": f"**{NAMES[side]}, round {r}**\n\n{text}",
            })
            yield chat, status_html(f"Round {r} of {rounds}: {NAMES[side]} has spoken.", side)

    yield chat, status_html(f"The debate on {country} is finished after {rounds} rounds.", "done")


def on_provider_change(provider):
    cfg = PROVIDERS[provider]
    return cfg["gov_model"], cfg["opp_model"]


# --------------------------------------------------------------------------- #
# Interface
# --------------------------------------------------------------------------- #

with gr.Blocks(title="Government versus Opposition", theme=THEME, css=CSS, js=FORCE_LIGHT_JS) as demo:
    gr.HTML(HERO)

    with gr.Row(equal_height=False):
        with gr.Column(scale=2, min_width=320, elem_id="controls"):
            country = gr.Textbox(label="Country", placeholder="For example: Egypt")
            gr.Examples(
                examples=[["Egypt"], ["France"], ["Brazil"], ["India"], ["Japan"]],
                inputs=country,
                label="Or pick one",
            )
            topic = gr.Textbox(label="Focus topic (optional)", placeholder="For example: economy, healthcare")
            rounds = gr.Slider(1, MAX_ROUNDS, value=5, step=1, label="Rounds (maximum 10)")
            api_key = gr.Textbox(
                label="Groq API key",
                type="password",
                placeholder="gsk_... (or set GROQ_API_KEY)",
            )
            with gr.Accordion("Model settings", open=False):
                provider = gr.Dropdown(list(PROVIDERS), value="Groq (free API key)", label="Free LLM provider")
                gov_model = gr.Textbox(label="Government model", value=PROVIDERS["Groq (free API key)"]["gov_model"])
                opp_model = gr.Textbox(label="Opposition model", value=PROVIDERS["Groq (free API key)"]["opp_model"])
            with gr.Row():
                start = gr.Button("Start debate", variant="primary", elem_id="start-btn", scale=3)
                stop = gr.Button("Stop", variant="secondary", scale=1)
            gr.HTML('<div class="note">AI models can get statistics wrong. Check any figure before you rely on it.</div>')

        with gr.Column(scale=5):
            status = gr.HTML(IDLE_STATUS)
            chatbot = gr.Chatbot(
                type="messages",
                height=600,
                show_label=False,
                elem_id="debate",
                placeholder="The debate will appear here.",
            )

    provider.change(on_provider_change, provider, [gov_model, opp_model])
    run = start.click(
        run_debate,
        [country, topic, rounds, provider, api_key, gov_model, opp_model],
        [chatbot, status],
    )
    stop.click(None, None, None, cancels=[run])

if __name__ == "__main__":
    demo.launch()
