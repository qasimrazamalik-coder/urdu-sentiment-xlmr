"""Gradio demo: paste Urdu or Roman Urdu text, get class probabilities and the detected script.

Starts without trained weights and then shows the exact commands to produce them.
"""

from __future__ import annotations

import argparse
import os
from typing import Any

import gradio as gr

from theme import CSS, build_theme
from urdu_sentiment.data import detect_script
from urdu_sentiment.predict import MAX_CHARS, Predictor
from urdu_sentiment.utils import load_config

EMPTY_STATE = """**No trained model found.** Train one, then restart this app:

```bash
make install
make data HF_ID=<owner/name> TEXT_COL=<text column> LABEL_COL=<label column>
make train
make eval
make app
```

To check the pipeline only, `make smoke` trains a tiny model on synthetic data
(its predictions are meaningless). See the README section "Choosing a dataset"."""

HOW_IT_WORKS = """Two models are scored on the same normalised text: a fine-tuned multilingual
transformer and a TF-IDF (word + character n-gram) linear classifier. Tables show the model
probabilities. The linear SVM option reports softmax of margins, which is not calibrated.
The script label comes from the share of Arabic-script letters in the text."""

LIMITATIONS = """- Domain shift: the models only know the domain of their training data. Product
  reviews will not transfer to news, politics or poetry.
- Sarcasm and negation are frequently misread, in both scripts.
- Code-mixing (Urdu with English words, or mixed scripts) is under-represented in most
  datasets and is often scored with lower confidence.
- Roman Urdu has no standard spelling. Unseen spellings can change the prediction.
- Roman Urdu cannot be told apart from English by the script check alone.
- Probabilities from a fine-tuned transformer are often over-confident. Do not use this
  for decisions about people."""

APP_CSS = (
    CSS
    + """
#text-input textarea {
  unicode-bidi: plaintext; text-align: start;
  font-family: "Noto Naskh Arabic", "Noto Sans Arabic", "IBM Plex Sans", system-ui, sans-serif;
  font-size: 1.1rem; line-height: 1.7;
}
"""
)


def validate(text: str | None) -> str | None:
    """Return an error message for unusable input, or None when the text is fine."""
    if text is None or not text.strip():
        return "Enter some text first."
    if len(text) > MAX_CHARS:
        return f"Text is {len(text)} characters; the limit is {MAX_CHARS}."
    return None


def format_result(result: dict[str, Any]) -> tuple[str, str, list[list[str]], list[str]]:
    """Turn a Predictor result into status text, script text and a probability table."""
    labels = result["labels"]
    headers = ["Class", *result["models"].keys()]
    rows = [
        [label, *[f"{probs[i]:.3f}" for probs in result["models"].values()]]
        for i, label in enumerate(labels)
    ]
    lines = []
    for name, probs in result["models"].items():
        best = max(range(len(labels)), key=probs.__getitem__)
        lines.append(f"{name}: **{labels[best]}** ({probs[best]:.3f})")
    return "  \n".join(lines), result["script"], rows, headers


def build_app(predictor: Predictor | None) -> gr.Blocks:
    """Build the UI. `predictor` may be None or not ready: the app then shows the empty state."""
    ready = predictor is not None and predictor.ready
    with gr.Blocks(title="Urdu sentiment") as demo:
        gr.Markdown("# Urdu and Roman Urdu sentiment")
        gr.Markdown(
            "Paste a review or comment in Urdu script or Roman Urdu. "
            f"Limit: {MAX_CHARS} characters."
        )
        if not ready:
            gr.Markdown(EMPTY_STATE, elem_classes=["notice"])
        with gr.Row():
            with gr.Column():
                box = gr.Textbox(
                    label="Text",
                    lines=5,
                    max_length=MAX_CHARS,
                    elem_id="text-input",
                    placeholder="یہ فلم بہت اچھی تھی   /   yeh film bohat achi thi",
                )
                run = gr.Button("Classify", variant="primary", interactive=ready)
            with gr.Column():
                status = gr.Markdown("Result appears here.")
                script = gr.Textbox(label="Detected script", interactive=False)
                table = gr.Dataframe(
                    headers=["Class", "Probability"], label="Class probabilities", interactive=False
                )

        def classify(text: str | None):
            error = validate(text)
            if error:
                return error, detect_script(text or "") if text else "", gr.update()
            if not ready:
                return EMPTY_STATE, "", gr.update()
            status_text, script_text, rows, headers = format_result(predictor.predict(text))
            return status_text, script_text, gr.update(value=rows, headers=headers)

        run.click(classify, inputs=box, outputs=[status, script, table])
        box.submit(classify, inputs=box, outputs=[status, script, table])
        with gr.Accordion("How it works", open=False):
            gr.Markdown(HOW_IT_WORKS)
        with gr.Accordion("Limitations", open=False):
            gr.Markdown(LIMITATIONS)
    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default=os.environ.get("URDU_SENTIMENT_CONFIG", "configs/default.yaml")
    )
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    predictor = Predictor(load_config(args.config))
    build_app(predictor).launch(
        server_name="127.0.0.1", server_port=args.port, theme=build_theme(), css=APP_CSS
    )


if __name__ == "__main__":
    main()
