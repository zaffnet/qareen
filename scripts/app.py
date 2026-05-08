import argparse
import html
import logging
from pathlib import Path
from typing import Any

import gradio as gr

from qareen.models import Settings
from qareen.retrieving.chroma_retriever import ChromaRetriever

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def create_demo(settings: Settings) -> Any:
    dataset_name = settings.dataset_path
    if not dataset_name:
        dataset_name = "data/marqo_30k"  # fallback or make it required

    model_id = settings.embedding_models[0]
    embedding_model = settings.create_embedding_model(model_id)
    embedding_model.load_model()

    retriever = ChromaRetriever(embedding_model=embedding_model, settings=settings)

    available_alphas = retriever.list_available_alphas(
        dataset_name=dataset_name, model_id=model_id, environment=settings.environment
    )

    if not available_alphas:
        available_alphas = [0.0, 0.5, 1.0]  # Fallback if no collections found

    def search(
        image: str | None, text: str | None, alpha: float, k: float, mmr_lambda: float
    ) -> str:
        try:
            # find the closest available alpha collection
            closest_alpha = min(available_alphas, key=lambda x: abs(x - alpha))

            vectorstore = retriever.get_vectorstore(
                dataset_name=dataset_name,
                model_id=model_id,
                alpha=closest_alpha,
                environment=settings.environment,
            )

            # The MMR lambda is passed to retriever
            docs_with_scores = retriever.query_multimodal(
                vectorstore=vectorstore,
                image=image,
                text=text if text else None,
                alpha=alpha,  # actual query alpha
                k=int(k),
                fetch_k=int(k) * 4,
                mmr_lambda=mmr_lambda,
            )

            if not docs_with_scores:
                return "No results found."

            html_output = "<div style='display: flex; flex-direction: column; gap: 20px;'>"
            for i, (doc, score) in enumerate(docs_with_scores):
                style = "border:1px solid #ccc; padding:10px; border-radius:5px;"
                html_output += f"<div style='{style}'>"
                html_output += f"<h4>Result {i + 1} (Score: {score:.4f})</h4>"

                # Check if document has an image
                has_image = doc.metadata.get("has_image", False)
                if has_image and "image" in doc.metadata:
                    # Depending on how image is stored in metadata, we might display it.
                    # Usually we just have text. Let's show text at least.
                    pass

                text_content = html.escape(doc.page_content)
                html_output += f"<p>{text_content}</p>"
                escaped_meta = html.escape(str(doc.metadata))
                html_output += f"<pre style='font-size: 0.8em; color: #666;'>{escaped_meta}</pre>"
                html_output += "</div>"
            html_output += "</div>"

            return html_output

        except Exception as e:
            logger.exception("Search failed")
            return f"Error: {str(e)}"

    with gr.Blocks(title="Qareen MMR Multimodal Search") as demo:
        gr.Markdown("# Qareen: Multimodal MMR Retrieval")
        gr.Markdown(f"Dataset: **{dataset_name}** | Model: **{model_id}**")

        with gr.Row():
            with gr.Column():
                input_image = gr.Image(type="filepath", label="Query Image (Optional)")
                input_text = gr.Textbox(label="Query Text", placeholder="Enter search text...")

                alpha_slider = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    step=0.1,
                    value=0.5,
                    label="Alpha (0.0 = Text Only, 1.0 = Image Only)",
                )

                mmr_lambda_slider = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    step=0.1,
                    value=0.5,
                    label="MMR Lambda (1.0 = Relevance Only, 0.0 = Diversity Only)",
                )

                k_slider = gr.Slider(
                    minimum=1, maximum=20, step=1, value=5, label="Number of Results (k)"
                )

                search_btn = gr.Button("Search", variant="primary")

            with gr.Column():
                output_html = gr.HTML(label="Results")

        search_btn.click(
            fn=search,
            inputs=[input_image, input_text, alpha_slider, k_slider, mmr_lambda_slider],
            outputs=[output_html],
        )

    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Qareen Gradio App")
    parser.add_argument("--config", type=Path, help="Path to .env config file")
    parser.add_argument("--dataset-name", type=str, help="Override dataset name")
    parser.add_argument("--share", action="store_true", help="Share Gradio app publicly")
    args = parser.parse_args()

    settings = Settings(_env_file=str(args.config)) if args.config else Settings()
    if args.dataset_name:
        settings.dataset_path = args.dataset_name

    demo = create_demo(settings)
    demo.launch(share=args.share)


if __name__ == "__main__":
    main()
