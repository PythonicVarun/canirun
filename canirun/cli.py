import logging

import click
from tabulate import tabulate

from . import __version__
from .human_readable import get_human_readable_size, get_human_readable_status
from .logic import ModelAnalyzer


@click.command()
@click.version_option(__version__, prog_name="canirun", message="%(prog)s v%(version)s")
@click.argument("model_id")
@click.option("--ctx", default=2048, help="Context length to simulate (default: 2048)")
@click.option(
    "--hf-token",
    default=None,
    help="Hugging Face API token for gated or private models",
)
@click.option("--verbose", is_flag=True, default=False, help="Enable detailed logging")
def main(model_id, ctx, hf_token, verbose):
    """
    LLM Memory Analyzer: Estimates if a model fits in your VRAM/RAM.
    """
    # Configure logging based on verbosity
    log_level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(format="%(levelname)s: %(message)s", level=log_level)

    analyzer = ModelAnalyzer(model_id, verbose=verbose, hf_token=hf_token)
    model_data = analyzer.fetch_model_data()

    if not model_data:
        click.secho(f"Error: Could not fetch data for '{model_id}'.", fg="red")
        return

    # Calculate memory requirements
    results = analyzer.calculate(model_data, ctx)
    table_data = [
        [
            r["quant"],
            get_human_readable_size(r["total_ram"]),
            get_human_readable_size(r["kv_cache"]),
            get_human_readable_status(r["status"]),
        ]
        for r in results
    ]

    click.echo("")
    click.secho(f" 🔍 ANALYSIS REPORT: {model_id} ", bg="blue", fg="white", bold=True)
    click.echo(f" {'Context Length':<15} : {ctx}")
    click.echo(f" {'Device':<15} : {analyzer.specs['name']}")
    click.echo(
        f" {'VRAM / RAM':<15} : {get_human_readable_size(analyzer.specs['vram'])} / {get_human_readable_size(analyzer.specs['ram'])}"
    )
    click.echo("")

    click.echo(
        tabulate(
            table_data,
            headers=["Quantization", "Total Est.", "KV Cache", "Compatibility"],
            tablefmt="fancy_grid",
            colalign=("left", "right", "right", "left"),
        )
    )
    click.echo("")


if __name__ == "__main__":
    main()
