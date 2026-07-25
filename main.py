from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

from src.bpmn_pipeline import BPMNKnowledgeError, BPMNKnowledgePipeline
from src.evaluation import EVALUATION_FIELD, evaluate
from src.pipeline import ExtractionPipeline, PipelineError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Extract structured knowledge from a PDTA PDF, evaluate an extraction, "
            "and derive BPMN-oriented lanes, activities, and resources."
        )
    )

    extraction = parser.add_argument_group("stage 1: PDTA extraction")
    extraction.add_argument("--pdf", type=Path, help="Input PDF path.")
    extraction.add_argument("--prompt", type=Path, help="Stage-1 prompt JSON path.")
    extraction.add_argument("--config", type=Path, help="LLM configuration JSON path.")
    extraction.add_argument(
        "--source-page-start",
        type=int,
        default=1,
        help="Inclusive first PDF page to process. Page numbering starts at 1.",
    )
    extraction.add_argument(
        "--source-page-end",
        type=int,
        default=None,
        help="Optional inclusive last PDF page to process.",
    )
    extraction.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional explicit Stage-1 output JSON path.",
    )
    extraction.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="Output directory (default: output).",
    )

    evaluation = parser.add_argument_group("evaluation")
    evaluation.add_argument(
        "--gold-standard",
        "--gold",
        dest="gold_standard",
        type=Path,
        help="Gold-standard JSON path.",
    )
    evaluation.add_argument(
        "--prediction",
        type=Path,
        help=(
            "Extracted Stage-1 JSON to evaluate. When omitted after Stage 1, "
            "the newly generated data JSON is evaluated."
        ),
    )
    evaluation.add_argument(
        "--field",
        default=EVALUATION_FIELD,
        choices=("evidence", "interpreted_values", "interpreted_value"),
        help=f"Field used for evaluation (default: {EVALUATION_FIELD}).",
    )
    evaluation.add_argument(
        "--model",
        default=None,
        help="Optional model label for the aggregate evaluation CSV.",
    )

    bpmn = parser.add_argument_group("stage 2: BPMN-oriented knowledge")
    bpmn.add_argument(
        "--bpmn-source",
        type=Path,
        help=(
            "Structured Stage-1 JSON used to extract lanes, activities, and "
            "resources. When omitted after Stage 1, the newly generated JSON is used."
        ),
    )
    bpmn.add_argument(
        "--bpmn-prompt",
        type=Path,
        help="BPMN-oriented prompt JSON path.",
    )
    bpmn.add_argument(
        "--bpmn-config",
        type=Path,
        help=(
            "LLM configuration for Stage 2. When omitted, --config is reused."
        ),
    )
    bpmn.add_argument(
        "--bpmn-output",
        type=Path,
        default=None,
        help="Optional explicit detailed BPMN-oriented output JSON path.",
    )
    bpmn.add_argument(
        "--bpmn-summary-output",
        type=Path,
        default=None,
        help="Optional explicit compact BPMN-oriented summary JSON path.",
    )

    return parser


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    extraction_values = (args.pdf, args.prompt, args.config)
    extraction_requested = any(value is not None for value in extraction_values)
    if extraction_requested and not all(value is not None for value in extraction_values):
        parser.error("--pdf, --prompt and --config must be supplied together.")

    evaluation_requested = args.gold_standard is not None or args.prediction is not None
    if args.prediction is not None and args.gold_standard is None:
        parser.error("--gold-standard is required when --prediction is supplied.")

    bpmn_requested = args.bpmn_prompt is not None or args.bpmn_source is not None
    if args.bpmn_source is not None and args.bpmn_prompt is None:
        parser.error("--bpmn-prompt is required when --bpmn-source is supplied.")
    if args.bpmn_prompt is not None and args.bpmn_config is None and args.config is None:
        parser.error("Provide --bpmn-config or --config for Stage 2.")

    if not extraction_requested and not evaluation_requested and not bpmn_requested:
        parser.error("No operation requested.")

    if args.field != "evidence" and args.gold_standard is not None:
        print(
            f"Warning: evaluation field is '{args.field}', not the default 'evidence'.",
            file=sys.stderr,
        )


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    validate_args(parser, args)

    started_at = datetime.now().astimezone()
    started_counter = time.perf_counter()

    print()
    print("*** PROGRAM START ***")
    print(f"Start date and time: {started_at.isoformat(timespec='seconds')}")
    print()

    try:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        args.output_dir.joinpath(".gitkeep").touch(exist_ok=True)
    except OSError as exc:
        print(f"Unable to initialise output directory: {exc}", file=sys.stderr)
        return 1

    stage1_data_path: Path | None = None
    extraction_requested = all(
        value is not None for value in (args.pdf, args.prompt, args.config)
    )

    if extraction_requested:
        pipeline = ExtractionPipeline(
            pdf_path=args.pdf,
            prompt_path=args.prompt,
            config_path=args.config,
            source_page_start=args.source_page_start,
            source_page_end=args.source_page_end,
            output_path=args.output,
            output_dir=args.output_dir,
        )
        try:
            stage1_data_path, metadata_path = pipeline.run()
        except PipelineError as exc:
            print(f"Stage-1 extraction failed: {exc}", file=sys.stderr)
            return 1

        print(f"Stage-1 data saved to: {stage1_data_path}")
        print(f"Stage-1 metadata saved to: {metadata_path}")

    prediction_path = args.prediction or stage1_data_path
    if args.gold_standard is not None:
        if prediction_path is None:
            print("No prediction JSON is available for evaluation.", file=sys.stderr)
            return 1

        try:
            details_path, summary_path, aggregate = evaluate(
                gold_standard_path=args.gold_standard,
                prediction_path=prediction_path,
                output_dir=args.output_dir,
                field=args.field,
                model_name=args.model,
            )
        except ValueError as exc:
            print(f"Evaluation failed: {exc}", file=sys.stderr)
            return 1

        print(f"Evaluation details saved to: {details_path}")
        print(f"Evaluation summary updated: {summary_path}")
        print(
            "Aggregate metrics: "
            f"accuracy={aggregate['accuracy']:.4f}, "
            f"precision={aggregate['precision']:.4f}, "
            f"recall={aggregate['recall']:.4f}, "
            f"F1={aggregate['f1_score']:.4f}"
        )

    bpmn_requested = args.bpmn_prompt is not None
    if bpmn_requested:
        bpmn_source = args.bpmn_source or stage1_data_path
        if bpmn_source is None:
            print(
                "No structured Stage-1 JSON is available for BPMN-oriented extraction.",
                file=sys.stderr,
            )
            return 1

        bpmn_config = args.bpmn_config or args.config
        assert bpmn_config is not None

        bpmn_pipeline = BPMNKnowledgePipeline(
            source_json_path=bpmn_source,
            prompt_path=args.bpmn_prompt,
            config_path=bpmn_config,
            output_path=args.bpmn_output,
            summary_output_path=args.bpmn_summary_output,
            output_dir=args.output_dir,
        )
        try:
            (
                bpmn_detailed_path,
                bpmn_summary_path,
                bpmn_metadata_path,
            ) = bpmn_pipeline.run()
        except BPMNKnowledgeError as exc:
            print(f"Stage-2 BPMN-oriented extraction failed: {exc}", file=sys.stderr)
            return 1

        print(f"BPMN-oriented detailed data saved to: {bpmn_detailed_path}")
        print(f"BPMN-oriented summary saved to: {bpmn_summary_path}")
        print(f"BPMN-oriented metadata saved to: {bpmn_metadata_path}")

    completed_at = datetime.now().astimezone()
    elapsed_minutes = (time.perf_counter() - started_counter) / 60
    print(f"End date and time: {completed_at.isoformat(timespec='seconds')}")
    print(f"Elapsed time: {elapsed_minutes:.2f} minutes")
    print("*** PROGRAM END ***")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
