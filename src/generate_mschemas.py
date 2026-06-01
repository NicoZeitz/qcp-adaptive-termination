#!/usr/bin/env python3
"""
Script to generate m-schemas for all database files in a specified folder.
Supports SQLite (.db, .sqlite), MySQL, and PostgreSQL databases.
"""

import argparse
import logging
from pathlib import Path
from sqlalchemy import create_engine
from stages.common.mschema.schema_engine import SchemaEngine


def setup_logging(verbose: bool = False):
    """Set up logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="[%(levelname)s] %(message)s"
    )


def generate_mschema_from_sqlite(
    db_path: Path,
    output_dir: Path,
    example_num: int = 3,
    show_type_detail: bool = False
):
    """Generate m-schema for a SQLite database file."""
    db_name = db_path.stem
    logging.info(f"Processing: {db_path}")

    try:
        # Create SQLAlchemy engine for SQLite
        engine = create_engine(f"sqlite:///{db_path}")

        # Create SchemaEngine instance
        schema_engine = SchemaEngine(
            engine=engine,
            db_name=db_name,
            sample_rows_in_table_info=example_num
        )

        # Generate m-schema string
        mschema_str = schema_engine.mschema.to_mschema(
            example_num=example_num,
            show_type_detail=show_type_detail
        )

        # Save m-schema as text file
        output_txt = output_dir / f"{db_name}_mschema.txt"
        with open(output_txt, "w", encoding="utf-8") as f:
            f.write(mschema_str)
        logging.info(f"  ✓ Saved text: {output_txt}")

        # Save m-schema as JSON file
        output_json = output_dir / f"{db_name}_mschema.json"
        schema_engine.mschema.save(str(output_json))
        logging.info(f"  ✓ Saved JSON: {output_json}")

        return True

    except Exception as e:
        logging.error(f"  ✗ Error processing {db_path}: {e}")
        return False


def generate_mschemas_for_folder(
    folder_path: Path,
    output_dir: Path | None = None,
    extensions: list[str] = [".db", ".sqlite"],
    example_num: int = 3,
    show_type_detail: bool = False,
    recursive: bool = False
):
    """Generate m-schemas for all database files in a folder."""

    if not folder_path.exists():
        logging.error(f"Folder does not exist: {folder_path}")
        return

    if not folder_path.is_dir():
        logging.error(f"Path is not a directory: {folder_path}")
        return

    # Set output directory
    if output_dir is None:
        output_dir = folder_path / "mschemas"

    output_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Output directory: {output_dir}")
    logging.info(f"Looking for files with extensions: {', '.join(extensions)}")

    # Find all database files
    db_files = []
    if recursive:
        for ext in extensions:
            db_files.extend(folder_path.rglob(f"*{ext}"))
    else:
        for ext in extensions:
            db_files.extend(folder_path.glob(f"*{ext}"))

    if not db_files:
        logging.warning(f"No database files found in {folder_path}")
        return

    logging.info(f"Found {len(db_files)} database file(s)")
    logging.info("")

    # Process each database file
    success_count = 0
    for db_file in sorted(db_files):
        if generate_mschema_from_sqlite(
            db_file,
            output_dir,
            example_num,
            show_type_detail
        ):
            success_count += 1
        logging.info("")

    # Summary
    logging.info("=" * 60)
    logging.info(f"Summary: {success_count}/{len(db_files)} databases processed successfully")
    logging.info(f"Output directory: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        prog="Generate M-Schemas",
        description="Generate m-schemas for all database files in a specified folder"
    )

    parser.add_argument(
        "--folder",
        type=str,
        required=True,
        help="Path to folder containing database files"
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory for m-schema files (default: <folder>/mschemas)"
    )

    parser.add_argument(
        "--extensions",
        type=str,
        nargs="+",
        default=[".db", ".sqlite"],
        help="Database file extensions to process (default: .db .sqlite)"
    )

    parser.add_argument(
        "--example-num",
        type=int,
        default=3,
        help="Number of example values to include per column (default: 3)"
    )

    parser.add_argument(
        "--show-type-detail",
        action="store_true",
        help="Show detailed type information (e.g., VARCHAR(255) instead of VARCHAR)"
    )

    parser.add_argument(
        "--recursive",
        "-r",
        action="store_true",
        help="Search for database files recursively in subdirectories"
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    setup_logging(args.verbose)

    folder_path = Path(args.folder)
    output_dir = Path(args.output) if args.output else None

    generate_mschemas_for_folder(
        folder_path=folder_path,
        output_dir=output_dir,
        extensions=args.extensions,
        example_num=args.example_num,
        show_type_detail=args.show_type_detail,
        recursive=args.recursive
    )


if __name__ == "__main__":
    main()
