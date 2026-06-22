import json
import shutil
import time
from pathlib import Path

from scripts.generate_mschemas import generate_mschema_from_sqlite
from splits.splitter import SplitType, create_agents

if __name__ == "__main__":
    dataset_dir = Path("../data/bird/dataset")
    dev_databases_dir = Path("../data/bird/dev/dev_databases")
    output_base_dir = Path("../data/bird/dataset_own")

    start_time = time.time()

    for question_file in sorted(dataset_dir.glob("*.json")):
        qid = question_file.stem  # e.g. "q1188"

        with open(question_file) as f:
            data = json.load(f)

        db_id = data["extra"]["db_id"]
        db_path = dev_databases_dir / db_id / f"{db_id}.sqlite"

        if not db_path.exists():
            print(f"Database not found: {db_path}, skipping {qid}")
            continue

        print(f"Processing {qid} (db: {db_id})")

        output_path = output_base_dir / db_id / qid
        output_path.mkdir(parents=True, exist_ok=True)

        # Copy question file to dataset_own root
        shutil.copy(question_file, output_base_dir / question_file.name)

        # Create agent splits
        # date = datetime.datetime.now().strftime("%Y-%m-%d")
        # output_path = f"{output_path}/{date}-{uuid.uuid4()}"
        data_agents = create_agents(
            output_path=str(output_path),
            database_url=f"sqlite:///{db_path.resolve()}",
            agent_amount=4,
            always_included_columns_mapping={},
            split_type=SplitType.MIXED,
            force_splits=False,
            max_splits_per_table=4,
            seed=42,
        )

        # Generate mschemas for agent databases and the original database
        mschemas_dir = output_path / "mschemas"
        mschemas_dir.mkdir(parents=True, exist_ok=True)

        for agent in data_agents:
            generate_mschema_from_sqlite(Path(agent.path), mschemas_dir)

        generate_mschema_from_sqlite(db_path, mschemas_dir)

    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")
