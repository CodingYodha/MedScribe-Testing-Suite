import json

required_keys = {"medicines","diseases","symptoms","tests","instructions"}
required_med_keys = {"name","dosage","frequency","duration","route","timing"}

with open("clean_prescription_dataset.jsonl") as f:
    for i, line in enumerate(f, 1):
        try:
            row = json.loads(line)
            output = json.loads(row["output"].replace(" AAA", ""))
            
            # schema validation
            assert set(output.keys()) == required_keys
            
            # medicine field validation
            for med in output["medicines"]:
                assert set(med.keys()) == required_med_keys
            
        except Exception as e:
            print(f" Row {i}: {e}")
