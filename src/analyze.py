import json

with open('results', 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        try:
            record = json.loads(line)
            if 'record_type' in record and record['record_type'] == 'strategy':
                print(f"STRATEGY ID: {record['data'].get('id')}")
                print(f"ITERATION: {record['data'].get('metadata', {}).get('iteration')}")
                print(f"ACCURACY: {record['data'].get('metadata', {}).get('dev_accuracy')}")
                print(f"PROMPT: {repr(record['data'].get('prompt_template'))}")
                print('-'*50)
            elif 'id' in record and 'prompt_template' in record:
                print(f"STRATEGY ID: {record.get('id')}")
                print(f"ITERATION: {record.get('metadata', {}).get('iteration')}")
                print(f"ACCURACY: {record.get('metadata', {}).get('dev_accuracy')}")
                print(f"PROMPT: {repr(record.get('prompt_template'))}")
                print('-'*50)
        except Exception as e:
            pass
