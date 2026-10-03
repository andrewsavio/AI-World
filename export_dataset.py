"""Turns what the students and the Professor learned into a fine-tuning dataset (chat-style JSONL).

Run:  python export_dataset.py        ->  training-data.jsonl
Each line is one example: a question and the answer the student's own notes support. Feed the file to a LoRA
fine-tuning tool (for example Unsloth or Hugging Face TRL on a free Colab GPU), merge, convert to GGUF, then
register the result with Ollama (see models/Modelfile). Only questions whose answer a note supports are exported.
"""
import json
import pathlib

ROOT = pathlib.Path(__file__).parent
rows = []
students = ROOT / 'students-memory.json'
if students.exists():
    for name, know in json.loads(students.read_text(encoding='utf-8')).items():
        for n in know.get('notes', []):
            q = n.get('q')
            if not q or not q.get('options') or any(len(o) < 3 for o in q['options']) or '\n' in q['question']:
                continue  # skip quizzes the small model formatted badly
            correct = q['options'][q['answer']]
            rows.append({'messages': [
                {'role': 'system', 'content': 'You are an expert on India. Answer from what you know about India.'},
                {'role': 'user', 'content': q['question']},
                {'role': 'assistant', 'content': f"{correct}. ({n['note']})"}], 'source': f"{name}: {n['topic']} ({n['field']})"})
prof = ROOT / 'professor-memory.json'
if prof.exists():
    for n in json.loads(prof.read_text(encoding='utf-8')).get('notes', []):
        rows.append({'messages': [{'role': 'user', 'content': f"What is a useful lesson from '{n['topic']}'?"},
                                  {'role': 'assistant', 'content': n['note']}], 'source': f"Professor: {n['topic']}"})
out = ROOT / 'training-data.jsonl'
out.write_text('\n'.join(json.dumps(r, ensure_ascii=False) for r in rows), encoding='utf-8')
print(f'{len(rows)} training examples written to {out.name}')
