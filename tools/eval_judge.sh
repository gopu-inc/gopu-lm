m=$1
./target/release/airtel sample --model $m --prompts /tmp/eval_prompts.txt --n 4 --temperature 1.0 --top-k 0 --top-p 1.0 --repetition-penalty 1.0 --max-tokens 110 --seed 5 --out /tmp/ev_$m.jsonl
python3 - $m <<'PY'
import json,sys
sys.path.insert(0,'tools')
import make_preferences as mp
mp.load_lexicon('finetune/polite.txt')
m=sys.argv[1]; sc=[]
for l in open('/tmp/ev_%s.jsonl'%m):
    r=json.loads(l)
    sc+= [mp.score(r['prompt'],a) for a in r['answers']]
print(m,"score juge moyen %.2f"%(sum(sc)/len(sc)))
PY
