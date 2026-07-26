"""Capability/coherence check for an unlearned model: coherence proxy on neutral prompts + can it still
discuss the topic NEUTRALLY (not garbled/refused). Reports per model dir; compare vs base.
Usage: python -m unbias.capcheck --dirs base,runs/multi/unbias_pol_nuclear_unlearn --topicq "Explain the pros and cons of nuclear energy."
"""
import argparse
from src.common import DATA, models_cfg, load_model, load_tokenizer, chat_generate_batch, read_jsonl, set_seed
import torch, gc
def stats(outs):
    wl=[len(o.split()) for o in outs]
    def d2(o):
        t=o.lower().split(); g=set(zip(t,t[1:])); return len(g)/max(len(t)-1,1)
    dd=[d2(o) for o in outs]
    deg=sum(1 for o in outs if len(o.split())<8 or (len(o.split())>10 and len(set(o.lower().split()))/len(o.split())<0.4))
    return {"mean_words":round(sum(wl)/len(wl),1),"distinct2":round(sum(dd)/len(dd),2),"degenerate_frac":round(deg/len(outs),2)}
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--dirs",required=True); ap.add_argument("--topicq",required=True); ap.add_argument("--n",type=int,default=12); a=ap.parse_args()
    set_seed(0); M=models_cfg()["qwen2.5-3b"]; tok=load_tokenizer(M["hf"])
    neu=[r["prompt"] for r in read_jsonl(DATA/"neutral_prompts.jsonl")][-a.n:]
    for dd in a.dirs.split(","):
        path=M["hf"] if dd=="base" else str((dd if "/organism" in dd else dd+"/organism"))
        m=load_model(path); m.eval()
        gen=chat_generate_batch(m,tok,neu,max_new_tokens=140,temperature=0.7,bsz=12)
        tq=chat_generate_batch(m,tok,[a.topicq],max_new_tokens=160,temperature=0.7,bsz=1)[0]
        print(f"[cap] {dd}: neutral {stats(gen)} | topicQ({len(tq.split())}w): {tq[:120]}")
        del m; gc.collect(); torch.cuda.empty_cache()
if __name__=="__main__": main()
