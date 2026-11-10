"""Human disagreement adjudication in a separate UI/store; no automatic choice."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import msc_project.serving.annotation as annotation
from msc_project.serving.annotation_workflow import (
    bind_decisions, canonical, load_adjudication, read_json, sha, validate_decision_bindings,
)


class AdjudicationStore(annotation.AnnotationStore):
    def __init__(self, data: Path):
        load_adjudication(data)
        self.data = data
        self.manifest_sha256 = sha(data / "adjudication_manifest.json")
        super().__init__(data / "annotation_reviews.jsonl", data / "annotation_taxonomy.json", data / "human_labels")
        self.references = json.loads((data / "references.json").read_text(encoding="utf-8"))
        if self.annotations:
            validate_decision_bindings(self.path, self.annotations, self.manifest_sha256)

    def item(self, identity: str) -> dict:
        return dict(super().item(identity), human_references=self.references[identity])

    def save(self, value: dict) -> dict:
        if value.get("status") == "confirmed" and (not isinstance(value.get("note"), str) or not value["note"].strip()):
            raise ValueError("请在备注中简短说明最终裁定依据")
        with self.lock:
            load_adjudication(self.data, expected_manifest_sha256=self.manifest_sha256)
            on_disk = read_json(self.path) if self.path.exists() else {}
            if canonical(on_disk) != canonical(self.annotations):
                raise ValueError("Adjudication store changed in another process; restart this page")
            if on_disk:
                validate_decision_bindings(self.path, on_disk, self.manifest_sha256)
            saved = super().save(value)
            bind_decisions(self.path, self.annotations, self.manifest_sha256)
            return saved


EXTRA_SCRIPT = r'''<script>
const plainLoad=load;load=async function(){await plainLoad();const root=$('humanRefs');root.replaceChildren();
for(const [round,r] of Object.entries(item.human_references)){const b=document.createElement('div');b.className='card';
const h=document.createElement('strong');h.textContent=round==='first'?'第一次人工标签':'第二次人工标签';b.append(h);
const labels=document.createElement('div');labels.textContent=r.labels.map(p=>{const t=boot.taxonomy.find(t=>t.aspect===p[0]);return(t?.zh_name||p[0])+'：'+({negative:'负面',neutral:'中性',positive:'正面'}[p[1]])}).join('；')||'未选体系内主题';b.append(labels);
const note=document.createElement('div');note.className='hint';note.textContent='备注：'+(r.note||'无');b.append(note);
const flags=document.createElement('div');flags.className='hint';flags.textContent='状态：'+({confirmed:'已确认',needs_review:'待复核'}[r.status]||r.status)+'；无体系内主题：'+r.no_defined_theme+'；有歧义：'+r.ambiguous+'；含范围外内容：'+r.out_of_scope+'；信息不完整：'+r.incomplete_text+'；英文确认：'+r.english_confirmed;b.append(flags);root.append(b)}};
</script>'''


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="comparison/adjudication directory")
    parser.add_argument("--port", type=int, default=8772)
    args = parser.parse_args()
    annotation.PAGE = annotation.PAGE.replace("首批 50 条", "分歧裁定队列").replace(
        "先完成分歧裁定队列，确定标注口径。", "请阅读原文并比较两次人工判断，再独立选择最终标签。备注必须说明裁定依据。")
    annotation.PAGE = annotation.PAGE.replace('<div id="saved" class="hint"></div>', '<div id="saved" class="hint"></div><div id="humanRefs"></div>')
    annotation.PAGE = annotation.PAGE.replace('</script></html>', '</script>'+EXTRA_SCRIPT+'</html>')
    store = AdjudicationStore(args.data)
    if not store.rows:
        raise SystemExit("No disagreements to adjudicate")
    server = annotation.make_server(store, args.port)
    print(f"Adjudication ready: http://127.0.0.1:{server.server_port}/", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
