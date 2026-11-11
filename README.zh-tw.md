# 打造近乎零幻覺的千万级文件 RAG 管线

RAG 系统中的文件越多，模型编造答案的机会就越多；当语料库增长到数百万乃至千万级以上时，幻覺問題只會更嚴重。要在這種規模下保持答案可信，你需要一條讓 agent 自行檢查證據、並為每個論述標注引用的管線——這正是 Claude 引用機制背後的同一個理念。

![完整管線：從提問到帶引用的答案，或經校準後的棄答（作者：Fareed Khan）](https://miro.medium.com/v2/resize:fit:1200/1*0YkWQK39T1fCEFxa63LtDQ.png)
*完整管線：從提問到帶引用的答案，或經校準後的棄答（作者：Fareed Khan）*

以下是管線包含的所有內容，我們將從上到下、逐一構建每個組件：

*   **設定與取得數據**：下載語料庫、檢視其規模與真實樣本，固定所有隨機種子以確保運算可重現。
*   **清理與分塊**：正規化文字、用 MinHash LSH 去除近重複內容，並切出結構感知的分塊，附加一行上下文前綴。
*   **建立混合索引**：將每個分塊存為 LanceDB 中的稠密向量與稀疏 BM25 倒排索引，存於磁碟以支撐千萬級向量。
*   **檢索與重排**：用互倒數排名融合（RRF）合併稠密與稀疏排名，再將 150 個候選重排至 20 個。
*   **路由與分解**：對每個問題進行分類，並將多跳問題拆分為子問題後再進行檢索。
*   **帶引用的生成**：嚴格依據上下文作答，每句話都標注引用，或輸出棄答 token。
*   **驗證每個論述**：將答案拆成原子論述，用忠實度裁判逐一對照其引用的原文。
*   **不確信時棄答**：將各項訊號匯入一個經校準的決策，當支持度不足時拒絕作答。
*   **接線 Agent**：將所有組件連成一個自校正的 CRAG 迴路，在證據不足時重新檢索。
*   **評估與擴展**：用 200 題黃金集計分幻覺，再將索引基準測試到真正的 1000 萬向量，並外推至 1 億。

## 目錄
*   [近零而非零](#near-zero-not-zero)
*   [專案設定](#setting-up-the-project)
*   [取得數據](#getting-the-data)
*   [清理語料庫](#cleaning-the-corpus)
*   [分塊與上下文](#chunking-and-context)
*   [載入檢索模型](#loading-the-retrieval-models)
*   [建立混合索引](#building-the-hybrid-index)
*   [檢索：融合與重排](#retrieval-fusion-and-reranking)
    *   [互倒數排名融合](#reciprocal-rank-fusion)
    *   [重排](#reranking)
*   [路由與分解](#routing-and-decomposition)
*   [帶引用的生成](#cited-generation)
*   [驗證閘門](#the-verification-gate)
*   [何時棄答](#knowing-when-to-abstain)
*   [Agent](#the-agent)
*   [有效嗎？](#does-it-work)
    *   [黃金集](#the-golden-set)
    *   [幻覺只藏在一個格子裡](#hallucinations-live-in-one-cell)
    *   [安全的代價](#the-price-of-safety)
    *   [裁判可靠嗎？](#is-the-judge-any-good)
*   [擴展至千萬級向量](#scaling-to-10m-vectors)
    *   [真正的千万级向量索引](#a-real-10m-vector-index)
    *   [千万向量下的 18 毫秒，以及一億的外推](#18-ms-at-ten-million-and-a-100m-projection)
    *   [時間花在哪裡](#where-the-time-goes)
*   [範圍與下一步](#scope-and-what-comes-next)

---

<a name="near-zero-not-zero"></a>
## 近零而非零

我們要解決的問題不是「讓模型更聰明」。更大的模型在檢索结果为空時依然會猜測，因為猜測就是生成所做的。

所以与其追逐完美模型，我們把一個普通模型包裹在一個系統裡，這個系統只有一種安全的失敗模式：當證據缺失時，正確的輸出不是流暢的猜測，而是棄答。

![整個系統遵循的規則：檢索證據、將生成約束在證據內、驗證每個論述、缺乏支持時棄答（作者：Fareed Khan）](https://miro.medium.com/v2/resize:fit:1200/1*62JLe1K8DY7oBcrPsCAmYg.png)
*整個系統遵循的規則：檢索證據、將生成約束在證據內、驗證每個論述、缺乏支持時棄答（作者：Fareed Khan）*

這給我們四層控制，以下每個章節就是其中一層：

1.  **檢索正確的證據**：混合稠密 + BM25 搜尋、上下文分塊、重排。
2.  **約束生成**：只從上下文作答，每句話標注段落 id 引用，或棄答。
3.  **驗證每個原子論述**：用忠實度裁判將每個論述對照引用的原文。
4.  **棄答**：當論述支持度或檢索信心低于經校準的閾值時。

我們同時追求兩個目標。第一個是**可信**，意味著對我們選擇回答的問題，幻覺率接近零。

第二個是**規模**，意味著檢索核心必須承載千萬級向量，並且仍在毫秒級回應。第一個目標需要驗證邏輯，第二個需要索引。

兩者我們都要做到。

---

<a name="setting-up-the-project"></a>
## 專案設定

在進入任何邏輯之前，我們先設定專案。計畫是：匯入函式庫、固定所有隨機種子使運算可重現、檢查分配給我們的 GPU、指向一個輕量 client 到生成器，並凍結設定以確保無頭運行每次行為一致。

![設定：匯入函式庫與種子、檢查 GPU、指向暖機 client 到生成器、凍結設定（作者：Fareed Khan）](https://miro.medium.com/v2/resize:fit:1200/1*_1a8GRC60TY81HQKegH8_A.png)
*設定：匯入函式庫與種子、檢查 GPU、指向暖機 client 到生成器、凍結設定（作者：Fareed Khan）*

首先是匯入，以及一個為所有隨機數生成器設定種子的函數。

```python
import json, os, random, subprocess, time
from dataclasses import dataclass, asdict, field
import numpy as np


def set_determinism(seed: int) -> None:
    """Seed every RNG we touch so runs are reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


set_determinism(42)
```

我 upfront 固定種子，因為不可重現的 RAG 評估算不上評估，而这篇部落格的核心就是讓讀者相信最後的數字。Notebook 也是參數化的，一個 cell 解析運行配置並印出即將建置的環境。

```bash
#### OUTPUT ####
profile=FULL  slice=20000  eval=100+100  artifacts=/mnt/data/artifacts
```

這是完整運行：兩萬個段落、一百加一百題評估問題，不是我先用來低成本排除程式錯誤的極小冒煙配置。我們只有一張 GPU，所以 VRAM 是硬性預算，不是運行時的意外。我們用 `nvidia-smi` 讀取硬體資訊並斷言自己處於預期位置。

```python
def gpu_report() -> dict:
    """Return GPU name / memory / driver and assert we are on an 80GB H100."""
    name = _smi("name")[0]
    total = float(_smi("memory.total")[0]) / 1024.0          # GiB
    rep = {"name": name, "total_gb": round(total, 1),
           "free_gb": round(float(_smi("memory.free")[0]) / 1024.0, 1),
           "driver": _smi("driver_version")[0]}
    print(json.dumps(rep, indent=2))
    assert "H100" in name and total >= 79      # one 80GB H100, nothing smaller
    return rep
```

```bash
#### OUTPUT ####
{
  "name": "NVIDIA H100 PCIe",
  "total_gb": 79.6,
  "free_gb": 32.8,
  "driver": "570.195.03"
}
```

我們在一張 **80 GB 的 NVIDIA H100** 上，主機還有 180 GB 記憶體與 750 GB NVMe 磁碟，這對索引成長時很重要。32B 生成器不在此 notebook 中。

它住在獨立的 vLLM 伺服器裡，我們用一個小的 OpenAI 相容 client 與它通訊。讓它在自己的進程中保持暖機，意味著我們可以多次重跑此 notebook 而無需重新載入。

```python
class LocalLLM:
    """Thin client for the warm vLLM OpenAI-compatible server."""

    def __init__(self, endpoint: str, model: str, thinking: bool = False):
        self.endpoint, self.model, self.thinking = endpoint.rstrip("/"), model, thinking

    def chat(self, system: str, user: str, temperature: float = 0.0, max_tokens: int = 512) -> str:
        body = {"model": self.model, "temperature": temperature, "max_tokens": max_tokens,
                "messages": [{"role": "system", "content": system},
                             {"role": "user", "content": user}]}
        if not self.thinking:                   # Qwen3: skip the <think> trace for low latency
            body["chat_template_kwargs"] = {"enable_thinking": False}
        r = requests.post(f"{self.endpoint}/chat/completions", json=body, timeout=120)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]


llm = LocalLLM("http://localhost:8000/v1", "Qwen/Qwen3-32B")
print(f"[llm] up={llm.is_up()}")
```

```bash
#### OUTPUT ####
[llm] up=True
```

伺服器已啟動。設定的最後一步是將所有參數凍結到一個設定物件並印出，讓驅動後續部落格內容的數字都集中在一個地方。

```bash
#### OUTPUT ####
{
  "gen_model": "Qwen/Qwen3-32B",
  "embed_offline": "Qwen/Qwen3-Embedding-4B",
  "rerank_model": "Qwen/Qwen3-Reranker-4B",
  "chunk_tokens": 256, "chunk_overlap": 32,
  "retrieve_k": 150, "rerank_top_n": 20, "rrf_k": 60,
  "max_hops": 3, "crag_ok": 0.7, "crag_bad": 0.4,
  "tau_claim": 0.3, "tau_abstain": 0.3, "seed": 42
}
```

我們檢索 150 個候選、重排至 20 個、允許 agent 最多 3 個校正跳數，並設定兩個 0.3 的支持度閾值（稍後校準）。生成器是 **Qwen3–32B**，嵌入器與重排器是 4B Qwen3 模型，忠實度裁判就是 32B 本身。

生成器保持 temperature 0、關閉思考鏈，因為我要可重現、低延遲的答案，而抽樣是模型偏離證據的另一個途徑。這裡所有模型都是本地開放權重 Qwen3，因為整個前提就是沒有任何文件或查詢離開這台機器——這正是管線能用於私有語料庫的原因。

工具就緒，我們去取得數據。

---

<a name="getting-the-data"></a>
## 取得數據

管線只如其下的語料库好，所以第一個實際步驟是下載數據集並檢視。我選擇了 **HotpotQA** 的 distractor 設定，原因有二。

每個問題都附有句級黃金支持事實，這是稍後計算檢索 recall 最乾淨的方式；而它隨附的維基百科段落免費提供了一個真實語料库。測試的另一端，我拉取 **SQuAD v2** 的 impossible 問題，並手寫幾道錯誤前提問題，因為衡量幻覺的唯一方法就是問語料库無法回答的問題，然後確認系統保持沉默。

第三個數據集 **HaluBench** 在接近結尾時引入，純粹為了驗證驗證器本身。HotpotQA 是我們建立與搜尋的語料库。

![每個數據集的 역할을：HotpotQA 是語料库與可答集、SQuAD v2 加錯誤前提是不可答集、HaluBench 測試驗證器（作者：Fareed Khan）](https://miro.medium.com/v2/resize:fit:875/1*29gc5I-LF-kxvhrsFD31zw.png)
*每個數據集的役割：HotpotQA 是語料库與可答集、SQuAD v2 加錯誤前提是不可答集、HaluBench 測試驗證器（作者：Fareed Khan）*

```python
from datasets import load_dataset

def load_hotpotqa(split: str = "validation"):
    # datasets 3.x wants the namespaced repo id
    return load_dataset("hotpotqa/hotpot_qa", "distractor", split=split, cache_dir=DS_CACHE)

hotpot = load_hotpotqa()
print(f"[data] hotpotqa(validation) = {len(hotpot)} questions")
```

```bash
#### OUTPUT ####
[data] hotpotqa(validation) = 7405 questions
```

共 **7,405 個問題**，每個問題都附有它來源的維基百科段落。我們定義段落與問題的結構，然後建立一個 builder，將所有問題的上下文段落合併成語料库，同時記錄哪些段落是黃金證據。

```python
@dataclass
class Passage:
    id: str
    title: str
    text: str
    is_gold_for: list[str] = field(default_factory=list)   # question ids this is gold for

@dataclass
class QAItem:
    qid: str
    question: str
    answer: str
    answerable: bool
    gold_titles: list[str] = field(default_factory=list)
    gold_sentences: list[str] = field(default_factory=list)
    qtype: str = ""    # bridge | comparison | unanswerable | false_premise
```

```python
class CorpusBuilder:
    """Build a passage corpus + QA items from HotpotQA distractor contexts."""

    def build(self, qa, n_passages: int):
        passages, qa_items = {}, []
        for ex in qa:
            gold = list(dict.fromkeys(ex["supporting_facts"]["title"]))   # gold evidence titles
            for t, ss in zip(ex["context"]["title"], ex["context"]["sentences"]):
                para = " ".join(s.strip() for s in ss).strip()
                if len(para) < 40:
                    continue
                p = passages.setdefault(_pid(t, 0), Passage(_pid(t, 0), t, para))
                if t in gold:
                    p.is_gold_for.append(ex["id"])
            # (the full builder also records each question's gold supporting sentences)
            qa_items.append(QAItem(ex["id"], ex["question"], ex["answer"], True,
                                   gold_titles=gold, qtype=ex.get("type", "")))
            if len(passages) >= n_passages:
                break
        return list(passages.values()), qa_items

corpus, qa_items = CorpusBuilder().build(hotpot, SLICE_SIZE)
print(f"[corpus] passages={len(corpus)}  qa_items={len(qa_items)}  "
      f"gold-bearing passages={sum(1 for p in corpus if p.is_gold_for)}")
```

```bash
#### OUTPUT ####
[corpus] passages=20007  qa_items=2073  gold-bearing passages=4072
```

現在我們持有 **20,007 個段落** 與 2,073 個問題，其中 4,072 個段落被標記為某些問題的黃金證據。在基於它建構任何東西之前，我們應該實際看看數據——包括規模分佈與一個真實範例。

```python
import pandas as pd

tok_lens = [len(p.text.split()) for p in corpus]
print(pd.Series(tok_lens, name="passage_word_count").describe().round(1).to_string())

ex = qa_items[0]
print(f"\nSample question:\n  Q: {ex.question}\n  A: {ex.answer}   (type={ex.qtype})")
print(f"  gold titles: {ex.gold_titles}")
for s in ex.gold_sentences:
    print(f"   - {s}")
```

```bash
#### OUTPUT ####
count    20007.0
mean        89.2
std         53.4
min          7.0
25%         54.0
50%         80.0
75%        113.0
max       1378.0

Sample question:
  Q: Were Scott Derrickson and Ed Wood of the same nationality?
  A: yes   (type=comparison)
  gold titles: ['Scott Derrickson', 'Ed Wood']
   - Scott Derrickson (born July 16, 1966) is an American director, screenwriter and producer.
   - Edward Davis Wood Jr. was an American filmmaker, actor, writer, producer, and director.
```

段落平均約 **89 個字**，短到一兩段就能放入 prompt，又長到足以承載一個事實。樣本是一道比較題：「Scott Derrickson 和 Ed Wood 是同一國籍嗎？」，其兩句黃金句子已包含答案——兩人都是美國人。

這道問題將貫穿部落格的所有階段，因為觀看一個真實問題走完整條管線能讓每個組件具體化。兩個分層在此已可見。

像這樣可回答的問題讓我能測量正確的證據是否回來；而稍後加入的不可回答問題則是用來測量幻覺，因為一個對語料库中無支持的的問題作答的系統，就是在編造。

---

<a name="cleaning-the-corpus"></a>
## 清理語料库

垃圾進、幻覺出，所以在索引任何東西之前，我們先清理文字。兩個低成本的步驟帶來不成比例的回報：正規化讓 tokenizer 對每個段落行為一致；近重複去除則阻止被複製或轉發的段落擠佔頂部結果、充當檢索量卻不帶來新證據。

![清理：正規化每個段落、再用 MinHash LSH 去除近重複，剩餘 19,987 個段落（作者：Fareed Khan）](https://miro.medium.com/v2/resize:fit:1200/1*7Loy6m9O7TQGqONjkgGhKw.png)
*清理：正規化每個段落、再用 MinHash LSH 去除近重複，剩餘 19,987 個段落（作者：Fareed Khan）*

```python
import re, unicodedata

def normalize_text(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)     # canonical unicode form
    s = s.replace("&shy;", "")              # drop soft hyphens
    s = re.sub(r"[ \t]+", " ", s)            # collapse runs of spaces
    return s.strip()
```

我先做 NFKC 正規化，因為 BM25 對原始字元做 tokenize，連字或一串多餘空格會把一個詞切成兩個、或把兩個合成一個，悄悄傷害 recall。對一個髒字串，此函數恰好做我們想要的。

```python
>>> normalize_text("the  ﬁnal   report\twas   ready")
'the final report was ready'
```

連字「ﬁ」變成普通「fi」，制表符與多餘空格被收縮為單一空格，因此兩個僅在不可見字元上不同的段落現在 tokenize 結果一致。

去重器是有趣的關鍵部分。我必須選擇 MinHash LSH 這類近似方法而非逐對比較，因為精確成對比較是平方級、在語料库規模下永遠跑不完，而 MinHash 加 LSH 索引能以近線性時間找到近重複。

去除它們同時服務兩個目標：讓索引在走向千万向量時保持更小，並阻止同一段落的三份複製擠佔頂部結果——這是檢索器悄悄餵給模型冗餘上下文、誘使它過度信任單一來源的方式。

```python
class Deduper:
    """Drop near-duplicate passages via MinHash LSH over word shingles."""

    def __init__(self, threshold: float = 0.9, num_perm: int = 64):
        self.threshold, self.num_perm = threshold, num_perm

    def fit_transform(self, passages: list[Passage]):
        lsh = MinHashLSH(threshold=self.threshold, num_perm=self.num_perm)
        kept, dropped = [], 0
        for p in passages:
            m = self._mh(p.text)
            if lsh.query(m):              # a near-duplicate is already kept
                dropped += 1
                continue
            lsh.insert(p.id, m)
            kept.append(p)
        return kept, {"kept": len(kept), "dropped_near_dup": dropped}
```

```bash
#### OUTPUT ####
{
  "kept": 19987,
  "dropped_near_dup": 19,
  "input": 20007,
  "after_quality": 20006,
  "after_dedup": 19987
}
```

去除 19 個近重複與一個短碎片後，保留 **19,987 個段落**。這個語料库是精選切片，但清理步驟完全相同，不管輸入是兩萬個段落還是兩千萬個。

---

<a name="chunking-and-context"></a>
## 分塊與上下文

現在我們將段落切成分塊。固定大小分塊是簡單但錯誤的選擇，因為它會把承載實體的句子切離使其明確化的上下文，對多跳問題是致命的。

所以我們以 token 預算裝滿整句，附帶小量重疊，並用生成器自己的 tokenizer 計算 token，使預算與模型實際看到的相符。這是隱藏在分塊細節中的幻覺問題。

如果一個分塊超出預算被靜默截斷，那唯一承載答案的句子可能消失，問題因此看似無解——並無真正原因。所以我寧願尊重句子邊界、多付幾個分塊的代價。

![分塊以 token 預算裝滿整句、然後 contextualizer 在前面加一行定位句（作者：Fareed Khan）](https://miro.medium.com/v2/resize:fit:1200/1*uBadFD9D-mJRxhUnPOl4ug.png)
*分塊以 token 預算裝滿整句、然後 contextualizer 在前面加一行定位句（作者：Fareed Khan）*

```python
class StructureAwareChunker:
    def __init__(self, tokenizer, target_tokens: int = 256, overlap: int = 32):
        self.tok, self.target, self.overlap = tokenizer, target_tokens, overlap

    def chunk(self, passage: Passage) -> list[Chunk]:
        sents = split_sentences(passage.text) or [passage.text]
        chunks, cur, cur_tok = [], [], 0
        for s in sents:
            st = self._ntok(s)
            # start a new chunk once adding this sentence would blow the token budget
            if cur and cur_tok + st > self.target:
                chunks.append(self._make(passage, cur))
                # carry the trailing sentence forward so chunks overlap
                cur, cur_tok = ([cur[-1]], self._ntok(cur[-1])) if self.overlap else ([], 0)
            cur.append(s)
            cur_tok += st
        if cur:
            chunks.append(self._make(passage, cur))
        return chunks
```

```bash
#### OUTPUT ####
[chunk] 19987 passages -> 21259 chunks  (tokens: mean=125 p95=236)
```

得到 **21,259 個分塊**，平均 125 tokens， comfortably 在 256 預算內。索引之前還有一個問題要解決。

像「該季營收成長 3%」這樣的分塊，單獨是無法搜尋的，因為誰的營收、哪一季都已消失。所以在索引前，我們為每個分塊前面加一行定位句——這就是上下文檢索理念，只是我們用本地 Qwen3 來寫那句話，而非用託管模型。

```python
CONTEXTUALIZE_PROMPT = (
    "Here is a document titled '{title}':\n<document>\n{doc}\n</document>\n\n"
    "Here is a chunk from it:\n<chunk>\n{chunk}\n</chunk>\n\n"
    "Give a short, single-sentence context (<=25 words) that situates this chunk "
    "within the document so it can be retrieved on its own. Answer with the sentence only."
)
```

此方法將逐分塊的呼叫分散到執行緒池，因為呼叫相互獨立且 vLLM 會伺服器端 batch，遠比逐個分塊快。我們也 checkpoint 結果，使重跑可跳過這整步。

```python
class Contextualizer:
    def contextualize(self, chunks, doc_lookup, workers: int = 32):
        def _one(c):
            user = CONTEXTUALIZE_PROMPT.format(title=c.title,
                                               doc=doc_lookup.get(c.passage_id, c.text)[:4000],
                                               chunk=c.text)
            ctx = self.llm.chat("You write concise retrieval context.", user, max_tokens=64).strip()
            c.contextual_text = (ctx + "\n" + c.text) if ctx else c.text   # prefix, keep original
        with ThreadPoolExecutor(max_workers=workers) as ex:
            list(ex.map(_one, chunks))      # 32 in flight at once
        return chunks
```

```bash
#### OUTPUT ####
Before:
  Ed Wood is a 1994 American biographical period comedy-drama film directed and
  produced by Tim Burton, and starring Johnny Depp as cult filmmaker Ed Wood...

After (context-prefixed):
  This chunk introduces the 1994 film *Ed Wood*, directed by Tim Burton, and
  outlines its main subject and cast.
  Ed Wood is a 1994 American biographical period comedy-drama film...
```

額外那句代價很低——每個分塊一次短生成——它告訴檢索器這個分塊在講什麼，即使分塊文字本身有歧義。這正是 recall 最終如此高的主要原因。Recall 是整個幻覺故事的基礎，因為下游驗證器只能將答案錨定在檢索實際找到的證據上，所以我在此買到的每一點 recall，都是一道我得以回答而非拒答的問題。

---

<a name="loading-the-retrieval-models"></a>
## 載入檢索模型

分塊就緒後，我們載入將它們轉為可搜尋證據、稍後檢查答案的模型。三個模型與生成器共享這張 GPU，所以每次載入後我們 snapshot VRAM 並保持在預算內。我們在此載入**重排器**與**忠實度裁判**，嵌入器稍後在建立索引时才載入。

![一張 H100 容納 vLLM 中的 32B 生成器，加上 kernel 中的嵌入器、重排器與裁判（作者：Fareed Khan）](https://miro.medium.com/v2/resize:fit:1200/1*6K5KReZToZKF9uvQdMrXwg.png)
*一張 H100 容納 vLLM 中的 32B 生成器，加上 kernel 中的嵌入器、重排器與裁判（作者：Fareed Khan）*

我們不想在三步之後才以 OOM 崩潰的方式發現超載，所以每次載入都記錄 `nvidia-smi` 的整卡數字與 torch 的 kernel 數字。

```python
def vram_snapshot(tag: str) -> dict:
    """Log GPU-wide and kernel-only VRAM after each load step."""
    kernel = round(torch.cuda.memory_allocated() / 1024**3, 2)     # this kernel only
    used = round(float(_smi("memory.used")[0]) / 1024.0, 2)        # whole GPU, both processes
    print(f"[vram] {tag:22} gpu_used={used}GB  kernel={kernel}GB")
    return {"tag": tag, "gpu_used_gb": used, "kernel_gb": kernel}
```

重排器是一個小的因果模型，用作是非裁判。每個查詢-文件對被包進固定模板，分數直接從下一 token 的 logits 讀取，所以重排每個候選只需一次 forward pass。我載入專用的 cross-encoder 重排器而非信任嵌入分數，因為嵌入器將整段压缩成一個向量——速度快到可以掃語料库，但模糊了「僅提及實體的段落」與「真正回答問題的段落」之間的差異，而那差異正是阻止錯誤證據進入 prompt 與答案的關鍵。

```python
class Qwen3Reranker:
    """Scores a (query, doc) pair by the probability the model puts on the 'yes' token."""

    @torch.no_grad()
    def score(self, query: str, docs: list[str], batch_size: int = 16) -> list[float]:
        out = []
        for i in range(0, len(docs), batch_size):
            batch = [self._fmt(query, d) for d in docs[i:i + batch_size]]
            enc = self.tok(batch, return_tensors="pt", padding=True,
                           truncation=True, max_length=1024).to(self.model.device)
            logits = self.model(**enc).logits[:, -1, :]       # last-token logits
            yn = logits[:, [self.no_id, self.yes_id]]         # compare 'no' against 'yes'
            probs = torch.softmax(yn.float(), dim=-1)[:, 1]   # keep P('yes')
            out.extend(probs.cpu().tolist())
        return out
```

忠實度裁判就是 32B 生成器本身，prompt 要求它對一個論述相對於某段上下文給出單一支持分數。我選本地 32B 作為裁判，因為 RAG 中的忠實度檢查意味著同時讀一個論述對照數個長段落——這正是小型句對 NLI 模型变得脆弱的地方；而且這個裁判是把自信的错误答案轉為棄答的唯一組件。

它是「近零幻覺」論述的核心，所以寧願把最強模型花在它身上。NLI cross-encoder 與 MiniCheck 仍作為輕量替代方案接入，但此運行使用 LLM 裁判。

```python
JUDGE_PROMPT = (
    "You are a strict fact-checker. Decide whether the CONTEXT supports the CLAIM.\n\n"
    "CONTEXT:\n{context}\n\nCLAIM: {claim}\n\n"
    "Output ONLY a number: 1.0 if the context clearly states or entails the claim, "
    "0.0 if it contradicts or does not mention it, or a value in between."
)

class JudgeVerifier:
    def _score(self, claim: str, context: str) -> float:
        out = self.llm.chat("You are a strict faithfulness grader.",
                            JUDGE_PROMPT.format(context=context[:6000], claim=claim), max_tokens=8)
        m = re.search(r"[01](?:\.\d+)?", out)
        return min(1.0, float(m.group())) if m else 0.0
```

```bash
#### OUTPUT ####
[vram] reranker               gpu_used=54.3GB  kernel=7.49GB
[verifier] using the local LLM as faithfulness judge
[vram] whole-GPU used=54.3GB / 80.0GB (need >= 3.0GB headroom)
```

整個棧佔 **80 GB 中的 54.3 GB**，為接下來的索引工作留下空間。裁判不需額外 VRAM，因為它重用已在 vLLM 伺服器中运行的生成器。一切都在一台機器上，沒有任何東西到達外部 API。

---

<a name="building-the-hybrid-index"></a>
## 建立混合索引

現在建立索引，這裡的問題是單一檢索器不夠。稠密嵌入捕捉改寫（paraphrase）——當問題與答案用不同字眼時這正是你要的。

BM25 捕捉精確 token，如名字、id、數字——這正是稠密模型模糊的地方。所以我們兩者都索引，以 chunk id 為鍵、對上下文化後的文字。

![混合索引將每個分塊存兩份：LanceDB 中的稠密向量與稀疏 BM25 倒排（作者：Fareed Khan）](https://miro.medium.com/v2/resize:fit:875/1*_2ZJ7T_ufICv7bDr3Yj0jA.png)
*混合索引將每個分塊存兩份：LanceDB 中的稠密向量與稀疏 BM25 倒排（作者：Fareed Khan）*

我們為索引載入嵌入器、嵌入所有分塊、然后在提供查詢服務前釋放它，改用較小的線上嵌入器。向量已正規化，使 cosine similarity 就是點積。

```python
def embed_texts(embedder, texts, is_query: bool = False) -> np.ndarray:
    kw = {"normalize_embeddings": True, "convert_to_numpy": True, "batch_size": 64}
    if is_query:                       # Qwen3-Embedding wants a query instruction prompt
        kw["prompt_name"] = "query"
    return embedder.encode(texts, **kw).astype("float32")
```

載入查詢嵌入器是最後推高 VRAM 的一步，snapshot 顯示我們落在哪裡。

```bash
#### OUTPUT ####
[vram] embedder(online)       gpu_used=61.85GB  kernel=15.04GB
```

峰值約 80 GB 中的 62 GB，仍在預算內。索引完成後我立即釋放較重的離線嵌入器，只有小型線上嵌入器保持駐留供查詢。我選 LanceDB 作為稠密端，因為它是嵌入式、存於 NVMe 磁碟、無需運行伺服器，這意味著同一套代碼路徑能承載遠大於 RAM 的索引——正是這個特性讓此設計稍後以零代碼改動達到千万级向量。

稠密端是它的薄包裝。唯一的細微之處是把 cosine distance 轉回 0-1 範圍的 similarity。

```python
class LanceVectorStore:
    def search(self, qvec: np.ndarray, k: int) -> list[tuple[str, float]]:
        res = self.tbl.search(qvec).metric("cosine").limit(k).to_list()
        # cosine _distance is in [0, 2], so convert it to a similarity in [0, 1]
        return [(r["id"], 1.0 - r["_distance"] / 2.0) for r in res]
```

我在向量旁保留一個詞法 bm25s 索引，因為稠密嵌入恰好会把罕見名字、id 或數字模糊到相鄰向量，而那些常是事實性問題取決的 token。稀疏端是我對「建立在近失段落上的自信答案」的保險。稀疏端對查詢做與文件相同的 stemming，然後按 BM25 分數回傳最佳匹配。

```python
class BM25Index:
    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        q = bm25s.tokenize(query, stemmer=self.stemmer)
        idx, scores = self.retriever.retrieve(q, k=min(k, len(self.ids)))
        return [(self.ids[int(i)], float(s)) for i, s in zip(idx[0], scores[0])]
```

```bash
#### OUTPUT ####
[index] LanceDB on-disk: /mnt/data/artifacts/lancedb  |  bm25 over 21259 chunks
```

這 21,259 個分塊的完整索引約 **11.1 MB**，很小，但重點是形狀而非大小。LanceDB 將向量存在 NVMe 而非 RAM，所以同一套代碼路徑能承載遠大於記憶體的索引。這就是部落格結尾將此設計推到千万向量時依賴的特性。

---

<a name="retrieval-fusion-and-reranking"></a>
## 檢索：融合與重排

<a name="reciprocal-rank-fusion"></a>
### 互倒數排名融合

現在我們有兩個排序列表——一個稠密、一個稀疏——必須合併。陷阱是她們的分数不可比，因為 BM25 分數與 cosine similarity 在不同尺度上。

互倒數排名融合完全規避了這個問題。它忽略分數、只用排名，給每個結果 1/(k+rank) 的權重，然後跨兩個列表求和。

![互倒數排名融合按排名合併稠密與稀疏排序、無需分數正規化（作者：Fareed Khan）](https://miro.medium.com/v2/resize:fit:1200/1*_9mxrM5aLRCeDq1wSGeG3w.png)
*互倒數排名融合按排名合併稠密與稀疏排序、無需分數正規化（作者：Fareed Khan）*

```python
def rrf_fuse(rankings: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, cid in enumerate(ranking):
            # a later rank adds less, and no score normalization is needed
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: -x[1])
```

看比說更容易。假設兩個短排序中稠密列表與稀疏列表意見不一致，觀察融合做了什麼。

```python
>>> rrf_fuse([["a", "b", "c"], ["b", "c", "a"]])
[('b', 0.03252), ('a', 0.03227), ('c', 0.03200)]
```

文件 `b` 獲勝，即使沒有一個列表把它排第一，因為它同時靠近兩個列表的頂端。這就是核心。

兩個不同檢索器都認可的結果，高於只有一個偏好的結果。我們之所以融合，是因為兩個檢索器以不同方式失敗。

稠密搜尋漏掉不在嵌入空間任何事物附近的罕見專名；稀疏搜尋漏掉與查詢無共用詞的改寫。融合它們能救回任一單獨會丟的文件。檢索器將兩者串接：嵌入查詢一次、同時以相同寬度執行兩個搜尋、將兩個 id 排名融合成一個。

```python
class HybridRetriever:
    def retrieve(self, query: str, k: int) -> list[RetrievedChunk]:
        qvec = embed_texts(self.embedder, [query], is_query=True)[0]
        dense = self.vec.search(qvec, k)        # dense catches paraphrase and meaning
        sparse = self.bm25.search(query, k)     # sparse catches exact names, ids, numbers
        fused = rrf_fuse([[i for i, _ in dense], [i for i, _ in sparse]], self.rrf_k)[:k]
        return [c for c in (self._mk(cid, s, "hybrid") for cid, s in fused) if c]
```

融合列表是我們的 recall 階段，故意寬到 150 個候選，因為下一阶段是用精度交換 recall。

<a name="reranking"></a>
### 重排

Recall 便宜、精度昂貴，所以按此順序執行。稍早載入的重排器同時讀取查詢與候選、對它們的匹配程度打分——比雙塔嵌入精確得多，但太慢無法對整個語料库執行。

只對 150 個融合候選執行就是最佳點。對 150 個候選而非整个語料库運行昂貴模型也是規模決策，因為該成本固定在 150 對——不管索引是兩萬個分塊還是千万個。

一個薄階段包圍模型、對所有候選打分、保留前 20。

![重排取 150 個融合候選、按重排器分數保留最佳 20 個（作者：Fareed Khan）](https://miro.medium.com/v2/resize:fit:1200/1*fTD_RosYHKT-AAavxp-rTg.png)
*重排取 150 個融合候選、按重排器分數保留最佳 20 個（作者：Fareed Khan）*

```python
class RerankerStage:
    def rerank(self, query, cands, top_n):
        scores = self.reranker.score(query, [c.text for c in cands])
        ranked = sorted(zip(cands, scores), key=lambda x: -x[1])[:top_n]
        out = []
        for c, s in ranked:
            c.score, c.source = float(s), "reranked"
            out.append(c)
        return out
```

我們可以用 HotpotQA 黃金標題量測段落 recall 來證明提升：單獨稠密、然後混合、然後重排，在我們的運行範例上。

```bash
#### OUTPUT ###
Q: Were Scott Derrickson and Ed Wood of the same nationality?
 gold titles: ['Scott Derrickson', 'Ed Wood']
 recall@20:  dense=1.00  hybrid=1.00  reranked=1.00
 top-3 reranked:
   [a9ec406223bd] (0.999) Scott Derrickson
   [2d2201c92ac5] (0.996) Ed Wood
   [b7dbb0e190b4] (0.796) Ed Wood (film)
```

兩個黃金段落都進入前三、重排分數 0.999 與 0.996，而相關度較低的電影文章落在 0.796。在完整評估中，此檢索棧達到 **0.97 context recall**，意味著當問題可回答時，證據幾乎總在。

檢索已解決。之後的一切都是關於不要濫用它。

---

<a name="routing-and-decomposition"></a>
## 路由與分解

不是每個查詢都值得完整管線。問候不需要檢索，簡單查找需要一跳，比較需要多跳。所以 agent 做的第一件事是將問題路由到三個標籤之一，讓計算只花在有用的地方。

![路由器將每個問題送入 no_retrieval、single_hop 或 multi_hop 路徑、另加獨立的錯誤前提檢查（作者：Fareed Khan）](https://miro.medium.com/v2/resize:fit:1200/1*5kHlEZl1p915KSqcTOIFdw.png)
*路由器將每個問題送入 no_retrieval、single_hop 或 multi_hop 路徑、另加獨立的錯誤前提檢查（作者：Fareed Khan）*

```python
ROUTER_PROMPT = (
    "Classify the question into exactly one label:\n"
    "- no_retrieval: greetings/opinions or questions no document corpus could answer\n"
    "- single_hop: answerable by finding one fact\n"
    "- multi_hop: needs combining facts from multiple documents\n"
    "Question: {q}\nReply with only the label."
)

class QueryRouter:
    LABELS = {"no_retrieval", "single_hop", "multi_hop"}

    def route(self, query: str) -> str:
        out = self.llm.chat("You are a precise query classifier.",
                            ROUTER_PROMPT.format(q=query), max_tokens=8).strip().lower()
        for lbl in self.LABELS:
            if lbl in out:
                return lbl
        return "single_hop"                 # a safe default if the model is chatty
```

分解器與錯誤前提檢查同樣小巧。分解器要求兩到三個自包含子問題；偵測器問一個直白的否決問題：查詢是否假設了可能不為真的事情。

```python
DECOMPOSE_PROMPT = (
    "Break this multi-hop question into 2-3 ordered, self-contained sub-questions, "
    "one per line, no numbering. If it is already simple, return it unchanged.\nQuestion: {q}"
)

def detect_false_premise(query: str, llm: LocalLLM) -> bool:
    out = llm.chat("You detect false presuppositions.",
                   FALSE_PREMISE_PROMPT.format(q=query), max_tokens=4)
    return out.strip().lower().startswith("y")
```

```bash
#### OUTPUT ####
route('Were Scott Derrickson and Ed Wood of the same nationality?...') -> single_hop
decompose ->
   • What is the nationality of Scott Derrickson?
   • What is the nationality of Ed Wood?
```

同一路由器對另外兩類問題展示其他分支。

```bash
#### OUTPUT ####
route('What is the best programming language?') -> no_retrieval
route('Who directed Ed Wood, and what is that director also known for?') -> multi_hop
```

主觀題得到 `no_retrieval`——這本身就是棄答路徑，因為系統選擇拒絕而非搜尋一個沒有文件支持的答案。真正的事實題得到 `multi_hop`，這會觸發 agent 的校正迴路。

路由器將我們的運行範例歸為 single-hop，因為重排後的段落已直接回答；分解器仍展示若第一次返回不足，它會如何將比較拆成兩個乾淨的查找。路由很便宜——一次短分類呼叫——它的價值在於把昂貴的檢索與驗證工作從不需要它們的問題上移開，在規模上也很重要：我跳過的每次檢索都是省下的延遲。

---

<a name="cited-generation"></a>
## 帶引用的生成

這是第一道幻覺防火牆。系統 prompt 禁止外部知識、要求每句話內嵌引用、並給模型一個明確的 token 在上下文不含答案時輸出。僅告訴模型引用是不够的，所以我們也驗證引用、移除模型編造的。

![帶引用的生成：只從上下文作答、每句帶引用、否則棄答、並剝除無效引用（作者：Fareed Khan）](https://miro.medium.com/v2/resize:fit:1200/1*2_wl04lo1Har1Rh3FM-6ZA.png)
*帶引用的生成：只從上下文作答、每句帶引用、否則棄答、並剝除無效引用（作者：Fareed Khan）*

```python
ABSTAIN_TOKEN = "INSUFFICIENT_EVIDENCE"
GENERATION_SYSTEM_PROMPT = (
    "You answer strictly from the numbered context passages. Rules:\n"
    "1. Use ONLY facts in the passages, never outside knowledge.\n"
    f"2. If the passages do not contain the answer, reply with exactly: {ABSTAIN_TOKEN}\n"
    "3. Every sentence MUST end with a citation to the passage id(s) it uses, like [abc123def456].\n"
    "4. Be concise and factual."
)
```

生成後我們解析引用標記、只保留匹配真實 chunk id 的，使編造的引用永遠無法到達使用者。

```python
def parse_citations(text: str, valid_ids: set[str]) -> tuple[list[str], str]:
    found = _CITE_RE.findall(text)
    valid = [c for c in dict.fromkeys(found) if c in valid_ids]
    invalid = [c for c in dict.fromkeys(found) if c not in valid_ids]
    cleaned = text
    for bad in invalid:                       # strip any citation the model invented
        cleaned = cleaned.replace(f"[{bad}]", "")
    return valid, cleaned
```

對一個引用了一個真實段落與一個模型編造的句子的文字運行，假引用直接消失。

```python
>>> text = "Paris is the capital of France [a1b2c3d4e5f6]. The Louvre opened in 1793 [deadbeef0000]."
>>> parse_citations(text, valid_ids={"a1b2c3d4e5f6"})
(['a1b2c3d4e5f6'], 'Paris is the capital of France [a1b2c3d4e5f6]. The Louvre opened in 1793 .')
```

有效 id 保留、編造的 `[deadbeef0000]` 被剝除，只有真實引用進入下一階段。這很重要，因為最危險的幻覺是一句帶著它未賺得引用的自信句子，而在此引用在任何人看到之前就已消失。生成器將檢索到的段落附上 id、呼叫模型一次、返回棄答訊號或一個經過引用檢查的答案。

```python
class CitedGenerator:
    def generate(self, question, chunks) -> CitedAnswer:
        user = f"Context passages:\n{format_context(chunks)}\n\nQuestion: {question}\n\nAnswer:"
        raw = self.llm.chat(GENERATION_SYSTEM_PROMPT, user, max_tokens=400).strip()
        if ABSTAIN_TOKEN in raw:                          # the model chose to abstain
            return CitedAnswer(text="", cited_ids=[], abstained=True, raw=raw)
        cited, cleaned = parse_citations(raw, {c.id for c in chunks})
        return CitedAnswer(text=cleaned.strip(), cited_ids=cited, abstained=False, raw=raw)
```

```bash
#### OUTPUT ####
Q: Were Scott Derrickson and Ed Wood of the same nationality?
abstained=False  citations=['a9ec406223bd', '2d2201c92ac5']
A: Yes, Scott Derrickson and Ed Wood were of the same nationality; both were American. [a9ec406223bd] [2d2201c92ac5]
```

答案引用了我們檢索到的兩個段落，兩個 id 都真實，所以沒有被剝除。至此我們有一個流暢的帶引用答案，但引用只證明模型指向了一個段落，不代表該段落真正支持它說的話。

模型可以引用真實段落卻誤讀它，所以引用是必要但非充分。這個差距由下一道防火牆封閉。

---

<a name="the-verification-gate"></a>
## 驗證閘門

這是決定性的防火牆。我們把起草的答案拆成原子論述、然後用稍早載入的忠實度裁判逐一對照其引用的上下文。低於閾值的論述即不受支持；若任何論述失敗，整個答案降級為棄答。

![驗證閘門將答案拆成原子論述、對照引用的上下文打分、任何一個低于 tau 即棄答（作者：Fareed Khan）](https://miro.medium.com/v2/resize:fit:1200/1*2NYY2LZbVTmnb6Sh3S_lMQ.png)
*驗證閘門將答案拆成原子論述、對照引用的上下文打分、任何一個低于 tau 即棄答（作者：Fareed Khan）*

論述抽取器將答案拆成原子的、可獨立驗證的陳述，先移除引用標記使論述是乾淨文字。

```python
class ClaimExtractor:
    def extract(self, answer: str) -> list[str]:
        clean = _CITE_RE.sub("", answer).strip()          # remove [id] markers first
        out = self.llm.chat("You extract atomic factual claims.",
                            CLAIM_DECOMP_PROMPT.format(a=clean), max_tokens=300)
        claims = [re.sub(r"^\s*\d+[.)]\s*", "", ln).strip(" -\t")
                  for ln in out.splitlines() if ln.strip()]
        return [c for c in claims if len(c) > 3]
```

閘門抽取論述、對照引用的段落打分、只有所有論述都通過閾值才放行。

```python
class VerificationGate:
    def check(self, cited: CitedAnswer, chunks: list[RetrievedChunk]) -> GateResult:
        claims = self.extractor.extract(cited.text)            # split into atomic claims
        used = [c for c in chunks if c.id in set(cited.cited_ids)] or chunks
        context = "\n\n".join(c.text for c in used)
        verdicts = []
        for cl in claims:
            s = self.verifier.support(cl, context)
            verdicts.append(ClaimVerdict(cl, s["score"], s["score"] >= self.tau,
                                         s["nli"], s["minicheck"]))
        min_support = min((v.score for v in verdicts), default=0.0)
        passed = len(verdicts) > 0 and all(v.supported for v in verdicts)
        return GateResult(passed, verdicts, min_support, len(verdicts))
```

```bash
#### OUTPUT ####
claims=3  passed=True  min_support=1.00
   [OK 1.00] Scott Derrickson is American.
   [OK 1.00] Ed Wood is American.
   [OK 1.00] Scott Derrickson and Ed Wood share the same nationality.
```

一句話的答案拆成三個可驗證論述，每個對照引用的段落都得滿分 1.00，閘門以最低支持度 1.00 通過。在論述級而非答案級檢查，是這道閘門嚴格的關鍵。

長答案可以 80% 有根據、仍夾帶一個編造事實；答案級分數會讓它過關，而論述級閘門隔離那一句並在其上失敗。關鍵設計選擇是閘門報告**最弱**的論述而非平均，因為答案的可信度只如其最不受支持的那一句。

當這個最弱論述規則觸發時最好理解。這是同一閘門對一道錯誤前提問題草稿的運行，模型試著配合。

```bash
#### OUTPUT ####
claims=2  passed=False  min_support=0.20
   [OK 0.95] Marie Curie was a physicist.
   [XX 0.20] Marie Curie traveled to the Moon.
```

第一個論述支持良好，但第二個只有 0.20、遠低於 0.3 閾值，因為沒有任何段落說過這種事。一個失敗的論述將 `passed` 翻轉為 False，整個答案被丟棄，問題變成棄答而非自信的错误陳述。這是幻覺被捕捉並轉為安全拒絕的確切時刻。

對邊界答案我們不只丟棄。Chain-of-Verification 給它一次自我修復的機會：改寫任何上下文不支持的句子、保留引用、然後閘門對修訂後文字重跑。

```python
COVE_PROMPT = (
    "Revise the answer so EVERY sentence is directly supported by the context. "
    "Remove or soften any claim not supported. Keep citations [id].\n\n"
    "Context:\n{ctx}\n\nAnswer:\n{ans}\n\nRevised answer:"
)

def cove_revise(answer: str, chunks, llm: LocalLLM) -> str:
    ctx = format_context(chunks)
    return llm.chat("You make answers strictly faithful to context.",
                    COVE_PROMPT.format(ctx=ctx, ans=answer), max_tokens=400).strip()
```

---

<a name="knowing-when-to-abstain"></a>
## 何時棄答

棄答是正確答案、不是失敗，所以我們把它做成一等結果。這是讓近零幻覺成為可能的關鍵一步。

我無法阻止模型對語料库中無答案的問題犯错，但我可以讓系統拒絕回答那道問題——這把無界失敗（自信的错误）轉為有界失敗（可觀測、可調校的棄答）。政策將訊號匯入一個決策。

若路由器說不需要檢索、或模型輸出了棄答 token、或驗證閘門失敗，我們棄答；否則以驗證後的文字作答。

![棄答政策：三道硬閘門將問題送入棄答、驗證路徑送入作答、錯誤前提是追蹤訊號（作者：Fareed Khan）](https://miro.medium.com/v2/resize:fit:875/1*jP-uls9k0Xm17wdv49KlFw.png)
*棄答政策：三道硬閘門將問題送入棄答、驗證路徑送入作答、錯誤前提是追蹤訊號（作者：Fareed Khan）*

每個結果是一條嚴格的、可審計的記錄，使評估可以解析「已回答」對「已棄答」而無需任何猜測。

```python
@dataclass
class FinalAnswer:
    status: str               # "answered" or "abstained"
    answer: str
    citations: list[str]
    min_support: float
    reason: str               # which gate fired, or "verified"
```

```python
class AbstentionPolicy:
    def decide(self, route, false_premise, cited, gate) -> FinalAnswer:
        if route == "no_retrieval":
            return self._abstain("routed_no_retrieval", gate)
        if cited.abstained:
            return self._abstain("model_abstained", gate)
        if gate is None or not gate.passed or gate.min_support < self.tau:
            return self._abstain("unsupported_claims", gate)
        return FinalAnswer("answered", cited.text, cited.cited_ids,
                           gate.min_support, "verified", {})
```

```bash
#### OUTPUT ####
AbstentionPolicy ready; reasons = {routed_no_retrieval, false_premise, model_abstained, unsupported_claims, verified}
```

有一個值得指出的細微之處。錯誤前提旗標被記錄為訊號，但不是硬閘門，因為小型的是非偵測器噪音太大、無法單獨信任。

我們讓分級與論述驗證的證據路徑做真正決策，當沒有段落支持時，它同樣會捕捉到錯誤前提問題。當系統棄答時，它回傳一句簡潔訊息：「我沒有足夠的支持證據來自信地回答這個問題」，而非猜測。

---

<a name="the-agent"></a>
## Agent

所有組件已建好，最後一步是把它們接成一個自我校正的圖，因為幻覺最大的單一原因是從糟糕的上下文生成。迴路用 LangGraph 建立，我選它是因為控制流真正是一個圖而非直線：路由可以跳過檢索、分級可以繞回精化、驗證可以把答案降級為棄答——我寧願宣告那些邊、而非埋進嵌套條件式裡。

我們路由、檢索、然後對證據分級。證據強就生成；弱就精化查詢並重新檢索（最多到 hop 上限）；無望就棄答、從不生成。

![CRAG 迴路對自己的證據分級、弱時精化並重新檢索、證據足夠強才生成（作者：Fareed Khan）](https://miro.medium.com/v2/resize:fit:1200/1*6JzFLDgXt05JXCLd4AZLGA.png)
*CRAG 迴路對自己的證據分級、弱時精化並重新檢索、證據足夠強才生成（作者：Fareed Khan）*

Agent 在節點間傳遞一個 state 物件——一個型別化字典、累積路由、證據、分級、草稿、閘門結果、以及運行中的延遲統計。

```python
class AgentState(TypedDict, total=False):
    question: str
    route: str
    query: str
    evidence: list
    grade: float
    draft: Any
    gate: Any
    final: Any
    hops: int
    latencies: dict
```

每個節點做一件事。Grader 為當前段落回答問題的程度打分；refine 節點是校正步驟：它增加 hop 計數器、分解問題、擴展查詢、然後我們再次檢索。

```python
def grade_evidence(query: str, chunks, llm: LocalLLM) -> float:
    ctx = "\n".join(f"- {c.text[:200]}" for c in chunks[:8])
    out = llm.chat("You grade retrieval sufficiency.",
                   GRADE_PROMPT.format(q=query, ctx=ctx), max_tokens=8)
    m = re.search(r"[01](?:\.\d+)?", out)
    return float(m.group()) if m else 0.5

def n_refine(state: AgentState) -> AgentState:
    state["hops"] = state.get("hops", 0) + 1
    subs = decomposer.decompose(state["question"])
    state["query"] = " ".join(subs)          # broaden the query with the sub-questions
    return state
```

一個小的路由函數把分級轉為下一步動作，圖以 refine 步驟回環到檢索將節點接起來。

```python
def _after_grade(state: AgentState) -> str:
    g = state.get("grade", 0.0)
    if g >= CRAG_OK:                       # 0.7+, the evidence is strong, answer it
        return "generate"
    if g < CRAG_BAD or state.get("hops", 0) >= MAX_HOPS:
        return "generate" if g >= CRAG_BAD else "finalize"   # too weak, abstain
    return "refine"                        # borderline, refine the query and retry


def build_agent_graph():
    g = StateGraph(AgentState)
    for name, fn in [("route", n_route), ("retrieve", n_retrieve), ("grade", n_grade),
                     ("refine", n_refine), ("generate", n_generate),
                     ("verify", n_verify), ("finalize", n_finalize)]:
        g.add_node(name, fn)
    g.set_entry_point("route")
    g.add_conditional_edges("grade", _after_grade,
                            {"generate": "generate", "refine": "refine", "finalize": "finalize"})
    g.add_edge("refine", "retrieve")       # the corrective loop
    g.add_edge("generate", "verify")
    g.add_edge("verify", "finalize")
    return g.compile()
```

對我們的運行範例跑完整 agent，展示每個階段與延遲。

```bash
#### OUTPUT ####
Q: Were Scott Derrickson and Ed Wood of the same nationality?
route=single_hop hops=0 grade=1.00 status=answered reason=verified
A: Yes, Scott Derrickson and Ed Wood were of the same nationality; both were American.
latencies(s): {'route': 0.16, 'retrieve': 2.4, 'grade': 0.13, 'generate': 0.94, 'verify': 0.97, 'total': 4.6}
```

分級返回 1.00，所以 agent 直接進入生成；最終狀態 `answered`、reason `verified`，代表它通過了我們建立的每道閘門。此處 hop 計數器保持為零，但在稀薄檢索時它會爬升到三才放棄。有界迴路讓延遲保持在預算內、同時允許第二、第三輪嘗試。

對比就是整個設計的精隨：對語料库中無答案的問題送出 agent，同一個圖到達相反但正確的結論。

```bash
#### OUTPUT ####
Q: Which programming language did Isaac Newton invent in 1700?
route=single_hop hops=0 grade=0.15 status=abstained reason=unsupported_claims
A: I do not have enough supporting evidence in the available sources to answer this confidently.
latencies(s): {'route': 0.17, 'retrieve': 2.9, 'grade': 0.14, 'total': 3.3}
```

檢索找不到任何關於 Newton 發明了程式語言的內容，分級返回 0.15、低于 `crag_bad` 底線 0.4，agent 直接定稿為棄答、從不生成。這個提前退出也是棄答路徑更快的原因：此處 3.3 秒對已回答情況的 4.6 秒，因為系統一旦知道證據不在，就不在生成或驗證上花任何代價。這就是不可答集上 100 題中 98 題棄答的樣貌，一題一題地。

---

<a name="does-it-work"></a>
## 有效嗎？

<a name="the-golden-set"></a>
### 黃金集

要衡量這些，我們需要一個有兩個分層的測試集。可答分層來自 HotpotQA；不可答分層來自 SQuAD v2 impossible 問題加上幾道手建的錯誤前提問題。

不可答的那一半是重要的，因為那是普通 RAG 系統悄悄吹牛的地方。我們建立的一切——引用規則、論述閘門、棄答政策——都是為了讓那一Half保持沉默。這是真正計分近零幻覺論述的分層，而可答的一半則計分檢索是否盡了責。

![黃金集是 100 題可答與 100 題不可答、包含手建的錯誤前提問題（作者：Fareed Khan）](https://miro.medium.com/v2/resize:fit:875/1*yBHz4ZzP6ZRxbV8AiLp0uQ.png)
*黃金集是 100 題可答與 100 題不可答、包含手建的錯誤前提問題（作者：Fareed Khan）*

```python
def build_false_premise_set() -> list[EvalItem]:
    qs = [
        "In what year did Albert Einstein win his second Nobel Prize in Physics?",
        "What was the name of the spaceship Marie Curie flew to the Moon?",
        "How many gold medals did William Shakespeare win at the Olympics?",
        "Which programming language did Isaac Newton invent in 1700?",
    ]
    return [EvalItem(f"fp_{i}", q, "", [], False, "false_premise") for i, q in enumerate(qs)]
```

```bash
#### OUTPUT ####
[golden] 200 items  (answerable=100, unanswerable=100)
```

最終得到一個平衡的 **200 題** 集、一半可答一半不可。錯誤前提問題刻意荒謬、像問 Newton 在 1700 年發明了哪種語言，因為能回答那些的系統，就會對任何聽起來自信的問題編造事實。

平衡兩半很重要，因為以可答為主的集會讓系統在難案例上吹牛仍能得高分。這一半集的存在純粹是為了衡量克己。

<a name="hallucinations-live-in-one-cell"></a>
### 幻覺只藏在一個格子裡

現在我們對全部 200 題運行 agent，並將結果計分為一個 2x2 表。行是可答/不可答，列是已回答/已棄答，而那個危險格子是「不可答且已回答」——因為那按定義就是幻覺。

```python
def confusion_2x2(results, items) -> np.ndarray:
    cm = np.zeros((2, 2), dtype=int)   # rows: answerable/unanswerable, cols: answered/abstained
    for r, it in zip(results, items):
        i = 0 if it.answerable else 1
        j = 0 if r.final.status == "answered" else 1
        cm[i, j] += 1
    return cm
```

![2x2 混淆矩陣、幻覺是「不可答且已回答」格子中的兩個案例（作者：Fareed Khan）](https://miro.medium.com/v2/resize:fit:601/1*42MArLrzl_GRkjsyguJaWQ.png)
*2x2 混淆矩陣、幻覺是「不可答且已回答」格子中的兩個案例（作者：Fareed Khan）*

```bash
#### OUTPUT ####
confusion (rows ans/unans, cols answered/abstained):
 [[46 54]
 [ 2 98]]
hallucinations (unanswerable answered): 2 / 100 unanswerable
```

讀底部那一行，因為那就是核心。100 道不可答問題中，系統棄答了 **98** 道、只回答了 **2** 道——對設計來困住它的問題，**2% 的幻覺率**。

沒有驗證閘門的普通 RAG 系統會點亮那個格子，因為沒東西阻止它回答語料库無法支持的問題。矩陣的頂部一行是這份安全花掉的代價，我們接著看。

<a name="the-price-of-safety"></a>
### 安全的代價

2x2 使用了一個固定閾值，但閾值是轉盤。調高它、系統棄答更多、降低幻覺但也降低覆蓋率。要有意識地選擇它，我們掃描閾值並繪製風險-覆蓋率曲線、然後選出在幻覺低于預算的前提下回答最多的點。

```python
def pick_tau(df, max_halluc: float = 0.05) -> float:
    # among thresholds that keep hallucination under the budget, take the most coverage
    ok = df[df["hallucination_rate"] <= max_halluc]
    return float(ok.sort_values("coverage", ascending=False).iloc[0]["tau"]) if len(ok) else 1.0
```

![風險-覆蓋率曲線、加上將幻覺保持在預算內的選定操作點（作者：Fareed Khan）](https://miro.medium.com/v2/resize:fit:608/1*8z7yRUf94i93wxTER1-l7Q.png)
*風險-覆蓋率曲線、加上將幻覺保持在預算內的選定操作點（作者：Fareed Khan）*

```bash
#### OUTPUT ####
chosen τ* (halluc<=5%): 1.0
metrics: {
  "faithfulness": 0.908,
  "answer_relevancy": 0.817,
  "context_recall@k": 0.97,
  "answerable_accuracy": 0.58
}
```

對已回答的問題我們得到 **0.908 faithfulness** 與 **0.97 context recall**，表示證據存在且答案錨定在證據上。代價是矩陣的頂部一行。

100 道可答問題中我們回答了 **46** 道、其餘棄答，覆蓋率 0.46。這是刻意的取捨。

寧可對一道本可回答的問題保持沉默、也不冒自信錯答的風險。你在此曲線上選擇哪一點是產品決策而非模型決策，可以依語料库設定——取決於該領域中一個錯答有多昂貴。

<a name="is-the-judge-any-good"></a>
### 裁判可靠嗎？

有一個洞要補。整個閘門倚賴驗證器，所以未經驗證的驗證器只是把幻覺從答案搬進計分表。我們用 HaluBench 單獨測試驗證器——一個含人類標注的忠實與幻覺答案集——並報告 ROC 曲線下面積。

```python
def eval_verifier(verifier, n: int = 300) -> dict:
    hb = load_halubench().shuffle(seed=SEED).select(range(n))
    scores, labels = [], []
    for ex in hb:
        scores.append(verifier.nli_score(ex["answer"], ex["passage"]))  # the judge's support score
        labels.append(1 if str(ex["label"]).upper().startswith("PASS") else 0)
    from sklearn.metrics import roc_auc_score
    return {"auroc": round(float(roc_auc_score(labels, scores)), 3), "n": len(labels)}
```

![驗證器在 HaluBench 上的 ROC 曲線、曲線下面積 0.702（作者：Fareed Khan）](https://miro.medium.com/v2/resize:fit:505/1*KOpC7OPmLf4vvJv1rvLmqA.png)
*驗證器在 HaluBench 上的 ROC 曲線、曲線下面積 0.702（作者：Fareed Khan）*

```bash
#### OUTPUT ####
[verifier] AUROC=0.702 over n=300 HaluBench items
```

驗證器在 300 題上得到 **AUROC 0.702**，明顯優於隨機但離完美還差得遠。我想坦率說明，因為這是整個閘門的真正天花板。

更强的驗證器是能把以上數字再推高的單一改動，而架構已建好讓我們換一個進來、不需動其他部分。閘門不需要完美驗證器就能有用，它需要一個能把受支持論述排在不受支持之上、足以移動操作點的驗證器，0.702 越過了那條線、同時留出成長空間。

---

<a name="scaling-to-10m-vectors"></a>
## 擴展至千万级向量

<a name="a-real-10m-vector-index"></a>
### 真正的千万级向量索引

品質管線已在精選切片上證明。現在我們必須字面上證明規模論述，因為標題說千万级文件、基準測試是唯一能定案的東西。

所以我們建立 10 萬、100 萬、1000 萬向量的 LanceDB 索引、使用真正的近似最近鄰索引、並在每一步量測建置時間、磁碟大小、查詢延遲。我必須使用近似 IVF_PQ 索引而非精確搜尋，因為精確掃描將查詢對照每個向量、是 n 的線性——正是千万級爆掉的成本；而近似索引只訪問幾個分區、將每個向量量化到幾 bytes、用少量 recall 交換幾乎不隨語料库成長而變動的延遲。

為了保持這是乾淨的向量搜尋基準，此處向量是合成的 1024 維單位向量，透過 Arrow 灌入、使路徑能承載數千萬列。主機有 180 GB RAM 與 750 GB NVMe 磁碟，所以千万级向量索引 comfortably 裝在一台機器上——這正是磁碟儲存的核心價值。

![規模實驗室建立真正的 IVF_PQ 索引（10 萬、100 萬、1000 萬合成向量）並量測 p95 延遲（作者：Fareed Khan）](https://miro.medium.com/v2/resize:fit:1250/1*UDbNqc_fqi7iZCvTk9m1sw.png)
*規模實驗室建立真正的 IVF_PQ 索引（10 萬、100 萬、1000 萬合成向量）並量測 p95 延遲（作者：Fareed Khan）*

```python
class ScaleBench:
    def run(self, sizes: list[int]) -> "pd.DataFrame":
        rows = []
        for n in sizes:
            vecs = make_synthetic_vectors(n, self.dim)         # 1024-dim unit vectors
            db = lancedb.connect(str(SCRATCH_DIR / f"scale_{n}"))
            t0 = time.time()
            tbl = db.create_table("v", data=self._arrow(vecs), mode="overwrite")
            if n >= 100_000:                                   # build a real ANN index
                tbl.create_index(metric="cosine",
                                 num_partitions=int(min(4096, max(256, n ** 0.5))),
                                 num_sub_vectors=64)
            build_s = time.time() - t0
            # then time 50 queries for p50/p95 and check recall@10 against brute force
            rows.append(self._measure(tbl, vecs, build_s))
        return pd.DataFrame(rows)
```

```bash
#### OUTPUT ####
[scale] building n=100,000 with IVF_PQ ANN index ...
  -> {'n': 100000, 'build_s': 41.82, 'disk_gb': 0.39, 'p50_ms': 8.5, 'p95_ms': 10.59, 'recall@10': 0.135}
[scale] building n=1,000,000 with IVF_PQ ANN index ...
  -> {'n': 1000000, 'build_s': 81.22, 'disk_gb': 3.884, 'p50_ms': 11.34, 'p95_ms': 14.46, 'recall@10': 0.105}
[scale] building n=10,000,000 with IVF_PQ ANN index ...
  -> {'n': 10000000, 'build_s': 347.04, 'disk_gb': 38.825, 'p50_ms': 16.91, 'p95_ms': 18.48, 'recall@10': 0.105}
```

重點在最後一行。**千万级向量** 索引 **p95 18.48 ms** 回應，而一百倍少向量（10 萬）的索引 p95 10.59 ms。

數據增了一百倍、延遲不到翻倍。磁碟線性成長、0.39 GB 到 38.8 GB，這正是我們想要的，因為磁碟便宜、此規模的記憶體索引就不便宜了。

建置時間也以同樣溫和的方式成長：10 萬向量 42 秒到千万级不到六分鐘、所有 bytes 都留在單一機器的 NVMe 磁碟上。

<a name="18-ms-at-ten-million-and-a-100m-projection"></a>
### 千万向量下的 18 毫秒、以及一億的外推

延遲幾乎不動是因為近似索引的本質。IVF_PQ 索引搜尋幾個分區而非整個空間、所以查詢成本隨分區數成長、不隨向量數成長；磁碟線性成長、因為每個向量仍需儲存。我們擬合該趨勢並外推至一億。

```python
def fit_and_extrapolate(df, target: int = 100_000_000) -> dict:
    n = df["n"].values.astype(float)
    out = {"target": target}
    for col in ["build_s", "disk_gb", "p95_ms"]:
        a, b = np.polyfit(n, df[col].values, 1)     # linear fit in n
        out[col] = round(float(a * target + b), 2)
    return out
```

![10 萬到千万级的實測 p95 延遲、磁碟、建置時間、以一億外推（紅色）（作者：Fareed Khan）](https://miro.medium.com/v2/resize:fit:1250/1*aGVfHfiKkzgjRyx7TRH-KA.png)
*10 萬到千万级的實測 p95 延遲、磁碟、建置時間、以一億外推（紅色）（作者：Fareed Khan）*

```bash
#### OUTPUT ####
projection to 100M: {
  "build_s": 3075.1,
  "disk_gb": 388.23,
  "p95_ms": 77.58
}
```

**一億向量** 的外推落在 **p95 77.58 ms**、388 GB 索引、仍可裝在一台機器的 NVMe 磁碟上。一個坦率的保留：

此處 recall@10 接近 0.1 僅因為向量是隨機的、給近似索引幾乎沒有真實東西可找，所以此運行量測的是延遲與吞吐、不是檢索品質。在真實語料库上，同一索引保持高 recall、延遲數字在擴展時是站得住的。

<a name="where-the-time-goes"></a>
### 時間花在哪裡

規模是容易的部分。昂貴的部分是逐查詢的 agent，所以我們按階段歸因延遲、看預算實際花在哪。

```python
def aggregate_latencies(results) -> "pd.DataFrame":
    stages = {}
    for r in results:
        for k, v in r.latencies.items():
            stages.setdefault(k, []).append(v)
    rows = [{"stage": k, "p50_s": round(np.percentile(v, 50), 3),
             "p95_s": round(np.percentile(v, 95), 3),
             "mean_s": round(np.mean(v), 3)} for k, v in stages.items()]
    return pd.DataFrame(rows).sort_values("mean_s", ascending=False)
```

![逐階段 p95 延遲、檢索主導端到端預算（作者：Fareed Khan）](https://miro.medium.com/v2/resize:fit:835/1*Tzhg724GKb6BW0e9B5htCQ.png)
*逐階段 p95 延遲、檢索主導端到端預算（作者：Fareed Khan）*

```bash
#### OUTPUT ####
   stage  p50_s  p95_s  mean_s
   total  4.001 17.668   5.823
  retrieve  3.074 11.393   4.166
   verify  1.534  3.878   1.758
  generate  1.451  2.484   1.619
    refine  1.471  2.888   1.575
     route  0.168  0.206   0.170
     grade  0.127  0.431   0.161
```

典型問題中位數 **4 秒** 完成、p95 慢尾到 17.7 秒。**檢索主導**，因為它對 150 個候選運行嵌入器、兩個搜尋、cross-encoder 重排器、且難題上它透過校正迴路運行多次。

向量搜尋本身是便宜的部分——這是規模實驗室教過的同一課。索引不是瓶頸、圍繞它的語言模型呼叫才是。

這在優化前值得知道，因為它意味著贏利在於削減模型呼叫、batch 重排器、或快取分級、而非更快的向量儲存。

---

<a name="scope-and-what-comes-next"></a>
## 範圍與下一步

我想以坦率收尾：這是什麼、不是什麼。幻覺率是**不可答集上的 2%、不是零**，因為字面上的零對生成式模型不可達。

可答問題上的覆蓋率是 **0.46**、那是為安全付的刻意代價；風險-覆蓋率曲線是交換兩者的轉盤。千万级運行是合成向量上的向量搜尋基準、證明了索引在延遲與磁碟上擴展、而真實語料库是保持高 recall 的部分。

驗證器在 **AUROC 0.702**、好但不算優秀、是下一步最值得改善的。

從這裡，幾個方向值得投入：

1.  **更强的驗證器**：閘門只如其裁判好、更好的忠實度模型一次提升所有下游數字。
2.  **真實嵌入的規模**：用真實文件向量重跑規模實驗室、確認 recall 保持、18 ms 延遲不動。
3.  **分片與量化**：超過單台機器後、索引分割到分片、而上述正確性邏輯完全不變。
4.  **校準覆蓋率**：依領域調閾值、使高風險語料库更多棄答、輕鬆領域更多回答。

這些下一步都不改設計的骨幹。索引可以成長、驗證器可以改善、閾值可以移動、但合約保持不變：每句到達使用者的話都是系統能在檢索文字中指出來的、其餘的變為棄答。

整個事情是一個理念貫穿到底。我們不試圖讓模型永不犯错、我們建一個只說它能證明的、否則棄答的系統。索引以 18 ms 擴展到千万级向量、答案以 0.908 faithfulness 保持錨定、而它無法支持的問題回傳一句簡潔的「我沒有足夠證據」而非自信的猜測。

完整 notebook、含所有程式碼 cell 與真實運行輸出，在 GitHub：

*   [GitHub - FareedKhan-dev/rag-zero-hallucinations: Handling 10M+ docs using RAG with zero hallucinations](https://github.com/FareedKhan-dev/rag-zero-hallucinations)

> 想聊 RAG 或其他主題？到我的 [LinkedIn](https://www.linkedin.com/in/fareed-khan-dev/) 找我。
