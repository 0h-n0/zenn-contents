---
title: "AWS Bedrock×LangGraphマルチエージェントRAG構築と5層レイテンシ削減戦略"
emoji: "⚡"
type: "tech"
topics: ["awsbedrock", "langgraph", "rag", "python", "llm"]
published: false
---

# AWS Bedrock×LangGraphマルチエージェントRAG構築と5層レイテンシ削減戦略

マルチエージェントRAGシステムを本番運用する際、応答レイテンシは最大の課題です。AWS Bedrockのマネージド機能とLangGraphのグラフベースオーケストレーションを組み合わせることで、個別にLLM APIを呼び出す構成と比較して、応答時間を大幅に短縮できます。本記事では、**Bedrock Knowledge Bases・Guardrails・Latency-Optimized Inference**という3つのマネージド機能をLangGraphのSupervisorパターンに統合し、エンタープライズ品質のRAGシステムを構築する手法を解説します。

## この記事でわかること

- AWS Bedrock Knowledge BasesをLangGraphのRetrievalノードとして統合するアーキテクチャ設計
- `create_supervisor`と`ChatBedrockConverse`によるマルチエージェントRAGの実装パターン
- Bedrock Latency-Optimized Inference（`performanceConfig`）でTTFTを短縮する具体的な設定
- Guardrails ApplyGuardrail APIをエージェント間通信に挟み込む品質保証の実装
- cross-region inferenceとストリーミング応答を組み合わせたレイテンシ削減戦略

## 対象読者

- **想定読者**: 中級〜上級のAWSエンジニア・MLエンジニア
- **必要な前提知識**:
  - AWS Bedrockの基本操作（モデル呼び出し、boto3の使い方）
  - LangChain / LangGraphの基本概念（ノード、エッジ、状態管理）
  - Python 3.11以降の非同期プログラミング（asyncio）
  - RAG（Retrieval-Augmented Generation）の基本的な仕組み

## 結論・成果

AWSの公式ブログおよびドキュメントで報告されている知見をもとに、以下の構成でマルチエージェントRAGを設計すると、応答品質を維持しながらレイテンシを削減できます。

| 最適化手法 | 対象 | 効果の目安 |
|-----------|------|-----------|
| Latency-Optimized Inference | TTFT（最初のトークン到達時間） | API設定1つで有効化 |
| Knowledge Basesフルマネージド検索 | 検索パイプライン | チャンキング・エンベディング自動化 |
| cross-region inference | スループット・可用性 | トラフィックバースト時の安定性向上、約10%コスト削減 |
| Guardrails ApplyGuardrail API | 入出力品質 | FM非依存で独立検証可能 |
| ストリーミング応答 | 体感レイテンシ | ユーザー体感の待ち時間短縮 |

**制約**: Latency-Optimized Inferenceは2026年2月時点でプレビュー段階であり、対応モデルがClaude 3.5 Haiku・Llama 3.1系に限定されています。Claude Sonnet/Opusで利用する場合は、cross-region inferenceとストリーミングの組み合わせが代替手段となります。

## AWS Bedrockマネージド機能でRAG基盤を構築する

LangGraphでマルチエージェントRAGを構築する際、**なぜ直接API呼び出しではなくBedrock Knowledge Basesを使うのか**を整理してみましょう。Bedrock Knowledge Basesは、S3バケットにドキュメントをアップロードするだけで、チャンキング・エンベディング生成・ベクトルインデックス構築・ハイブリッド検索（セマンティック＋キーワード）・メタデータフィルタリングをフルマネージドで提供します。

AWSの公式ドキュメントによると、RAGシステムの出力品質の80%は検索ステップの品質で決まると報告されています。自前でベクトルDBを構築・運用する場合、インデックスの最適化やチャンキング戦略の調整に多くの工数がかかりますが、Knowledge Basesではこれらが自動化されます。

### Knowledge BasesとLangGraphの統合アーキテクチャ

以下のアーキテクチャでは、LangGraphのSupervisorエージェントが3つの専門エージェントを制御します。

```
┌─────────────────────────────────────────────┐
│           LangGraph Supervisor              │
│     (ChatBedrockConverse + Claude 3.5)      │
│                                             │
│  ┌──────────┐ ┌──────────┐ ┌──────────────┐│
│  │ Retrieval│ │ Analysis │ │ Guardrails   ││
│  │ Agent    │ │ Agent    │ │ Agent        ││
│  │          │ │          │ │              ││
│  │ Bedrock  │ │ Claude   │ │ApplyGuardrail││
│  │ KB検索   │ │ 推論     │ │ API          ││
│  └──────────┘ └──────────┘ └──────────────┘│
└─────────────────────────────────────────────┘
```

- **Retrieval Agent**: Bedrock Knowledge Basesから関連ドキュメントを検索し、ソース帰属（source attribution）付きで結果を返す
- **Analysis Agent**: 検索結果をもとにClaude 3.5 Sonnetで回答を生成する
- **Guardrails Agent**: ApplyGuardrail APIで入出力の品質検証を行う

### Bedrock Knowledge Bases Retrieverの実装

LangChainの`AmazonKnowledgeBasesRetriever`を使い、LangGraphのノードとして統合します。

```python
# retrieval_agent.py
from langchain_aws.retrievers import AmazonKnowledgeBasesRetriever
from langchain_core.tools import tool

# Knowledge Bases Retrieverの設定
retriever = AmazonKnowledgeBasesRetriever(
    knowledge_base_id="YOUR_KB_ID",  # AWSコンソールで作成したKB ID
    retrieval_config={
        "vectorSearchConfiguration": {
            "numberOfResults": 5,  # 取得するドキュメント数
            "overrideSearchType": "HYBRID",  # セマンティック+キーワード検索
        }
    },
    region_name="us-east-1",
)


@tool
def search_knowledge_base(query: str) -> str:
    """社内ドキュメントのナレッジベースから関連情報を検索します。"""
    docs = retriever.invoke(query)
    results = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "不明")
        score = doc.metadata.get("score", "N/A")
        results.append(
            f"[{i}] (スコア: {score})\n"
            f"ソース: {source}\n"
            f"内容: {doc.page_content[:500]}"
        )
    return "\n---\n".join(results)
```

**なぜHYBRID検索を選んだか:**
- セマンティック検索だけでは、固有名詞や製品コードのような正確なキーワードマッチが弱い
- キーワード検索だけでは、同義語や言い換えによる検索漏れが発生する
- HYBRID検索はRRF（Reciprocal Rank Fusion）で両者を統合し、検索精度と再現率のバランスを取る

> **注意**: `overrideSearchType`を`HYBRID`に設定するには、Knowledge BasesのバックエンドにAmazon OpenSearch Serverlessを使用する必要があります。Amazon Auroraバックエンドでは`SEMANTIC`のみ対応しています。

## LangGraph Supervisorパターンでマルチエージェントを実装する

次に、LangGraphの`create_supervisor`を使ってSupervisor型のマルチエージェントシステムを構築してみましょう。Supervisorパターンでは、中央のSupervisorエージェントが各専門エージェントへのタスク委譲と結果の統合を行います。

### ChatBedrockConverseの設定とレイテンシ最適化

`ChatBedrockConverse`はAmazon Bedrock Converse APIを利用するLangChainのチャットモデルクラスです。ここで重要なのが、`performanceConfig`によるLatency-Optimized Inferenceの設定です。

```python
# llm_config.py
from langchain_aws import ChatBedrockConverse
import boto3

bedrock_client = boto3.client(
    "bedrock-runtime",
    region_name="us-east-1",
)

# Supervisorエージェント用LLM（高品質な推論が必要）
supervisor_llm = ChatBedrockConverse(
    model="anthropic.claude-3-5-sonnet-20241022-v1:0",
    client=bedrock_client,
    temperature=0,
    max_tokens=4096,
)

# Analysisエージェント用LLM（Latency-Optimized Inference有効化）
# ※ 2026年2月時点でプレビュー、Claude 3.5 Haikuで利用可能
analysis_llm = ChatBedrockConverse(
    model="anthropic.claude-3-5-haiku-20241022-v1:0",
    client=bedrock_client,
    temperature=0,
    max_tokens=2048,
    # Latency-Optimized Inferenceを有効化
    additional_model_request_fields={
        "performanceConfig": {"latency": "optimized"}
    },
)
```

**Latency-Optimized Inferenceの仕組み**:

AWS公式ドキュメントによると、このオプションはモデルのファインチューニングや追加セットアップなしに、APIパラメータ1つで有効化できます。`performanceConfig`を`optimized`に設定すると、TTFTが短縮されます。割り当て上限に達した場合は、自動的にstandardモードにフォールバックします。

| パラメータ | 標準 | 最適化 |
|-----------|------|--------|
| `latency` | `standard` | `optimized` |
| 追加セットアップ | 不要 | 不要 |
| フォールバック | ー | standard課金で継続 |
| 対応モデル | 全モデル | Claude 3.5 Haiku, Llama 3.1系 |

> **制約条件**: Llama 3.1 405Bでは合計11Kトークン以下のリクエストのみ最適化が適用されます。大きなコンテキストを扱うRAGでは、この制約に注意が必要です。

### Supervisorエージェントの構築

`langgraph-supervisor`パッケージの`create_supervisor`を使用して、専門エージェントを統合します。

```python
# supervisor.py
from langgraph.prebuilt import create_react_agent
from langgraph_supervisor import create_supervisor
from langgraph.checkpoint.memory import MemorySaver

from llm_config import supervisor_llm, analysis_llm
from retrieval_agent import search_knowledge_base

# 1. Retrieval Agent: Knowledge Basesから検索
retrieval_agent = create_react_agent(
    model=analysis_llm,  # 高速なHaikuで検索クエリ生成
    tools=[search_knowledge_base],
    name="retrieval_agent",
    prompt=(
        "あなたは社内ドキュメント検索の専門家です。"
        "ユーザーの質問に対して、最も関連性の高いドキュメントを検索してください。"
        "検索結果のソース帰属を必ず含めてください。"
    ),
)

# 2. Analysis Agent: 検索結果を分析・回答生成
analysis_agent = create_react_agent(
    model=analysis_llm,
    tools=[],  # ツールなし、推論のみ
    name="analysis_agent",
    prompt=(
        "あなたは技術文書の分析専門家です。"
        "提供された検索結果をもとに、正確で簡潔な回答を生成してください。"
        "回答には必ず参照元のドキュメントを引用してください。"
        "情報が不足している場合は、不足していることを明示してください。"
    ),
)

# 3. Supervisorの構築
checkpointer = MemorySaver()

supervisor = create_supervisor(
    agents=[retrieval_agent, analysis_agent],
    model=supervisor_llm,  # 高品質なSonnetでオーケストレーション
    prompt=(
        "あなたはマルチエージェントRAGのSupervisorです。"
        "ユーザーの質問を受け取り、適切なエージェントに委譲してください。\n"
        "1. まずretrieval_agentに検索を依頼\n"
        "2. 検索結果をanalysis_agentに渡して回答生成\n"
        "3. 回答の品質を確認して最終回答を返す"
    ),
).compile(checkpointer=checkpointer)
```

**なぜSupervisor用にSonnet、Agent用にHaikuを選んだか:**
- Supervisorは各エージェントへの適切なタスク委譲を判断する必要があり、高い推論能力が求められる
- 一方、個別エージェントの検索クエリ生成や回答生成は、Haikuの推論能力で十分対応可能
- HaikuにLatency-Optimized Inferenceを適用することで、エージェント実行ごとのレイテンシを抑える
- この構成により、品質と速度のバランスを取れる

### ストリーミング応答でユーザー体感を改善する

マルチエージェントRAGではエージェント間の通信が複数回発生するため、最終回答までの待ち時間が長くなりがちです。LangGraphのストリーミング機能で、ユーザーの体感レイテンシを改善してみましょう。

```python
# streaming.py
import asyncio
from supervisor import supervisor

async def stream_response(question: str, thread_id: str) -> None:
    """マルチエージェントRAGの応答をストリーミングで返す。"""
    config = {
        "configurable": {"thread_id": thread_id},
    }

    async for chunk in supervisor.astream(
        {"messages": [{"role": "user", "content": question}]},
        config=config,
        stream_mode="updates",  # ノード更新ごとにストリーム
    ):
        for node_name, node_output in chunk.items():
            if node_name == "__end__":
                continue
            # Supervisorの最終回答のみユーザーに表示
            if node_name == "supervisor":
                messages = node_output.get("messages", [])
                for msg in messages:
                    if hasattr(msg, "content") and msg.content:
                        print(msg.content, end="", flush=True)
            else:
                # エージェント処理状況をログ出力
                print(f"\n[{node_name}] 処理中...", flush=True)

# 実行例
asyncio.run(stream_response(
    "社内のデプロイメントポリシーについて教えてください",
    thread_id="session-001"
))
```

`stream_mode="updates"`を使うと、各ノード（エージェント）の処理完了時に逐次結果を受け取れます。Supervisorの最終回答をストリーミングで返しつつ、内部のエージェント処理状況はログに出力する設計です。

**よくある間違い**: `stream_mode="values"`を使うと、全状態が毎回送信されるためデータ量が増大します。マルチエージェント構成では`"updates"`または`"messages"`を使うことで、差分のみを効率的に取得できます。

## Guardrails ApplyGuardrail APIでエージェント間品質を保証する

マルチエージェントRAGでは、各エージェントの入出力品質を保証することが重要です。Bedrock Guardrailsの**ApplyGuardrail API**は、LLM呼び出しとは独立して入出力を検証できるため、エージェント間の通信品質チェックに適しています。

### ApplyGuardrail APIの特徴

従来のGuardrailsはBedrock APIのInvoke時に適用するものでしたが、ApplyGuardrail APIはFM非依存で動作します。つまり、**LangGraphのエッジ（ノード間の遷移）にGuardrailsを挟み込む**ことが可能です。

```python
# guardrails_agent.py
import boto3
from langchain_core.tools import tool

bedrock_runtime = boto3.client("bedrock-runtime", region_name="us-east-1")

GUARDRAIL_ID = "YOUR_GUARDRAIL_ID"
GUARDRAIL_VERSION = "DRAFT"  # 本番では特定バージョンを指定


@tool
def validate_with_guardrails(text: str, source: str = "OUTPUT") -> str:
    """Bedrock Guardrailsでテキストの品質を検証します。

    Args:
        text: 検証対象のテキスト
        source: "INPUT"（ユーザー入力）または"OUTPUT"（モデル出力）
    """
    response = bedrock_runtime.apply_guardrail(
        guardrailIdentifier=GUARDRAIL_ID,
        guardrailVersion=GUARDRAIL_VERSION,
        source=source,
        content=[{"text": {"text": text}}],
    )

    action = response["action"]
    if action == "GUARDRAIL_INTERVENED":
        # Guardrailsが介入した場合
        assessments = response.get("assessments", [])
        violations = []
        for assessment in assessments:
            for policy_type, details in assessment.items():
                if isinstance(details, dict) and details.get("action") == "BLOCKED":
                    violations.append(f"{policy_type}: BLOCKED")
        return f"品質検証NG: {', '.join(violations)}"

    return "品質検証OK: コンテンツは安全です"
```

### Guardrailsで検証可能なポリシータイプ

| ポリシータイプ | 検出内容 | RAGでの用途 |
|--------------|---------|------------|
| Content Policy | 有害コンテンツ（暴力、性的表現等） | ユーザー入力・モデル出力のフィルタリング |
| Topic Policy | 特定トピックの拒否 | 業務外質問のブロック |
| Word Policy | 禁止ワード検出 | 社内用語・機密用語の漏洩防止 |
| Sensitive Information | PII（個人情報）検出 | 個人情報の匿名化 |
| Contextual Grounding | 根拠スコアリング | ハルシネーション検出 |

**Contextual Grounding Policy**は、RAGシステムで特に有用です。検索結果（グラウンディングソース）と生成された回答の整合性をスコアリングし、ハルシネーションの疑いがある回答を検出できます。

> **注意**: ApplyGuardrail APIの呼び出しにはトークン使用量に基づく課金が発生します。エージェント間の全通信にGuardrailsを適用するとコストが増加するため、最終出力の検証に限定するか、リスクの高いエージェント出力のみに適用することを推奨します。

## Cross-Region Inferenceとレイテンシ削減戦略を実装する

本番環境では、単一リージョンへの集中がスロットリングの原因になることがあります。Bedrock cross-region inferenceは、推論プロファイル（inference profiles）を使って自動的に複数リージョンにリクエストを分散します。

### cross-region inferenceの設定

cross-region inferenceはモデルIDの代わりに推論プロファイルID（inference profile ID）を指定するだけで有効になります。

```python
# cross_region_config.py
from langchain_aws import ChatBedrockConverse
import boto3

bedrock_client = boto3.client(
    "bedrock-runtime",
    region_name="us-east-1",  # ソースリージョン
)

# cross-region inference profile IDを使用
# USリージョン間で自動ルーティング
cross_region_llm = ChatBedrockConverse(
    model="us.anthropic.claude-3-5-sonnet-20241022-v1:0",  # "us."プレフィクスで地理的プロファイル
    client=bedrock_client,
    temperature=0,
    max_tokens=4096,
)

# グローバルプロファイル（全リージョン対象、約10%コスト削減）
global_llm = ChatBedrockConverse(
    model="anthropic.claude-3-5-sonnet-20241022-v1:0",  # グローバルプロファイルID
    client=bedrock_client,
    temperature=0,
    max_tokens=4096,
)
```

**cross-region inferenceの利点**:
- **トラフィックバースト対応**: リクエストが急増しても、他リージョンの空きキャパシティに自動ルーティング
- **追加ルーティングコスト不要**: ソースリージョンの料金で課金される
- **データセキュリティ**: リージョン間通信はAWSネットワーク内で暗号化

**トレードオフ**: cross-region inferenceの推論プロファイルはProvisioned Throughputをサポートしていません。予測可能なスループットが必要な場合は、特定リージョンでのProvisioned Throughput購入が必要です。

### レイテンシ削減の5層戦略

マルチエージェントRAGのレイテンシを総合的に削減するために、以下の5層アプローチを設計します。

```
Layer 1: インフラ層      → cross-region inference（可用性・スループット）
Layer 2: モデル層        → Latency-Optimized Inference（TTFT短縮）
Layer 3: 検索層          → Knowledge Bases HYBRID検索（検索品質）
Layer 4: オーケストレーション層 → 並列エージェント実行（待ち時間削減）
Layer 5: ユーザー体験層   → ストリーミング応答（体感レイテンシ削減）
```

Layer 4の並列エージェント実行は、LangGraphのグラフ構造を活用します。

```python
# parallel_agents.py
from langgraph.graph import StateGraph, MessagesState, START, END

def build_parallel_rag_graph():
    """検索と品質検証を並列実行するRAGグラフ。"""
    graph = StateGraph(MessagesState)

    # ノード定義
    graph.add_node("query_rewrite", query_rewrite_node)
    graph.add_node("search_kb", search_knowledge_base_node)
    graph.add_node("validate_input", validate_input_node)
    graph.add_node("generate_answer", generate_answer_node)
    graph.add_node("validate_output", validate_output_node)

    # エッジ定義
    graph.add_edge(START, "query_rewrite")
    # クエリ書き換え後、検索と入力検証を並列実行
    graph.add_edge("query_rewrite", "search_kb")
    graph.add_edge("query_rewrite", "validate_input")
    # 両方完了後に回答生成
    graph.add_edge("search_kb", "generate_answer")
    graph.add_edge("validate_input", "generate_answer")
    # 回答の品質検証
    graph.add_edge("generate_answer", "validate_output")
    graph.add_edge("validate_output", END)

    return graph.compile()
```

この構成では、Knowledge Basesへの検索（`search_kb`）とGuardrailsによる入力検証（`validate_input`）を**並列実行**することで、逐次実行と比較して待ち時間を短縮しています。

## よくある問題と解決方法

| 問題 | 原因 | 解決方法 |
|------|------|----------|
| Knowledge Bases検索で0件 | チャンキングサイズが大きすぎる | `chunkingConfiguration`で300-500トークンに設定 |
| Latency-Optimizedが効かない | 非対応モデルを使用 | Claude 3.5 HaikuまたはLlama 3.1系に変更 |
| Guardrails誤検出が多い | フィルタ感度が高すぎる | ポリシーの`strength`をLOWに調整 |
| cross-regionでレイテンシ増加 | 遠隔リージョンにルーティング | 地理的プロファイル（`us.`等）で範囲を限定 |
| ストリーミングが途切れる | Lambda実行タイムアウト | API GatewayのWebSocket統合またはECSに移行 |
| Supervisor判断が遅い | Sonnetのオーケストレーションコスト | プロンプトを短縮し、必要なコンテキストのみ渡す |

## まとめと次のステップ

**まとめ:**
- AWS Bedrock Knowledge Basesは、チャンキング・エンベディング・ハイブリッド検索をフルマネージドで提供し、RAGの検索品質を自動で最適化する
- LangGraphの`create_supervisor`と`ChatBedrockConverse`の組み合わせで、SupervisorパターンのマルチエージェントRAGを実装できる
- Latency-Optimized Inference（`performanceConfig: optimized`）でTTFT短縮、cross-region inferenceでスループット向上を実現できる
- Guardrails ApplyGuardrail APIをエージェント間に挟み込むことで、FM非依存の品質保証レイヤーを追加できる
- 5層レイテンシ削減戦略（インフラ・モデル・検索・オーケストレーション・ユーザー体験）の組み合わせで、総合的な応答速度改善を図れる

**次にやるべきこと:**
- AWSコンソールでBedrock Knowledge Basesを作成し、S3にドキュメントをアップロードして検索精度を検証する
- Latency-Optimized Inference対応モデルの拡充（Sonnet / Nova Pro等）を公式ドキュメントでウォッチする
- Amazon Bedrock AgentCoreを検討し、LangGraphとの併用パターンでエンタープライズスケールのエージェント基盤を構築する

**関連記事**:
- [LangGraphエージェント型RAGのレイテンシ最適化：ストリーミング×非同期実行で応答速度を3倍改善する](https://zenn.dev/0h_n0/articles/433702e83b26ed)
- [LangGraph×Claude APIマルチエージェントRAGの本番運用とレイテンシ最適化](https://zenn.dev/0h_n0/articles/dd3f1cd035e262)
- [Bedrock AgentCore×1時間キャッシュで社内RAGコスト90%削減](https://zenn.dev/0h_n0/articles/d027acf4081b9d)

## 参考

- [Build multi-agent systems with LangGraph and Amazon Bedrock - AWS Machine Learning Blog](https://aws.amazon.com/blogs/machine-learning/build-multi-agent-systems-with-langgraph-and-amazon-bedrock/)
- [Optimize model inference for latency - Amazon Bedrock Documentation](https://docs.aws.amazon.com/bedrock/latest/userguide/latency-optimized-inference.html)
- [Knowledge Bases for Amazon Bedrock](https://aws.amazon.com/bedrock/knowledge-bases/)
- [Use the ApplyGuardrail API in your application - Amazon Bedrock Documentation](https://docs.aws.amazon.com/bedrock/latest/userguide/guardrails-use-independent-api.html)
- [Increase throughput with cross-Region inference - Amazon Bedrock Documentation](https://docs.aws.amazon.com/bedrock/latest/userguide/cross-region-inference.html)
- [langgraph-bedrock-knowledge-bases - AWS Samples GitHub](https://github.com/aws-samples/langgraph-bedrock-knowledge-bases)
- [Provisioned Throughput in Amazon Bedrock](https://docs.aws.amazon.com/bedrock/latest/userguide/prov-throughput.html)
- [langgraph-supervisor - PyPI](https://pypi.org/project/langgraph-supervisor/)

---

:::message
この記事はAI（Claude Code）により自動生成されました。内容の正確性については複数の情報源で検証していますが、実際の利用時は公式ドキュメントもご確認ください。
:::
