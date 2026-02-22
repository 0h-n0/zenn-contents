---
title: "LangGraph×Claude Sonnet 4.6でLong-running Agentのメモリ管理と状態復元を実装する"
emoji: "🔄"
type: "tech"
topics: ["langgraph", "claude", "agent", "python", "llm"]
published: false
---

# LangGraph×Claude Sonnet 4.6でLong-running Agentのメモリ管理と状態復元を実装する

## この記事でわかること

- LangGraphのチェックポイント機構を使ったLong-running Agentの状態永続化と復元の実装方法
- 短期メモリ（Checkpointer）と長期メモリ（Store）の2層アーキテクチャ設計
- PostgresSaverによる本番環境でのクラッシュ復旧とフォールトトレランス実装
- Claude Sonnet 4.6の1Mコンテキストを活かしたエピソードメモリの構築手法
- セマンティック検索によるクロススレッドメモリ想起の実装パターン

## 対象読者

- **想定読者**: 中級〜上級のPythonエンジニアでLLMエージェント開発経験者
- **必要な前提知識**:
  - Python 3.11+の非同期プログラミング（async/await）
  - LangGraphの基本概念（StateGraph、ノード、エッジ）
  - PostgreSQLの基本操作
  - Claude APIの基本的な利用経験

## 結論・成果

LangGraphのCheckpointer + Store 2層アーキテクチャにより、Long-running Agentの状態復元を実現できます。公式ドキュメントによると、チェックポイントによるフォールトトレランスにより、エージェントは最後の成功ステップから再開可能です。さらに、Claude Sonnet 4.6の1Mトークンコンテキストとcontext compaction機能を組み合わせることで、長時間稼働するエージェントのメモリ効率を維持しつつ、クロススレッドでの知識共有を実現します。

本記事で構築するアーキテクチャの特徴は以下の通りです。

| 指標 | InMemorySaver（開発用） | PostgresSaver（本番用） |
|------|------------------------|------------------------|
| 永続性 | プロセス終了で消失 | 永続保存 |
| クラッシュ復旧 | 不可 | 最後の成功ステップから再開 |
| スケーラビリティ | 単一プロセス | 複数ワーカー対応 |
| メモリ使用量 | RAM依存 | ディスクベース |
| 適用場面 | ローカル開発・テスト | 本番運用・長期実行タスク |

## LangGraphの2層メモリアーキテクチャを理解する

Long-running Agentの構築において、メモリ管理は最も重要な設計判断の1つです。LangGraphは**短期メモリ**と**長期メモリ**を明確に分離した2層アーキテクチャを採用しています。

### 短期メモリ: Checkpointerによるスレッドスコープ管理

短期メモリはCheckpointerが担当し、1つの会話スレッド内の状態を管理します。LangGraphの公式ドキュメントによると、「Checkpointerはグラフのスーパーステップごとにステートのスナップショットを保存する」仕組みです。

```python
# checkpointer_setup.py
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from typing import TypedDict, Annotated
from operator import add


class AgentState(TypedDict):
    """Long-running Agentの状態定義"""
    messages: Annotated[list[dict], add]  # メッセージ履歴（reducer: 追加結合）
    current_task: str                      # 現在実行中のタスク
    completed_steps: Annotated[list[str], add]  # 完了ステップ（reducer: 追加結合）
    error_count: int                       # エラー回数


async def create_checkpointed_graph(db_uri: str) -> tuple:
    """チェックポイント付きグラフを構築する"""
    checkpointer = AsyncPostgresSaver.from_conn_string(db_uri)
    await checkpointer.setup()  # テーブル作成（初回のみ）

    builder = StateGraph(AgentState)
    # ノード登録（後述）
    builder.add_node("plan", plan_node)
    builder.add_node("execute", execute_node)
    builder.add_node("validate", validate_node)

    builder.add_edge(START, "plan")
    builder.add_edge("plan", "execute")
    builder.add_edge("execute", "validate")
    builder.add_edge("validate", END)

    graph = builder.compile(checkpointer=checkpointer)
    return graph, checkpointer
```

**なぜCheckpointerを本番でPostgreSQLにするのか:**

- InMemorySaverはプロセス終了時に全状態を失います。Long-running Agentでは数時間〜数日の実行が想定されるため、プロセス再起動に耐える永続化が必須です
- PostgresSaverはコネクションプールに対応しており、複数ワーカーから同一スレッドにアクセスできます

> **注意点:** AsyncPostgresSaverでコネクションを手動作成する場合、`autocommit=True`と`row_factory=dict_row`の設定が必須です。これを忘れると`.setup()`でテーブル作成がコミットされず、チェックポイントの読み書きで`KeyError`が発生します。

### 長期メモリ: Storeによるクロススレッド管理

長期メモリはStoreが担当し、複数の会話スレッドをまたいで情報を保持します。LangGraphの公式ドキュメントでは、「長期メモリはカスタムネームスペースにスコープされ、単一のthread_idに限定されない」と説明されています。

```python
# store_setup.py
from langgraph.store.postgres import AsyncPostgresStore
from langchain.embeddings import init_embeddings


async def create_memory_store(db_uri: str) -> AsyncPostgresStore:
    """セマンティック検索対応のメモリストアを構築する"""
    store = await AsyncPostgresStore.from_conn_string(
        db_uri,
        index={
            "embed": init_embeddings("openai:text-embedding-3-small"),
            "dims": 1536,
            "fields": ["content", "summary"],  # 検索対象フィールド
        },
    )
    await store.setup()
    return store
```

**3種類のメモリの使い分け:**

LangGraphの公式ドキュメントでは、長期メモリを以下の3種類に分類しています。

| メモリ種別 | 用途 | 実装パターン | 更新頻度 |
|-----------|------|-------------|---------|
| **セマンティックメモリ** | ユーザー属性・事実の蓄積 | プロフィール更新（前回の値を渡して差分生成） | 低〜中 |
| **エピソードメモリ** | 過去の行動・結果の記録 | Few-shotプロンプトとして活用 | 中〜高 |
| **手続き型メモリ** | エージェントの動作ルール | プロンプトリフレクションで自己改善 | 低 |

## チェックポイントによる状態復元を実装する

Long-running Agentで最も重要な機能の1つが、障害発生時のクラッシュ復旧です。LangGraphのチェックポイント機構を使えば、最後に成功したステップから実行を再開できます。

### フォールトトレラントなノード実装

各ノードでは、リトライロジックとエラーハンドリングを組み込みつつ、チェックポイントに状態を記録します。

```python
# nodes.py
import anthropic
from langgraph.graph import StateGraph
from tenacity import retry, stop_after_attempt, wait_exponential


client = anthropic.Anthropic()


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=30),
)
async def call_claude(messages: list[dict], system: str) -> str:
    """Claude Sonnet 4.6を呼び出す（指数バックオフ付きリトライ）"""
    response = client.messages.create(
        model="claude-sonnet-4-6-20250514",
        max_tokens=8192,
        system=system,
        messages=messages,
    )
    return response.content[0].text


async def plan_node(state: AgentState) -> dict:
    """タスク計画ノード: 実行計画を立案する"""
    system = (
        "あなたはタスク計画エージェントです。"
        "ユーザーの依頼を分析し、実行ステップを計画してください。"
    )
    result = await call_claude(state["messages"], system)
    return {
        "messages": [{"role": "assistant", "content": result}],
        "current_task": "planning_complete",
        "completed_steps": ["plan"],
    }


async def execute_node(state: AgentState) -> dict:
    """実行ノード: 計画に基づいてタスクを実行する"""
    system = (
        "あなたはタスク実行エージェントです。"
        "計画に基づいてタスクを実行し、結果を報告してください。"
    )
    result = await call_claude(state["messages"], system)
    return {
        "messages": [{"role": "assistant", "content": result}],
        "current_task": "execution_complete",
        "completed_steps": ["execute"],
    }


async def validate_node(state: AgentState) -> dict:
    """検証ノード: 実行結果を検証する"""
    system = (
        "あなたは品質検証エージェントです。"
        "実行結果が要件を満たしているか検証してください。"
    )
    result = await call_claude(state["messages"], system)
    return {
        "messages": [{"role": "assistant", "content": result}],
        "current_task": "validation_complete",
        "completed_steps": ["validate"],
    }
```

### クラッシュからの復旧フロー

エージェントが実行中にクラッシュした場合、同じ`thread_id`で再度`invoke`するだけで、最後のチェックポイントから実行が再開されます。

```python
# recovery.py
import asyncio


async def run_with_recovery(graph, thread_id: str, initial_input: dict | None = None):
    """クラッシュ復旧対応のエージェント実行"""
    config = {"configurable": {"thread_id": thread_id}}

    # 既存のチェックポイントがあるか確認
    state = await graph.aget_state(config)

    if state.values:
        # チェックポイントが存在 → 中断地点から再開
        print(f"[復旧] thread_id={thread_id} の中断地点から再開します")
        print(f"  完了済みステップ: {state.values.get('completed_steps', [])}")
        print(f"  次に実行するノード: {state.next}")

        # Noneを渡すと中断地点から実行再開
        result = await graph.ainvoke(None, config=config)
    else:
        # 新規実行
        print(f"[新規] thread_id={thread_id} で新規実行を開始します")
        result = await graph.ainvoke(initial_input, config=config)

    return result


async def main():
    db_uri = "postgresql://user:pass@localhost:5432/langgraph_agent"
    graph, checkpointer = await create_checkpointed_graph(db_uri)

    # 初回実行（途中でクラッシュする可能性あり）
    try:
        result = await run_with_recovery(
            graph,
            thread_id="long-task-001",
            initial_input={
                "messages": [{"role": "user", "content": "大規模データの分析レポートを作成してください"}],
                "current_task": "",
                "completed_steps": [],
                "error_count": 0,
            },
        )
    except Exception as e:
        print(f"[エラー] 実行中に障害が発生: {e}")
        print("[復旧] プロセス再起動後、同じthread_idで再実行してください")
        # 次回起動時に run_with_recovery を同じthread_idで呼べば
        # 最後の成功ステップから再開される


if __name__ == "__main__":
    asyncio.run(main())
```

**なぜこの復旧パターンが有効か:**

- LangGraphのCheckpointerはスーパーステップ単位で状態を保存するため、ノード実行の粒度で復旧ポイントが作られます
- `graph.aget_state(config)`で現在の状態を確認し、`state.next`に次に実行すべきノードが格納されています
- 復旧時は成功済みのノードは再実行されず、失敗ノードから再開されます（LangGraphの公式ドキュメントによる）

> **注意点:** チェックポイントの復旧はスーパーステップ単位のため、ノード内部の部分的な進捗は保存されません。長時間かかるノードは内部で状態を分割し、複数のノードに分けることを検討してください。

## Claude Sonnet 4.6の1Mコンテキストを活かしたエピソードメモリを構築する

Long-running Agentでは、過去の行動とその結果を記憶し、類似タスクに再利用する「エピソードメモリ」が重要です。Claude Sonnet 4.6の1Mトークンコンテキストウィンドウを活用すると、大量のエピソードをプロンプト内に含められます。

### エピソードメモリの記録と想起

```python
# episodic_memory.py
import uuid
from datetime import datetime, timezone

from langgraph.store.postgres import AsyncPostgresStore


async def save_episode(
    store: AsyncPostgresStore,
    agent_id: str,
    task_description: str,
    actions_taken: list[str],
    outcome: str,
    success: bool,
) -> str:
    """エピソード（タスク実行の記録）を保存する"""
    episode_id = str(uuid.uuid4())
    namespace = (agent_id, "episodes")

    episode = {
        "content": f"タスク: {task_description}\n結果: {outcome}",
        "task": task_description,
        "actions": actions_taken,
        "outcome": outcome,
        "success": success,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    await store.aput(namespace, episode_id, episode)
    return episode_id


async def recall_similar_episodes(
    store: AsyncPostgresStore,
    agent_id: str,
    current_task: str,
    limit: int = 5,
) -> list[dict]:
    """セマンティック検索で類似エピソードを想起する"""
    namespace = (agent_id, "episodes")

    # LangGraphのBaseStoreはセマンティック検索を標準サポート
    results = await store.asearch(
        namespace,
        query=current_task,
        limit=limit,
    )

    episodes = []
    for item in results:
        episodes.append({
            "task": item.value.get("task", ""),
            "actions": item.value.get("actions", []),
            "outcome": item.value.get("outcome", ""),
            "success": item.value.get("success", False),
        })

    return episodes


def format_episodes_as_prompt(episodes: list[dict]) -> str:
    """エピソードをFew-shotプロンプトとしてフォーマットする"""
    if not episodes:
        return ""

    lines = ["## 過去の類似タスク実行履歴\n"]
    for i, ep in enumerate(episodes, 1):
        status = "成功" if ep["success"] else "失敗"
        lines.append(f"### エピソード{i}（{status}）")
        lines.append(f"- タスク: {ep['task']}")
        lines.append(f"- 実行アクション: {', '.join(ep['actions'])}")
        lines.append(f"- 結果: {ep['outcome']}")
        lines.append("")

    return "\n".join(lines)
```

### エピソードメモリ統合ノード

エピソードメモリをエージェントのノードに統合し、過去の経験に基づいてタスク計画を改善します。

```python
# memory_integrated_node.py
from langgraph.store.base import BaseStore


async def plan_with_memory(state: AgentState, *, store: BaseStore) -> dict:
    """エピソードメモリを参照してタスク計画を立案するノード"""
    agent_id = state.get("agent_id", "default")
    current_task = state["messages"][-1]["content"]

    # 類似エピソードを想起
    episodes = await recall_similar_episodes(store, agent_id, current_task)
    episode_prompt = format_episodes_as_prompt(episodes)

    system = (
        "あなたはタスク計画エージェントです。\n"
        "過去の類似タスクの実行履歴を参考に、実行計画を立案してください。\n"
        "成功したエピソードのアプローチは積極的に採用し、"
        "失敗したエピソードの原因を回避してください。\n\n"
        f"{episode_prompt}"
    )

    result = await call_claude(state["messages"], system)

    # 今回の計画もエピソードとして記録
    await save_episode(
        store=store,
        agent_id=agent_id,
        task_description=current_task,
        actions_taken=["plan_created"],
        outcome=result[:200],  # 先頭200文字を概要として保存
        success=True,
    )

    return {
        "messages": [{"role": "assistant", "content": result}],
        "current_task": "planning_complete",
        "completed_steps": ["plan"],
    }
```

**Claude Sonnet 4.6を選択する理由:**

- **1Mトークンコンテキスト**: 大量のエピソード履歴をプロンプト内に含められるため、Few-shot学習の精度が向上します
- **context compaction**: ベータ機能として、会話が長くなった際に古いコンテキストを自動要約する機能があり、Long-running Agentのメモリ効率に寄与します（Anthropic公式ドキュメントによる）
- **エージェント性能**: SWE-benchでの高いスコアが報告されており、複雑なタスク分解と実行に適しています

> **制約事項:** エピソードメモリの蓄積が増えるとプロンプトサイズも増大します。セマンティック検索で上位N件に絞り込み、必要に応じてエピソードを要約してからプロンプトに含めるアプローチが推奨されます。1Mトークンのコンテキストであっても、不要な情報を含めると精度低下のリスクがあります。

## 本番環境でのPostgresSaverセットアップとコネクション管理を実装する

Long-running Agentの本番運用では、データベースコネクションの適切な管理が重要です。特に、長時間実行されるワークフローではコネクションタイムアウトが頻出するため、コネクションプールの設定が必須です。

### 本番用AsyncPostgresSaverの構成

```python
# production_setup.py
import asyncio
from psycopg_pool import AsyncConnectionPool
from psycopg.rows import dict_row

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.postgres import AsyncPostgresStore


async def create_production_infrastructure(db_uri: str):
    """本番環境用のインフラストラクチャを構築する"""

    # コネクションプールの設定
    # Long-running Agentでは長時間コネクションを保持するため、
    # プールサイズとタイムアウトの設定が重要
    pool = AsyncConnectionPool(
        conninfo=db_uri,
        min_size=2,        # 最小接続数
        max_size=10,       # 最大接続数（ワーカー数に応じて調整）
        max_idle=300,      # アイドルコネクションの最大生存時間（秒）
        max_lifetime=3600, # コネクションの最大生存時間（秒）
        kwargs={
            "autocommit": True,    # 必須: setup()のCOMMITに必要
            "row_factory": dict_row,  # 必須: 辞書アクセスに必要
        },
    )
    await pool.open(wait=True, timeout=30)

    # Checkpointer: スレッドスコープの状態管理
    checkpointer = AsyncPostgresSaver(pool)
    await checkpointer.setup()

    # Store: クロススレッドの長期メモリ
    store = AsyncPostgresStore(pool)
    await store.setup()

    return checkpointer, store, pool


async def cleanup(pool: AsyncConnectionPool):
    """リソースクリーンアップ"""
    await pool.close()
```

**なぜコネクションプールが必要か:**

- デフォルトのPostgresSaverは単一コネクションを実行全体で保持するため、長時間ワークフローでコネクションタイムアウトが発生します（LangChain公式フォーラムの報告による）
- `max_lifetime=3600`でコネクションを定期的にリフレッシュすることで、タイムアウトを防止します
- 複数のLong-running Agentが並行動作する場合、`max_size`を適切に設定しないとコネクション枯渇が発生します

### グラフのコンパイルとStore統合

```python
# graph_compile.py
from langgraph.graph import StateGraph, START, END


async def build_production_graph(db_uri: str):
    """本番用グラフを構築する"""
    checkpointer, store, pool = await create_production_infrastructure(db_uri)

    builder = StateGraph(AgentState)

    # メモリ統合ノードを使用
    builder.add_node("plan", plan_with_memory)
    builder.add_node("execute", execute_node)
    builder.add_node("validate", validate_node)

    builder.add_edge(START, "plan")
    builder.add_edge("plan", "execute")
    builder.add_edge("execute", "validate")
    builder.add_edge("validate", END)

    # CheckpointerとStoreの両方を指定してコンパイル
    graph = builder.compile(
        checkpointer=checkpointer,
        store=store,
    )

    return graph, pool
```

### Time Travelによるデバッグ

本番環境でエージェントの動作を分析するために、LangGraphのTime Travel機能が有効です。任意のチェックポイントに「巻き戻し」、そこから新しいフォークとして実行を分岐できます。

```python
# time_travel.py
async def debug_agent_execution(graph, thread_id: str):
    """エージェント実行の履歴を分析する"""
    config = {"configurable": {"thread_id": thread_id}}

    # 全チェックポイント履歴を取得
    history = []
    async for snapshot in graph.aget_state_history(config):
        history.append({
            "checkpoint_id": snapshot.config["configurable"]["checkpoint_id"],
            "next_nodes": snapshot.next,
            "completed_steps": snapshot.values.get("completed_steps", []),
            "error_count": snapshot.values.get("error_count", 0),
        })
        print(
            f"  checkpoint={snapshot.config['configurable']['checkpoint_id']}"
            f"  next={snapshot.next}"
            f"  steps={snapshot.values.get('completed_steps', [])}"
        )

    return history


async def replay_from_checkpoint(graph, thread_id: str, checkpoint_id: str):
    """特定のチェックポイントから実行を再開（フォーク）する"""
    config = {
        "configurable": {
            "thread_id": thread_id,
            "checkpoint_id": checkpoint_id,
        }
    }

    # checkpoint_idを指定してinvokeすると、そこから新しいフォークとして実行
    # 成功済みノードは再実行されず、リプレイされる
    result = await graph.ainvoke(None, config=config)
    return result
```

## よくある問題と解決方法

Long-running Agentの運用で頻出する問題と、その解決パターンをまとめます。

| 問題 | 原因 | 解決方法 |
|------|------|----------|
| `asyncio.TimeoutError`で接続失敗 | コネクションプール枯渇 | `max_size`を増加、不要なコネクションの`max_idle`を短縮 |
| 復旧後に同じノードが再実行される | チェックポイントが保存されていない | `autocommit=True`の設定を確認 |
| メモリ検索で関連エピソードが見つからない | embeddingモデルの不一致 | Store作成時の`embed`と検索時のモデルを統一 |
| 長時間実行でOOM発生 | Stateにメッセージが蓄積 | メッセージ履歴のウィンドウサイズを制限（直近N件のみ保持） |
| `KeyError`でチェックポイント読み取り失敗 | `row_factory`未設定 | `row_factory=dict_row`を接続設定に追加 |
| Time Travel時にノードが見つからない | グラフ定義の変更 | チェックポイント作成時のグラフ定義を維持するか、マイグレーション戦略を策定 |

### メッセージ履歴のウィンドウ制限

Long-running Agentではメッセージが無制限に蓄積するとメモリ圧迫の原因になります。以下のパターンで履歴を管理できます。

```python
# message_window.py
from langgraph.graph import MessagesState


def trim_messages(messages: list[dict], max_count: int = 50) -> list[dict]:
    """メッセージ履歴を直近N件に制限する

    先頭のシステムメッセージは常に保持し、
    残りを直近max_count件に制限します。
    """
    if len(messages) <= max_count:
        return messages

    # システムメッセージ（先頭）は保持
    system_msgs = [m for m in messages[:1] if m.get("role") == "system"]
    recent_msgs = messages[-max_count:]

    return system_msgs + recent_msgs


async def execute_with_trimming(state: AgentState) -> dict:
    """メッセージ履歴を制限しつつ実行するノード"""
    trimmed = trim_messages(state["messages"], max_count=30)

    result = await call_claude(trimmed, system="タスクを実行してください。")

    return {
        "messages": [{"role": "assistant", "content": result}],
        "current_task": "execution_complete",
        "completed_steps": ["execute"],
    }
```

## まとめと次のステップ

**まとめ:**

- LangGraphの2層メモリアーキテクチャ（Checkpointer + Store）により、Long-running Agentの状態永続化とクロススレッドメモリを実現できます
- PostgresSaverのコネクションプール設定（`autocommit=True`、`row_factory=dict_row`、適切な`max_size`）が本番運用の安定性に直結します
- エピソードメモリをセマンティック検索で想起し、Few-shotプロンプトとしてClaude Sonnet 4.6に渡すことで、過去の経験に基づく計画改善が可能です
- チェックポイントのフォールトトレランスにより、クラッシュ時は最後の成功ステップから再開でき、Time Travel機能でデバッグも容易です
- メッセージ履歴のウィンドウ制限やエピソードの要約など、メモリ効率の管理も必要です

**次にやるべきこと:**

- 自身のユースケースに合わせてAgentStateの状態スキーマを設計する
- PostgreSQLとコネクションプールの設定をローカル環境で検証する
- エピソードメモリの蓄積と想起の精度を評価し、検索パラメータ（limit、embedding modelの選択）を調整する

## 参考

- [LangGraph Persistence公式ドキュメント](https://docs.langchain.com/oss/python/langgraph/persistence)
- [LangGraph Memory Overview](https://docs.langchain.com/oss/python/langgraph/memory)
- [langgraph-checkpoint-postgres PyPI](https://pypi.org/project/langgraph-checkpoint-postgres/)
- [LangGraph & Redis: Build smarter AI agents with memory & persistence](https://redis.io/blog/langgraph-redis-build-smarter-ai-agents-with-memory-persistence/)
- [Claude Sonnet 4.6公式ページ](https://www.anthropic.com/claude/sonnet)
- [LangGraph Checkpointing Best Practices](https://sparkco.ai/blog/mastering-langgraph-checkpointing-best-practices-for-2025)
- [LangGraph Long-Term Memory Support Changelog](https://changelog.langchain.com/announcements/langgraph-long-term-memory-support)
- [Build durable AI agents with LangGraph and Amazon DynamoDB](https://aws.amazon.com/blogs/database/build-durable-ai-agents-with-langgraph-and-amazon-dynamodb/)

---

:::message
この記事はAI（Claude Code）により自動生成されました。内容の正確性については複数の情報源で検証していますが、実際の利用時は公式ドキュメントもご確認ください。
:::
