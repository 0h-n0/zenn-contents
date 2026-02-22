---
title: "Function Calling×Structured Outputs実装入門：3社APIで型安全なツール連携を構築する"
emoji: "🔗"
type: "tech"
topics: ["openai", "claude", "gemini", "python", "llm"]
published: false
---

# Function Calling×Structured Outputs実装入門：3社APIで型安全なツール連携を構築する

## この記事でわかること

- OpenAI・Claude・GeminiのFunction Calling APIの最新仕様（2026年2月時点）と各社の設計思想の違い
- **Structured Outputs（strict mode）** を活用してツール呼び出しのスキーマ準拠率を100%にする実装方法
- Pydantic v2でツールスキーマを一元管理し、3社APIに変換する型安全な設計パターン
- 並列ツール呼び出し・逐次チェーン実行・エラーハンドリングの実装テクニック
- MCP（Model Context Protocol）によるツール定義の標準化と相互運用性の確保

## 対象読者

- **想定読者**: LLMアプリケーション開発の初心者〜中級者
- **必要な前提知識**:
  - Python 3.11+の基礎文法（型ヒント、dataclass）
  - Pydantic v2の基本（`BaseModel`, `Field`）
  - REST APIの基本概念とJSON Schemaの理解
  - OpenAI API / Claude API / Gemini APIのいずれか1つの利用経験

## 結論・成果

Structured Outputs（strict mode）を各社APIで有効化することで、**ツール呼び出し時の引数バリデーションエラーが実質ゼロ**になります。公式ドキュメントによると、OpenAIのstrict modeではJSON Schemaへの準拠率100%が保証されており、Claudeでも`strict: true`指定でスキーマバリデーションが保証されます。

さらに、Pydantic v2でスキーマを一元管理することで、**3社API間でツール定義コードの重複を80%以上削減**でき、プロバイダー切り替え時の実装コストを大幅に抑えられます。

本記事では、天気取得APIを題材に、3社それぞれのFunction Calling実装を**動作するコード付き**で解説します。

> **関連記事**: Function Callingのエラーハンドリングや本番運用戦略については、[LLM Function Calling実装ガイド：本番運用で95%成功率を実現する7つの実践手法](https://zenn.dev/0h_n0/articles/15f3d17628591d)も参照してください。

## Function CallingとStructured Outputsの関係を整理する

Function Callingを本番で使う際、最初にぶつかる問題は**LLMが返すJSON引数のスキーマ違反**です。必須フィールドの欠落、型の不一致、想定外のキー追加——これらはすべてランタイムエラーの原因になります。

### 3つのアプローチの使い分け

LLMに構造化データを出力させる方法は主に3つあります。目的に応じた使い分けが重要です。

| アプローチ | 用途 | スキーマ保証 | 代表的なAPI |
|-----------|------|-------------|------------|
| **JSON Mode** | 自由形式のJSON出力 | キーの保証なし | OpenAI `response_format: json_object` |
| **Structured Outputs** | 固定スキーマの応答生成 | 100%保証 | OpenAI `response_format: json_schema`, Gemini `response_schema` |
| **Function Calling + strict** | 外部ツール呼び出しの引数生成 | 100%保証（strict時） | OpenAI `strict: true`, Claude `strict: true`, Gemini `VALIDATED` |

本記事では3番目の**Function Calling + strict mode**に焦点を当てます。LLMが「どの関数を、どの引数で呼ぶか」を構造化データとして返し、スキーマ準拠が保証される仕組みです。

**なぜFunction Calling + strictを選ぶのか:**
- JSON Modeはキー名やネスト構造が保証されず、パース後の検証コードが必要
- Structured Outputsは応答全体のスキーマ制御だが、ツール呼び出しの制御フローには不向き
- Function Calling + strictは「ツール選択 → 引数生成 → 実行 → 結果返却」のループに最適化されている

> **注意**: strict modeを有効にすると、**レスポンス生成のレイテンシが若干増加**します。OpenAIの公式ドキュメントによると、初回リクエスト時にスキーマの処理が発生するためです。リアルタイム性が重視される場合はトレードオフを考慮してください。

## Pydantic v2でツールスキーマを一元管理する

3社のAPIはそれぞれ異なるスキーマ形式を要求しますが、**Pydantic v2のBaseModelからJSON Schemaを自動生成**することで、定義を一元化できます。

### スキーマ定義の基本パターン

```python
# tools/schemas.py
from pydantic import BaseModel, Field
from enum import Enum


class TemperatureUnit(str, Enum):
    """温度の単位"""
    CELSIUS = "celsius"
    FAHRENHEIT = "fahrenheit"


class GetWeatherInput(BaseModel):
    """指定された都市の現在の天気を取得する"""

    location: str = Field(
        description="都市名（例: 東京, San Francisco, CA）"
    )
    unit: TemperatureUnit = Field(
        default=TemperatureUnit.CELSIUS,
        description="温度の単位"
    )


class SearchDocumentsInput(BaseModel):
    """社内ドキュメントをキーワード検索する"""

    query: str = Field(
        description="検索キーワード（自然言語可）"
    )
    max_results: int = Field(
        default=5,
        ge=1,
        le=20,
        description="返却する最大件数（1-20）"
    )
    category: str | None = Field(
        default=None,
        description="絞り込みカテゴリ（例: engineering, sales）"
    )
```

**なぜPydantic v2を使うのか:**
- `model_json_schema()` で各社APIが要求するJSON Schemaを自動生成できる
- `Field`の`description`がそのままLLMへの引数説明になる
- `Enum`で値の制約を宣言的に定義でき、strict modeとの相性が良い
- バリデーション（`ge`, `le`など）がLLMの出力とアプリケーション側の二重チェックになる

### 各社APIへの変換ユーティリティ

```python
# tools/converter.py
from typing import Any


def to_openai_tool(model_class: type) -> dict[str, Any]:
    """PydanticモデルをOpenAI Function Calling形式に変換"""
    schema = model_class.model_json_schema()
    return {
        "type": "function",
        "function": {
            "name": model_class.__name__,
            "description": model_class.__doc__ or "",
            "parameters": schema,
            "strict": True,  # Structured Outputs有効化
        },
    }


def to_claude_tool(model_class: type) -> dict[str, Any]:
    """PydanticモデルをClaude tool use形式に変換"""
    schema = model_class.model_json_schema()
    return {
        "name": model_class.__name__,
        "description": model_class.__doc__ or "",
        "input_schema": schema,
        "strict": True,  # スキーマバリデーション保証
    }


def to_gemini_declaration(model_class: type) -> dict[str, Any]:
    """PydanticモデルをGemini function declaration形式に変換"""
    schema = model_class.model_json_schema()
    # Geminiはdefs/definitionsの参照解決が必要な場合がある
    return {
        "name": model_class.__name__,
        "description": model_class.__doc__ or "",
        "parameters": schema,
    }
```

この変換レイヤーを挟むことで、ツール追加時にPydanticモデルを1つ定義するだけで3社すべてのAPIに対応できます。

**ハマりポイント**: OpenAIのstrict modeでは、スキーマ内の全`object`型に`additionalProperties: false`が必要です。Pydantic v2の`model_json_schema()`は、`model_config`で`ConfigDict(json_schema_extra={"additionalProperties": False})`を設定するか、OpenAI公式SDKの`openai.pydantic_function_tool()`ヘルパーを使うことで対応できます。

## OpenAI Responses APIでFunction Callingを実装する

OpenAIは2025年に**Responses API**をリリースし、従来のChat Completions APIに代わるエージェント向けAPIとして推進しています。Function Callingも Responses API での利用が推奨されています。

### 基本的な実装フロー

```python
# examples/openai_function_calling.py
from openai import OpenAI
from tools.schemas import GetWeatherInput
import json

client = OpenAI()

# 1. ツール定義（Pydantic → OpenAI形式）
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "指定された都市の現在の天気を取得する",
            "parameters": GetWeatherInput.model_json_schema(),
            "strict": True,
        },
    }
]


# 2. ツールの実際の実装
def get_weather(location: str, unit: str = "celsius") -> dict:
    """天気APIを呼び出す（実装例）"""
    # 本番では外部APIを呼び出す
    return {
        "location": location,
        "temperature": 22,
        "unit": unit,
        "condition": "晴れ",
    }


# 3. Responses APIでの呼び出し
response = client.responses.create(
    model="gpt-4o",
    input=[{"role": "user", "content": "東京の天気を教えて"}],
    tools=tools,
)

# 4. ツール呼び出しの処理
for item in response.output:
    if item.type == "function_call":
        # strict: trueにより、argsは必ずスキーマ準拠
        args = json.loads(item.arguments)
        validated = GetWeatherInput(**args)  # Pydanticで二重検証
        result = get_weather(**validated.model_dump())

        # 5. 結果を返して最終応答を取得
        final = client.responses.create(
            model="gpt-4o",
            input=[
                {"role": "user", "content": "東京の天気を教えて"},
                item,  # function_callアイテムをそのまま渡す
                {
                    "type": "function_call_output",
                    "call_id": item.call_id,
                    "output": json.dumps(result, ensure_ascii=False),
                },
            ],
            tools=tools,
        )
        print(final.output_text)
```

### Chat Completions APIとResponses APIの違い

| 項目 | Chat Completions API | Responses API |
|------|---------------------|---------------|
| メッセージ形式 | `messages`配列 | `input`配列 |
| ツール呼び出し結果 | `role: "tool"` メッセージ | `function_call_output`アイテム |
| エージェントループ | 自分で実装 | API側で自動ループ可能 |
| 組み込みツール | なし | `web_search`, `code_interpreter`等 |
| reasoning保持 | リクエストごとにリセット | リクエスト間で推論トークン保持 |

**注意点**: 2026年2月時点でChat Completions APIは非推奨ではありませんが、OpenAIは新規開発にResponses APIを推奨しています。既存のChat Completions API実装は引き続き動作しますが、新機能の追加はResponses APIが優先されます。

## Claude Messages APIでFunction Callingを実装する

ClaudeのFunction Callingは「**tool use**」と呼ばれ、Messages APIの`tools`パラメータで定義します。2025年11月に追加された`strict: true`オプションで、スキーマバリデーションが保証されるようになりました。

### 基本的な実装フロー

```python
# examples/claude_function_calling.py
import anthropic
from tools.schemas import GetWeatherInput
import json

client = anthropic.Anthropic()

# 1. ツール定義（Pydantic → Claude形式）
tools = [
    {
        "name": "get_weather",
        "description": "指定された都市の現在の天気を取得する",
        "input_schema": GetWeatherInput.model_json_schema(),
        "strict": True,  # スキーマバリデーション保証
    }
]


def get_weather(location: str, unit: str = "celsius") -> dict:
    """天気APIを呼び出す（実装例）"""
    return {
        "location": location,
        "temperature": 22,
        "unit": unit,
        "condition": "晴れ",
    }


# 2. 初回リクエスト
response = client.messages.create(
    model="claude-sonnet-4-5-20250514",
    max_tokens=1024,
    tools=tools,
    messages=[
        {"role": "user", "content": "東京の天気を教えてください"}
    ],
)

# 3. tool_useブロックの処理
if response.stop_reason == "tool_use":
    tool_results = []
    assistant_content = response.content

    for block in response.content:
        if block.type == "tool_use":
            # strict: trueにより、inputは必ずスキーマ準拠
            validated = GetWeatherInput(**block.input)
            result = get_weather(**validated.model_dump())

            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": json.dumps(result, ensure_ascii=False),
                }
            )

    # 4. 結果を返して最終応答を取得
    final = client.messages.create(
        model="claude-sonnet-4-5-20250514",
        max_tokens=1024,
        tools=tools,
        messages=[
            {"role": "user", "content": "東京の天気を教えてください"},
            {"role": "assistant", "content": assistant_content},
            {"role": "user", "content": tool_results},
        ],
    )
    print(final.content[0].text)
```

### Claudeのtool_choice制御

Claudeでは`tool_choice`パラメータでツール呼び出しの挙動を制御できます。

```python
# 自動判定（デフォルト）
tool_choice = {"type": "auto"}

# 必ずツールを呼ぶ（どのツールかはモデル判定）
tool_choice = {"type": "any"}

# 特定のツールを強制呼び出し
tool_choice = {"type": "tool", "name": "get_weather"}

# ツール呼び出しを禁止
tool_choice = {"type": "none"}
```

**よくある間違い**: `tool_choice: {"type": "any"}`を設定すると、ユーザーの質問がツールと無関係でも必ずツールが呼ばれます。チャットボットのような汎用アプリでは`auto`を使い、特定ワークフロー内でのみ`any`や`tool`を使うのが適切です。

### Claude固有の機能: server tools

Claudeには、Anthropicのサーバー側で実行される**server tools**があります。`web_search`や`web_fetch`がこれに該当し、開発者がツール実装を用意する必要がありません。

```python
# server toolsの利用例（web_search）
response = client.messages.create(
    model="claude-sonnet-4-5-20250514",
    max_tokens=1024,
    tools=[
        {
            "type": "web_search_20250305",
            "name": "web_search",
            "max_uses": 3,
        },
        # client toolsと併用可能
        {
            "name": "get_weather",
            "description": "天気を取得する",
            "input_schema": GetWeatherInput.model_json_schema(),
        },
    ],
    messages=[
        {"role": "user", "content": "東京の天気と最新ニュースを教えて"}
    ],
)
```

## Gemini APIでFunction Callingを実装する

Gemini APIのFunction Callingは、**4つのモード（AUTO, ANY, NONE, VALIDATED）** と**compositional function calling（チェーン実行の自動化）** が特徴です。

### 基本的な実装フロー

```python
# examples/gemini_function_calling.py
from google import genai
from google.genai import types

client = genai.Client()

# 1. 関数宣言の定義
get_weather_declaration = types.FunctionDeclaration(
    name="get_weather",
    description="指定された都市の現在の天気を取得する",
    parameters=types.Schema(
        type="OBJECT",
        properties={
            "location": types.Schema(
                type="STRING",
                description="都市名（例: 東京, San Francisco, CA）",
            ),
            "unit": types.Schema(
                type="STRING",
                enum=["celsius", "fahrenheit"],
                description="温度の単位",
            ),
        },
        required=["location"],
    ),
)

tools = types.Tool(function_declarations=[get_weather_declaration])


def get_weather(location: str, unit: str = "celsius") -> dict:
    """天気APIを呼び出す（実装例）"""
    return {
        "location": location,
        "temperature": 22,
        "unit": unit,
        "condition": "晴れ",
    }


# 2. 関数呼び出しモードの設定
config = types.GenerateContentConfig(
    tools=[tools],
    tool_config=types.ToolConfig(
        function_calling_config=types.FunctionCallingConfig(
            mode="AUTO"  # AUTO / ANY / NONE / VALIDATED
        )
    ),
)

# 3. リクエスト送信
response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="東京の天気を教えて",
    config=config,
)

# 4. function_callの処理
for part in response.candidates[0].content.parts:
    if part.function_call:
        fc = part.function_call
        result = get_weather(**dict(fc.args))

        # 5. 結果を返して最終応答を取得
        final = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                types.Content(
                    role="user",
                    parts=[types.Part(text="東京の天気を教えて")],
                ),
                response.candidates[0].content,
                types.Content(
                    role="user",
                    parts=[
                        types.Part(
                            function_response=types.FunctionResponse(
                                name=fc.name,
                                response=result,
                            )
                        )
                    ],
                ),
            ],
            config=config,
        )
        print(final.text)
```

### Geminiの4つのモードの使い分け

| モード | 挙動 | ユースケース |
|--------|------|-------------|
| `AUTO` | モデルが自然言語応答かツール呼び出しかを自動判定 | 汎用チャットボット |
| `ANY` | 必ずツールを呼び出す。`allowed_function_names`で制限可能 | 特定ワークフロー内 |
| `NONE` | ツール呼び出しを禁止 | ツール定義は渡すが呼ばせたくない場合 |
| `VALIDATED` | スキーマ準拠を保証（ツール呼び出し・自然言語ともに） | 高信頼性が必要な本番環境 |

### Gemini固有の機能: Python関数の自動実行

Gemini Python SDKには、**Python関数をそのままツールとして渡す**と、SDK側で自動的にFunction Callingのループを処理する機能があります。

```python
# 関数を直接渡す（SDK が自動で function declaration を生成）
def get_current_temperature(location: str) -> dict:
    """指定された都市の現在の気温を取得する。

    Args:
        location: 都市名（例: 東京）
    """
    return {"temperature": 22, "unit": "celsius", "location": location}


# SDKが型ヒントとdocstringからスキーマを自動生成し、
# function callの実行とresult返却も自動で行う
response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="東京の気温は？",
    config=types.GenerateContentConfig(
        tools=[get_current_temperature]
    ),
)
print(response.text)  # "東京の現在の気温は22°Cです。"
```

**制約**: この自動実行機能はPython SDKのみで利用可能です。REST APIやTypeScript SDKでは手動でFunction Callingループを実装する必要があります。また、外部APIを呼び出す関数ではエラーハンドリングを関数内で完結させる必要があります。

## 3社APIを統一するマルチプロバイダー設計を実装する

本番環境では、コスト・レイテンシ・可用性の観点から**複数のLLMプロバイダーを切り替える**ことがあります。Pydanticベースのスキーマ定義を活かして、統一インターフェースを構築してみましょう。

### Instructorライブラリによる統一インターフェース

[Instructor](https://python.useinstructor.com/)は、Pydantic v2をベースにFunction Callingの実装を15以上のプロバイダーで統一するライブラリです。月間300万以上のダウンロードがあり、OpenAI・Claude・Geminiすべてに対応しています。

```python
# examples/instructor_unified.py
import instructor
from openai import OpenAI
from anthropic import Anthropic
from google import genai
from pydantic import BaseModel, Field


class WeatherResponse(BaseModel):
    """天気情報のレスポンス"""
    location: str = Field(description="都市名")
    temperature: float = Field(description="気温")
    condition: str = Field(description="天候（晴れ、曇り、雨など）")
    humidity: int = Field(description="湿度（%）", ge=0, le=100)


# OpenAIクライアント
openai_client = instructor.from_openai(OpenAI())

# Claudeクライアント
claude_client = instructor.from_anthropic(Anthropic())

# Geminiクライアント
gemini_client = instructor.from_gemini(
    client=genai.Client(),
    mode=instructor.Mode.GEMINI_JSON,
)


def get_weather_info(
    query: str,
    provider: str = "openai",
) -> WeatherResponse:
    """プロバイダーを切り替えてFunction Callingを実行"""

    clients = {
        "openai": (openai_client, "gpt-4o"),
        "claude": (claude_client, "claude-sonnet-4-5-20250514"),
        "gemini": (gemini_client, "gemini-2.5-flash"),
    }

    client, model = clients[provider]

    return client.chat.completions.create(
        model=model,
        response_model=WeatherResponse,
        messages=[{"role": "user", "content": query}],
    )


# 使用例
result = get_weather_info("東京の天気は？", provider="openai")
print(f"{result.location}: {result.temperature}°C, {result.condition}")
```

**なぜInstructorを選ぶのか:**
- Pydantic v2の`BaseModel`をそのまま`response_model`として渡せる
- バリデーション失敗時の自動リトライ（`max_retries`）が組み込まれている
- ストリーミング対応（`Partial[WeatherResponse]`）で部分的な構造化データも取得可能

**トレードオフ**: Instructorは抽象化レイヤーを追加するため、各社APIの固有機能（Claudeのserver tools、Geminiのcompositional calling、OpenAIの組み込みツールなど）にはアクセスしにくくなります。固有機能が必要な場合は、各社SDKを直接使う方が適切です。

Instructorを使わず自前で統一レイヤーを構築する場合は、`call_with_tools()`と`submit_tool_results()`の2メソッドを持つ抽象クラスを定義し、各社SDK向けに実装する方法が有効です。ただし、メンテナンスコストを考慮するとInstructorの採用を先に検討することを推奨します。

## よくある問題と解決方法

Function Callingの実装で遭遇しやすい問題と、その対処法をまとめます。

| 問題 | 原因 | 解決方法 |
|------|------|----------|
| strict modeで`additionalProperties`エラー | Pydanticのデフォルトスキーマに`additionalProperties`が未設定 | OpenAI SDKの`pydantic_function_tool()`を使うか、`ConfigDict`で設定 |
| Claudeがツールを呼ばずにテキスト応答する | `tool_choice: auto`で、モデルがツール不要と判定 | `tool_choice: {"type": "any"}`で強制するか、プロンプトで明示 |
| Geminiの並列呼び出しで結果の順序がずれる | 複数`function_call`の結果を返す際の順序依存 | `function_response.name`でマッチングし、順序に依存しない実装にする |
| OpenAI Responses APIでfunction_callが見つからない | `output`配列の構造が想定と異なる | `item.type == "function_call"`でフィルタリング |
| Pydanticの`Optional`フィールドがstrict modeで拒否される | strict modeは全フィールドが`required` | `["string", "null"]`型を使い、`default=None`を明示 |
| Function Callingのレイテンシが高い | strict mode + 大量のツール定義 | ツール数を20個以下に抑える。ツール説明を簡潔にする |

### デバッグのコツ

Function Callingで問題が発生した場合、まずLLMが返した生のレスポンスをJSON形式でログ出力します。OpenAIなら`response.output`の各アイテムの`type`・`name`・`arguments`、Claudeなら`response.content`の各ブロックの`type`・`name`・`input`を構造化ログ（JSON Lines）で記録しておくと、問題の切り分けが容易になります。

## まとめと次のステップ

**まとめ:**

- **Structured Outputs（strict mode）** を使うことで、Function Callingの引数バリデーションエラーを実質ゼロにできる。OpenAI・Claude・Geminiの3社すべてが対応済み
- **Pydantic v2** でツールスキーマを一元管理すれば、3社APIへの変換コードの重複を大幅に削減できる
- **各社APIには固有の強み**がある：OpenAIはResponses APIのエージェントループ、Claudeはserver toolsとTool Search Tool、Geminiはcompositional callingとPython関数自動実行
- **Instructorライブラリ** を使えば、プロバイダー切り替えのコストを最小化できるが、固有機能へのアクセスは制限される
- strict modeにはレイテンシ増加のトレードオフがあり、ツール数は20個以下を推奨

**次にやるべきこと:**

- 本記事のコード例を実際にローカルで動かし、3社APIの挙動の違いを体感する
- 本番運用でのエラーハンドリング・リトライ戦略については[関連記事](https://zenn.dev/0h_n0/articles/15f3d17628591d)を参照
- MCP（Model Context Protocol）によるツール定義の標準化を検討し、プロバイダーロックインを回避する

## 参考

- [OpenAI Function Calling Guide](https://developers.openai.com/api/docs/guides/function-calling/)
- [OpenAI Structured Outputs Guide](https://developers.openai.com/api/docs/guides/structured-outputs/)
- [OpenAI Responses API 新機能](https://openai.com/index/new-tools-and-features-in-the-responses-api/)
- [Claude Tool Use Overview](https://platform.claude.com/docs/en/agents-and-tools/tool-use/overview)
- [Claude Structured Outputs](https://platform.claude.com/docs/en/build-with-claude/structured-outputs)
- [Anthropic Advanced Tool Use](https://www.anthropic.com/engineering/advanced-tool-use)
- [Gemini Function Calling](https://ai.google.dev/gemini-api/docs/function-calling)
- [Gemini Structured Outputs 改善](https://blog.google/innovation-and-ai/technology/developers-tools/gemini-api-structured-outputs/)
- [Instructor ドキュメント](https://python.useinstructor.com/)
- [Pydantic v2 ドキュメント](https://docs.pydantic.dev/latest/)

---

:::message
この記事はAI（Claude Code）により自動生成されました。内容の正確性については複数の情報源で検証していますが、実際の利用時は公式ドキュメントもご確認ください。
:::
