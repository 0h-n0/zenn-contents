---
title: "Gemini 2.0マルチモーダルAPI実践ガイド：画像・音声・動画をPythonで統合処理する"
emoji: "🔮"
type: "tech"
topics: ["gemini", "multimodal", "python", "googlecloud", "ai"]
published: false
---

# Gemini 2.0マルチモーダルAPI実践ガイド：画像・音声・動画をPythonで統合処理する

## この記事でわかること

- Gemini 2.0系のマルチモーダル機能（画像・音声・動画の入出力）の全体像と各モダリティの使い分け
- `google-genai` SDKを使った画像理解・ネイティブ画像生成・音声理解・TTS・動画理解の実装方法
- Multimodal Live APIによるリアルタイム双方向対話の構築パターン
- トークン消費量とコストの見積もり方法、本番運用での制約と対策
- Gemini 2.0から2.5/3.0への移行パスと互換性の注意点

## 対象読者

- **想定読者**: 中級者のPython開発者で、LLM APIの基本的な利用経験がある方
- **必要な前提知識**:
  - Python 3.10+の基本的な非同期処理（`async/await`）
  - REST APIまたはWebSocketの基礎知識
  - LLMのトークンとプロンプトの概念理解

## 結論・成果

Gemini 2.0系のマルチモーダルAPIを活用すると、画像・音声・動画を**1つのAPI呼び出し**で統合的に処理できます。Google公式ベンチマークによると、Gemini 2.0 Flashは1.5 Proを上回る精度を**2倍の速度**で達成しており、動画理解ではVideoMMEで84.7%の精度が報告されています。コスト面では入力$0.10/Mトークン・出力$0.40/Mトークンと、マルチモーダル処理としては低コストな価格設定です。

ただし、Gemini 2.0 Flashは**2026年3月31日に非推奨化**が予定されているため、本記事では2.5 Flashへの移行パスも含めて解説します。APIの基本的な呼び出しパターンは2.0と2.5で共通しているため、ここで学ぶ実装はそのまま移行に活用できます。

## Gemini 2.0マルチモーダルの全体像を理解する

Gemini 2.0は、テキストだけでなく画像・音声・動画を**ネイティブに**理解・生成できるマルチモーダルモデルです。従来のLLMが外部ツールを介してマルチモーダル処理を行っていたのに対し、Geminiは単一モデル内で複数のモダリティを直接処理します。

### 対応モダリティの一覧

| モダリティ | 入力 | 出力 | 主な用途 |
|-----------|------|------|----------|
| テキスト | ✅ | ✅ | 通常のテキスト生成・要約 |
| 画像 | ✅ | ✅（ネイティブ生成） | 画像理解・キャプション・画像生成 |
| 音声 | ✅ | ✅（TTS） | 音声文字起こし・音声合成 |
| 動画 | ✅ | ❌ | 動画要約・シーン分析・QA |
| PDF | ✅ | ❌ | ドキュメント解析・情報抽出 |

### モデルの進化と選定基準

Gemini 2.0の登場以降、モデルは急速に進化しています。2026年2月時点でのモデル選定の指針を整理します。

| モデル | コンテキスト長 | 特徴 | 推奨用途 |
|--------|---------------|------|----------|
| Gemini 2.0 Flash | 100万トークン | 高速・低コスト | 2026/3/31非推奨、移行推奨 |
| Gemini 2.5 Flash | 100万トークン | 2.0 Flash上位互換 | 汎用マルチモーダル処理 |
| Gemini 2.5 Pro | 100万トークン | 高精度・思考機能 | 高精度が必要なタスク |
| Gemini 3 Flash | 100万トークン | 最新世代 | 新規開発推奨 |

**なぜ2.0 Flashの理解が重要か:**

Gemini 2.5/3.0系のマルチモーダルAPIは、2.0で導入されたアーキテクチャとAPI設計をそのまま継承しています。モデルIDを変更するだけで移行できるケースが多いため、2.0の実装パターンを理解しておくことは2.5/3.0への移行においても有用です。

**注意点:**
> Gemini 2.0 Flashは2026年3月31日に非推奨化が予定されています。新規プロジェクトではGemini 2.5 Flash以降のモデルを使用し、既存プロジェクトでは移行計画を立ててください。公式ドキュメントによると、モデルIDの変更（`gemini-2.0-flash` → `gemini-2.5-flash`）で基本的な移行が可能です。

## 画像理解・生成を実装する

Geminiのマルチモーダル機能のうち、もっとも実用的な画像の入出力処理を実装してみましょう。

### 環境構築

まず、`google-genai` SDKをインストールします。

```bash
pip install google-genai pillow
```

APIキーは環境変数で管理します。

```bash
export GOOGLE_API_KEY="your-api-key-here"
```

### 画像理解（Image Understanding）を実装する

ローカルファイルの画像をGeminiに渡して内容を分析させる基本パターンです。

```python
# image_understanding.py
from google import genai
from google.genai import types

def analyze_image(image_path: str, prompt: str) -> str:
    """画像を分析してテキストで結果を返す"""
    client = genai.Client()

    with open(image_path, "rb") as f:
        image_bytes = f.read()

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[
            types.Part.from_bytes(
                data=image_bytes,
                mime_type="image/jpeg",
            ),
            prompt,
        ],
    )
    return response.text


if __name__ == "__main__":
    result = analyze_image(
        "sample.jpg",
        "この画像に写っている物体をすべて列挙し、それぞれの位置関係を説明してください。",
    )
    print(result)
```

**なぜ`Part.from_bytes`を使うのか:**

- `inline_data`を直接構築する方法もありますが、`Part.from_bytes`はMIMEタイプの指定ミスを防ぎやすい
- 20MB以下のファイルに適しており、それ以上の場合はFiles APIを使用する

大きなファイルの場合は、Files APIを使ったアップロードが推奨されます。

```python
def analyze_large_image(image_path: str, prompt: str) -> str:
    """大きな画像ファイルをFiles API経由で分析する"""
    client = genai.Client()

    # Files APIでアップロード（20MB超のファイル向け）
    uploaded_file = client.files.upload(file=image_path)

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[uploaded_file, prompt],
    )
    return response.text
```

### ネイティブ画像生成を実装する

Gemini 2.0 Flashで導入されたネイティブ画像生成は、テキストと画像を**インターリーブ（交互配置）**で出力できる点が特徴です。

```python
# image_generation.py
import base64
from pathlib import Path

from google import genai
from google.genai import types


def generate_story_with_images(prompt: str, output_dir: str = "output") -> None:
    """テキストと画像を交互に生成するストーリーを作成する"""
    client = genai.Client()
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    response = client.models.generate_content(
        model="gemini-2.0-flash-exp",
        contents=prompt,
        config=types.GenerateContentConfig(
            response_modalities=["Text", "Image"],
        ),
    )

    image_count = 0
    for part in response.candidates[0].content.parts:
        if part.text is not None:
            print(part.text)
        elif part.inline_data is not None:
            image_count += 1
            image_path = output_path / f"image_{image_count}.png"
            image_data = base64.b64decode(part.inline_data.data)
            image_path.write_bytes(image_data)
            print(f"[画像保存: {image_path}]")


if __name__ == "__main__":
    generate_story_with_images(
        "日本の四季をテーマに、春・夏・秋・冬それぞれの風景を"
        "イラスト付きで紹介する短いストーリーを生成してください。"
    )
```

**制約条件:**
> ネイティブ画像生成は`gemini-2.0-flash-exp`（実験版）で利用可能です。2026年2月時点では正式GA版ではないため、本番環境での利用にはリスクがあります。画像生成の品質や一貫性は、専用の画像生成モデル（Imagen 3など）と比較すると劣る場合があります。テキストと画像のインターリーブ出力が必要なユースケースで特に有効です。

## 音声処理を実装する

Geminiは音声の理解（入力）と生成（出力・TTS）の両方に対応しています。実際のコードで見ていきましょう。

### 音声理解（Audio Understanding）を実装する

音声ファイルをGeminiに渡して、文字起こしや内容分析を行います。

```python
# audio_understanding.py
from google import genai
from google.genai import types


def transcribe_audio(audio_path: str) -> str:
    """音声ファイルを文字起こしする"""
    client = genai.Client()

    with open(audio_path, "rb") as f:
        audio_bytes = f.read()

    # ファイル拡張子からMIMEタイプを判定
    mime_map = {
        ".mp3": "audio/mp3",
        ".wav": "audio/wav",
        ".flac": "audio/flac",
        ".aac": "audio/aac",
        ".ogg": "audio/ogg",
    }
    suffix = audio_path.rsplit(".", 1)[-1]
    mime_type = mime_map.get(f".{suffix}", "audio/mp3")

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[
            types.Part.from_bytes(data=audio_bytes, mime_type=mime_type),
            "この音声の内容を日本語で文字起こししてください。"
            "話者が複数いる場合は話者を区別してください。",
        ],
    )
    return response.text


def analyze_audio_sentiment(audio_path: str) -> str:
    """音声の感情・トーンを分析する"""
    client = genai.Client()

    uploaded = client.files.upload(file=audio_path)

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[
            uploaded,
            "この音声の話者の感情やトーンを分析してください。"
            "JSON形式で{speaker, emotion, confidence}を返してください。",
        ],
    )
    return response.text
```

**トークン消費の目安:** 音声は約**32トークン/秒**で処理されます。1分の音声で約1,920トークン、1時間で約115,200トークンです。最大9.5時間の音声を1回のリクエストで処理できます。

### テキスト音声変換（TTS）を実装する

Gemini 2.5 Flash Preview TTSを使って、テキストから音声を生成します。

```python
# tts_generation.py
import wave
from google import genai
from google.genai import types


def text_to_speech(
    text: str,
    output_path: str = "output.wav",
    voice_name: str = "Kore",
) -> str:
    """テキストを音声に変換してWAVファイルとして保存する"""
    client = genai.Client()

    response = client.models.generate_content(
        model="gemini-2.5-flash-preview-tts",
        contents=text,
        config=types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name=voice_name,
                    )
                )
            ),
        ),
    )

    pcm_data = response.candidates[0].content.parts[0].inline_data.data
    _save_wav(output_path, pcm_data)
    return output_path


def _save_wav(
    filename: str,
    pcm: bytes,
    channels: int = 1,
    rate: int = 24000,
    sample_width: int = 2,
) -> None:
    """PCMデータをWAVファイルとして保存する"""
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(rate)
        wf.writeframes(pcm)


if __name__ == "__main__":
    text_to_speech(
        "Geminiのテキスト音声変換機能を使えば、"
        "自然な日本語音声を生成できます。",
        voice_name="Kore",
    )
    print("音声ファイルを生成しました: output.wav")
```

TTS機能の主な仕様は以下の通りです。

| 項目 | 仕様 |
|------|------|
| 対応モデル | gemini-2.5-flash-preview-tts, gemini-2.5-pro-preview-tts |
| 音声の種類 | 30種類のプリセット音声 |
| 対応言語 | 80言語以上（日本語対応） |
| 出力形式 | PCM 24kHz, 16bit, モノラル |
| 最大入力 | 32,000トークン |
| 話者数 | 最大2名（マルチスピーカー対応） |

**注意点:**
> TTS専用モデルは**テキスト入力のみ**を受け付けます。音声や画像を入力して音声出力を得たい場合は、Multimodal Live APIを使用する必要があります。

## 動画理解を実装する

Geminiの動画理解機能は、映像と音声の両方を同時に分析できます。

### 動画の内容を分析する

```python
# video_understanding.py
from google import genai
from google.genai import types


def summarize_video(video_path: str) -> str:
    """動画の内容を要約する"""
    client = genai.Client()

    # Files APIでアップロード（動画は通常20MB超のためFiles API推奨）
    video_file = client.files.upload(file=video_path)

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[
            video_file,
            "この動画の内容を以下の観点で要約してください：\n"
            "1. 主なトピック\n"
            "2. キーポイント（3-5個）\n"
            "3. 登場人物や話者の情報",
        ],
    )
    return response.text


def analyze_video_segment(
    video_path: str,
    start_sec: int,
    end_sec: int,
    prompt: str,
) -> str:
    """動画の特定区間を分析する"""
    client = genai.Client()

    video_file = client.files.upload(file=video_path)

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[
            types.Content(
                parts=[
                    types.Part(
                        file_data=types.FileData(
                            file_uri=video_file.uri,
                            mime_type=video_file.mime_type,
                        ),
                        video_metadata=types.VideoMetadata(
                            start_offset=f"{start_sec}s",
                            end_offset=f"{end_sec}s",
                        ),
                    ),
                    types.Part(text=prompt),
                ]
            )
        ],
    )
    return response.text
```

**トークン消費と制約:**

動画処理のトークン消費量は以下の通りです。

| 解像度設定 | トークン/秒 | 1分あたり | 10分あたり |
|-----------|------------|----------|-----------|
| デフォルト | 約300 | 約18,000 | 約180,000 |
| 低解像度 | 約100 | 約6,000 | 約60,000 |

対応フォーマット: MP4, MPEG, MOV, AVI, FLV, WebM, WMV, 3GPP

**ハマりポイント:**
> 動画のアップロード後、Files APIで処理状態が`ACTIVE`になるまで待つ必要があります。大きなファイル（数百MB）の場合、処理に数分かかることがあります。アップロード直後にリクエストを送ると、ファイルが見つからないエラーが発生する場合があります。

## Multimodal Live APIでリアルタイム対話を構築する

Multimodal Live APIは、WebSocketベースのステートフルなAPIで、音声や映像のリアルタイム双方向対話を実現します。製造業の品質検査や、カスタマーサポートの自動化などの用途で活用が進んでいます。

### 基本的なアーキテクチャ

```
クライアント ←→ WebSocket ←→ Gemini Live API
  (音声/映像)    (双方向)      (リアルタイム処理)
```

Multimodal Live APIの主な特徴は以下の通りです。

- **双方向ストリーミング**: クライアントからの音声・映像入力と、サーバーからの音声・テキスト応答が同時に流れる
- **ステートフル**: 会話の文脈を保持し、連続的な対話が可能
- **低レイテンシ**: WebSocket接続により、HTTPリクエスト/レスポンスよりも低遅延で通信

### 簡易的なリアルタイム音声対話の実装

```python
# live_api_example.py
import asyncio
from google import genai
from google.genai import types


async def run_live_session() -> None:
    """Multimodal Live APIで音声対話セッションを実行する"""
    client = genai.Client()

    config = types.LiveConnectConfig(
        response_modalities=["AUDIO", "TEXT"],
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(
                    voice_name="Puck",
                )
            )
        ),
    )

    async with client.aio.live.connect(
        model="gemini-2.0-flash-live-001",
        config=config,
    ) as session:
        # テキストメッセージを送信
        await session.send_client_content(
            turns=types.Content(
                role="user",
                parts=[types.Part(text="こんにちは。今日の天気について教えてください。")],
            )
        )

        # レスポンスを受信
        async for message in session.receive():
            if message.text is not None:
                print(f"テキスト応答: {message.text}")
            if message.data is not None:
                print(f"音声データ受信: {len(message.data)} bytes")


if __name__ == "__main__":
    asyncio.run(run_live_session())
```

**制約条件:**
> Multimodal Live APIは2026年2月時点でまだ制限付きの提供です。セッションの最大持続時間や、同時接続数に制限がある場合があります。本番利用前に、Google Cloud の公式ドキュメントで最新の制限事項を確認してください。

## 本番運用での注意点とコスト最適化を実践する

マルチモーダルAPIを本番運用する際に知っておくべきコスト構造と最適化手法を解説します。

### コスト構造の理解

Gemini 2.0 Flashの料金体系は以下の通りです（公式ドキュメントより）。

| 項目 | 料金（100万トークンあたり） |
|------|---------------------------|
| テキスト入力 | $0.10 |
| 画像入力 | $0.10（258トークン/画像） |
| 音声入力 | $0.10（32トークン/秒） |
| 動画入力 | $0.10（300トークン/秒） |
| テキスト出力 | $0.40 |

**コスト試算の具体例:**

1分の動画を処理する場合:
- 入力トークン: 300 × 60 = 18,000トークン
- コスト: 18,000 / 1,000,000 × $0.10 = **$0.0018**（約0.27円）

これにテキストプロンプトと出力トークンが加算されますが、動画理解のコストは非常に低いことがわかります。

### コスト最適化の4つの手法

1. **動画の低解像度設定**: `media_resolution`パラメータで解像度を下げると、トークン消費を1/3に削減可能
2. **動画クリッピング**: `start_offset`/`end_offset`で必要な区間だけを処理
3. **キャッシュの活用**: 同じファイルに対する複数の質問は、Files APIでアップロード済みファイルを再利用
4. **モデルの使い分け**: 単純なタスクはFlash、高精度が必要なタスクのみProを使用

```python
# コスト最適化の実装例：低解像度での動画処理
def analyze_video_low_cost(video_path: str, prompt: str) -> str:
    """低解像度設定でコストを抑えた動画分析"""
    client = genai.Client()
    video_file = client.files.upload(file=video_path)

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[video_file, prompt],
        config=types.GenerateContentConfig(
            media_resolution=types.MediaResolution.MEDIA_RESOLUTION_LOW,
        ),
    )
    return response.text
```

### よくある問題と解決方法

| 問題 | 原因 | 解決方法 |
|------|------|----------|
| `FileNotFoundError` | アップロード直後のリクエスト | `files.get()`で`ACTIVE`状態を確認してからリクエスト |
| `InvalidArgument` | MIMEタイプの不一致 | ファイル拡張子とMIMEタイプの対応を正しく設定 |
| `ResourceExhausted` | レート制限超過 | 指数バックオフ付きリトライを実装 |
| 画像生成の品質低下 | 実験版モデルの制約 | プロンプトの詳細化、または専用モデル（Imagen 3）の併用 |
| TTS出力の不自然さ | 音声名の選択ミス | 30種類の音声から用途に合ったものを選択 |

### Gemini 2.0から2.5への移行チェックリスト

2026年3月31日の非推奨化に向けた移行手順は以下の通りです。

1. **モデルIDの変更**: `gemini-2.0-flash` → `gemini-2.5-flash`
2. **テストの実行**: 既存のプロンプトとレスポンスの互換性を確認
3. **新機能の活用**: 2.5系で追加された思考（Thinking）機能の検討
4. **料金の確認**: 2.5 Flashは2.0 Flashと同等のコスト競争力を維持（公式ブログより）

**トレードオフ:**
> モデルIDを変更するだけで基本的な移行は完了しますが、レスポンスの微妙な違い（表現の変化、トークン数の増減）が発生する可能性があります。特にJSON出力のスキーマに依存している場合は、出力形式の検証を事前に行ってください。

## まとめと次のステップ

**まとめ:**

- Gemini 2.0系は画像・音声・動画の入出力を**1つのAPI**で統合的に処理でき、`google-genai` SDKで簡潔に実装可能
- ネイティブ画像生成とTTSにより、マルチモーダルな**出力**も可能になった（実験的機能を含む）
- 動画処理のコストは1分あたり約$0.0018と低コストだが、トークン消費（300トークン/秒）に注意が必要
- Gemini 2.0 Flashは2026年3月31日に非推奨化されるため、2.5 Flash以降への移行が必要
- APIの基本パターンは2.0と2.5/3.0で共通しており、モデルIDの変更で移行可能

**次にやるべきこと:**

- [Google AI Studio](https://aistudio.google.com/)でAPIキーを取得し、画像理解のコード例を実行してみる
- 自身のユースケースでトークン消費量を計測し、コスト見積もりを行う
- Gemini 2.0 Flashを使用中の場合は、2.5 Flashへの移行テストを開始する

**関連記事:**
- [Gemini 3.1 Proで構築するマルチエージェント協調コーディングの実践手法](https://zenn.dev/0h_n0/articles/a7935e0412571c)
- [GeminiとClaudeを使い分けるマルチLLMルーティング実装ガイド](https://zenn.dev/0h_n0/articles/ecc929fbeb5871)

## 参考

- [Gemini API Models - Google AI for Developers](https://ai.google.dev/gemini-api/docs/models)
- [Image Understanding - Gemini API](https://ai.google.dev/gemini-api/docs/image-understanding)
- [Video Understanding - Gemini API](https://ai.google.dev/gemini-api/docs/video-understanding)
- [Audio Understanding - Gemini API](https://ai.google.dev/gemini-api/docs/audio)
- [Speech Generation (TTS) - Gemini API](https://ai.google.dev/gemini-api/docs/speech-generation)
- [Gemini 2.0 Flash Native Image Generation - Google Developers Blog](https://developers.googleblog.com/en/experiment-with-gemini-20-flash-native-image-generation/)
- [Gemini 2.0 Flash Deprecation Migration Guide](https://www.isumsoft.com/internet/gemini-2-flash-deprecation-migration-guide.html)
- [Gemini Developer API Pricing](https://ai.google.dev/gemini-api/docs/pricing)
- [Gemini 2.5 Flash GA - Google Cloud Blog](https://cloud.google.com/blog/products/ai-machine-learning/gemini-2-5-flash-lite-flash-pro-ga-vertex-ai)

---

:::message
この記事はAI（Claude Code）により自動生成されました。内容の正確性については複数の情報源で検証していますが、実際の利用時は公式ドキュメントもご確認ください。
:::
