# fallible-by-design

**人もAIも必ず間違える。問題は「どれだけ」ではなく「どう」間違えるか。**

AIの導入では「AIは間違えるから使えない」という議論がよく起きます。しかし人間も間違えます。
比べるべきは「AIか、完璧か」ではなく、「人間とAIが、それぞれどう間違えるか」です。
間違え方が違えば、必要なチェック体制も変わります。

このリポジトリは、その違いを再現可能なシミュレーションで確かめるためのものです。
すべての実験は公開データまたはシミュレーションのみで構成しています。

## 実験一覧

| # | テーマ | 主な結論 | 記事 |
| --- | --- | --- | --- |
| 01 | [誤りの相関](experiments/01_error_correlation/) | 平均エラー率が同じでも、まとめて間違えるAIは「悪い日」が約2.2倍。同じAIを2回通す二重チェックは、人間2人の約5.7倍すり抜ける | Zenn（準備中）／ note（準備中） |
| 02 | 自動化バイアス（予定） | 確認する人間の見逃しまで設計に入れる | |
| 03 | 信頼度による振り分け（予定） | どの出力を人間に回すか | |

## すぐに試す

Docker があれば、ローカル環境を汚さずに再現できます。

```bash
git clone https://github.com/shisei-kbrg322/fallible-by-design.git
cd fallible-by-design
docker compose run --rm test   # テスト
docker compose run --rm sim    # 実験01を実行し、図とCSVを出力
```

パラメータは `experiments/01_error_correlation/config.yaml` で変えられます。

## 構成

```
src/fallible/        共通ロジック（判定者モデル・組み合わせ方・指標・図の体裁）
experiments/NN_*/    実験ごとの config・実行スクリプト・結果
tests/               モデルの性質（平均エラー率の保存・理論値との一致）のテスト
```

## ライセンス

MIT

## 解説記事

- [Zenn：技術者向けの詳しい解説]https://zenn.dev/shisei_kbrg322/articles/e4808f04b8e8be
- [note：数式なしで、組織としての向き合い方をまとめた記事]https://note.com/shisei_kbrg322/n/nd983de113eeb