# Beads + GitHub Issues 同期メモ

このrepoでは Beads が一次情報（source of truth）で、必要に応じて GitHub Issues を「外部参照」として紐付けます。

## 前提

- Beads: `.beads/issues.jsonl` / `.beads/interactions.jsonl` をgit管理
- Beads sync ブランチ: `beads-sync`（`.beads/config.yaml` の `sync-branch`）
- GitHub CLI: `gh` がログイン済み

## 2つの環境（2台）での基本運用

各環境で作業前後にこれを徹底します。

```bash
git pull --rebase
bd sync --no-push   # まずはローカル統合（ネットワーク状況に応じて）
```

作業が終わったら（共有したい状態になったら）:

```bash
bd sync             # pull/merge/commit/push まで実行
git push
```

競合したら:

```bash
bd resolve-conflicts
bd sync
```

## GitHub Issue へのリンク（手動）

```bash
bd update <id> --external-ref gh-123
```

## GitHub Issue へのリンク（自動：タイトル一致 or 作成）

このrepoには Beads issue を GitHub issue にリンク/作成するスクリプトがあります。

```bash
python3 tools/link_beads_to_github.py --dry-run
python3 tools/link_beads_to_github.py
```

- 既に `external_ref` がある Beads issue はスキップします
- タイトル完全一致の GitHub issue があればそこへリンクします
- なければ GitHub issue を作成してリンクします

### gh のログイン

`gh auth status` で失敗する場合:

```bash
gh auth login -h github.com
```

