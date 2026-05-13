# GitHub 与超算持续同步说明

## 1. 目标

将当前 `SkillRL` 工作目录同步到：

```text
https://github.com/yilinpotato/SkillRL
```

并提供一套超算上可持续同步的脚本。

## 2. 新增文件

- `scripts/git_sync_to_github.sh`
- `scripts/git_continuous_sync.sh`
- `.gitignore`

## 3. 一次性同步

在仓库根目录执行：

```bash
bash scripts/git_sync_to_github.sh
```

可选参数：

```bash
TARGET_REMOTE_NAME=github
TARGET_REMOTE_URL=https://github.com/yilinpotato/SkillRL.git
TARGET_BRANCH=main
COMMIT_MESSAGE="sync: initial push"
PUSH=1
```

只提交不推送：

```bash
PUSH=0 bash scripts/git_sync_to_github.sh
```

## 4. 超算上持续同步

在超算节点上执行：

```bash
SYNC_INTERVAL_SECONDS=300 \
SYNC_PUSH=1 \
bash scripts/git_continuous_sync.sh
```

含义：

- 每 `300` 秒检查一次工作区
- 若检测到变更，则自动：
  - `git add -A`
  - `git commit`
  - `git push`

## 5. GitHub 认证

如果仓库是私有仓库，或当前环境未保存 GitHub 凭证，需要提前配置认证。

常见方式：

### 5.1 HTTPS + Personal Access Token

第一次 push 时输入：

- GitHub 用户名
- Personal Access Token

如果希望超算上免交互，建议先配置：

```bash
git config --global credential.helper store
```

然后手动执行一次：

```bash
git push https://github.com/yilinpotato/SkillRL.git HEAD:main
```

输入成功后，凭证会保存在本地。

### 5.2 SSH

如果超算支持 SSH key，建议把远端改成：

```bash
git remote set-url github git@github.com:yilinpotato/SkillRL.git
```

然后确保：

- `~/.ssh/id_rsa` 或 `~/.ssh/id_ed25519` 已配置
- 公钥已加入 GitHub

## 6. 当前 `.gitignore` 行为

已忽略以下内容：

- `__pycache__/`
- `outputs/`
- `skillrl_outputs/`
- `wandb/`
- `diagnostics/`
- 本地数字目录 `1031360/`, `2953823/`, `3637881/`

这能避免把训练产物、缓存文件和本地临时目录推到仓库。
