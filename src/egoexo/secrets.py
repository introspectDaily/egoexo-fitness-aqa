"""安全地取用凭证。

设计原则（重要，别改）：

1. **绝不打印明文**。所有日志走 :func:`mask`，只出前 6 位 + 长度。
2. **绝不写进仓库**。本模块只从环境/Colab Secret 读，不落盘到项目目录。
3. **绝不作为命令行参数传递**。argv 在 `ps` 里对同机所有进程可见。
4. **优先读 Colab Secret**，回落到 `~/.hf_env`（CLI 专用），再回落到环境变量。

Colab 浏览器路径（每个 Secret 都要在左侧钥匙图标里打开 "Notebook access"）：

    from egoexo.secrets import get_secret
    token = get_secret("HUGGINFACE_ACCESS_KEY_COLAB_CLI")

CLI 路径（`colab exec`）为什么要多一个 `~/.hf_env`：
    ⚠️ **Colab Secrets 只能在浏览器 UI 里读**。CLI 驱动的 kernel 里调
    `userdata.get()` 会直接报 "Secrets can only be fetched when running from the Colab UI."
    而把 token 通过 `colab exec --env KEY=VALUE` 传，又会让它出现在**本机 argv** 里
    （`ps` 可见，违背原则 2）。折中：只在 bootstrap 那一次用 `--env`，把凭证写进 VM 上
    `chmod 600` 的 `~/.hf_env`，之后所有命令都从文件读，argv 里再也不会出现明文。

本地开发：

    source ~/.zprofile     # 里面已 export 同名变量
"""

from __future__ import annotations

import os
from pathlib import Path

# 允许的名字白名单：防止手滑把任意环境变量（比如 AWS_SECRET）当凭证取出来用
KNOWN_SECRETS = {
    "HUGGINFACE_ACCESS_KEY_COLAB_CLI",
    "GITHUB_ACCESS_KEY_FINE_GRAINED_FOR_COALB",
}

# 同义名 -> 规范名。方便 Colab 上用了别的变量名时不必改代码
_ALIASES = {
    "HF_TOKEN": "HUGGINFACE_ACCESS_KEY_COLAB_CLI",
    "HUGGINGFACE_TOKEN": "HUGGINFACE_ACCESS_KEY_COLAB_CLI",
    "HUGGING_FACE_HUB_TOKEN": "HUGGINFACE_ACCESS_KEY_COLAB_CLI",
    "GITHUB_TOKEN": "GITHUB_ACCESS_KEY_FINE_GRAINED_FOR_COALB",
    "GH_TOKEN": "GITHUB_ACCESS_KEY_FINE_GRAINED_FOR_COALB",
}


def mask(secret: str | None, keep: int = 6) -> str:
    """把凭证变成可以安全写进日志/notebook 输出的字符串。

    >>> mask("hf_abcdefghijklmn")
    'hf_abc…(len=17)'
    >>> mask(None)
    '<unset>'
    """
    if not secret:
        return "<unset>"
    head = secret[:keep] if len(secret) > keep else ""
    return f"{head}…(len={len(secret)})"


def _canonical(name: str) -> str:
    return _ALIASES.get(name, name)


def _from_colab(name: str) -> str | None:
    """Colab Secret。不在 Colab 环境时返回 None（不抛异常）。"""
    try:
        from google.colab import userdata  # type: ignore
    except Exception:
        return None
    try:
        value = userdata.get(name)
    except Exception:
        # Secret 不存在 / 未勾选 Notebook access，都会走到这里
        return None
    return value or None


# CLI 驱动时用的近似 dotenv。故意钉死这一个路径，不读任意 .env ——
# 避免误把项目目录里某个陌生人写的 .env 当凭证源（那是典型的凭证投毒场景）。
DOTENV_PATH = Path.home() / ".hf_env"


def _from_dotenv(name: str) -> tuple[str | None, str]:
    """从 ~/.hf_env 读。只接受 `export KEY=VALUE` / `KEY=VALUE` 行。

    权限检查：文件必须只有属主可读写（0o600）。Colab VM 虽然是一次性的，
    但一个全局可读的凭证文件等于没设防，直接拒读并提示修法。
    """
    if not DOTENV_PATH.exists():
        return None, "missing"
    try:
        mode = DOTENV_PATH.stat().st_mode & 0o777
        if mode & 0o077:
            print(
                f"[secrets] ⚠️ {DOTENV_PATH} 权限是 {oct(mode)}，应为 0o600，拒绝读取。\n"
                f"            修: chmod 600 {DOTENV_PATH}"
            )
            return None, "insecure-perm"
        for raw in DOTENV_PATH.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export ") :].strip()
            key, _, value = line.partition("=")
            if key.strip() == name:
                value = value.strip().strip('"').strip("'")
                if value:
                    return value, "ok"
    except OSError:
        return None, "unreadable"
    return None, "not-found"


def get_secret(name: str, required: bool = True) -> str | None:
    """按 Colab Secret -> 环境变量 的顺序取凭证。

    取不到时：required=True 抛 RuntimeError（带排查提示），否则返回 None。
    """
    canonical = _canonical(name)

    value = _from_colab(canonical)
    source = "colab-secret"
    if not value:
        value, why = _from_dotenv(canonical)
        source = f"dotenv({why})"
    if not value:
        value = os.environ.get(canonical)
        source = "env"

    if not value:
        if required:
            hint = (
                f"取不到凭证 '{canonical}'。按运行方式选排查路径：\n"
                f"  [浏览器 notebook] 左侧 🔑 图标里确认名字是 '{canonical}'，"
                f"且该行右侧 'Notebook access' 已打开（默认是关的）\n"
                f"  [colab exec CLI]     Secrets 在 CLI kernel 里读不到（Colab 的限制），"
                f"必须走 {DOTENV_PATH}，例如：\n"
                f"                        printf 'export {canonical}=%s\\n' \"$TOKEN\""
                f" > {DOTENV_PATH} && chmod 600 {DOTENV_PATH}\n"
                f"  [本地]              `source ~/.zprofile` 后再跑\n"
                f"  别名也认: {[k for k, v in _ALIASES.items() if v == canonical]}"
            )
            raise RuntimeError(hint)
        return None

    if canonical not in KNOWN_SECRETS:
        # 不阻止，但要留痕：提醒调用方这不在白名单里
        print(f"[secrets] 注意: '{canonical}' 不在已知凭证白名单内，来源={source}")
    return value


def hf_login(quiet: bool = True) -> str | None:
    """登录 HuggingFace Hub。返回 token（调用方若要打日志请用 mask()）。

    用 login() 而不是设 HF_TOKEN：login() 写到 ~/.cache/huggingface/token，
    后续 huggingface_hub / hf_transfer 的下载都会自动带上，且不出现在进程环境里。
    """
    token = get_secret("HUGGINFACE_ACCESS_KEY_COLAB_CLI", required=False)
    if not token:
        return None
    try:
        from huggingface_hub import login
    except ImportError:
        # 还没装依赖，退回环境变量方式（功能等价，只是会进 environ）
        os.environ["HF_TOKEN"] = token
        return token

    login(token=token, add_to_git_credential=False, skip_if_logged_in=False)
    if not quiet:
        print(f"[secrets] HF 已登录: {mask(token)}")
    return token


def describe() -> dict[str, str]:
    """给 notebook 用的一行式自检：只返回掩码，安全可打印。"""
    return {name: mask(get_secret(name, required=False)) for name in sorted(KNOWN_SECRETS)}
