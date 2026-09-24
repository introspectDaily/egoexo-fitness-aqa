"""安全地取用凭证。

设计原则（重要，别改）：

1. **绝不打印明文**。所有日志走 :func:`mask`，只出前 6 位 + 长度。
2. **绝不写进仓库**。本模块只从环境/Colab Secret 读，不落盘到项目目录。
3. **绝不作为命令行参数传递**。argv 在 `ps` 里对同机所有进程可见。
4. **优先读 Colab Secret**，回落到环境变量；两者都没有才报错。

Colab 侧用法（每个 Secret 都要在左侧钥匙图标里打开 "Notebook access"）：

    from egoexo.secrets import get_secret
    token = get_secret("HUGGINFACE_ACCESS_KEY_COLAB_CLI")

本地侧用法：

    source ~/.zprofile     # 里面已 export 同名变量
"""

from __future__ import annotations

import os

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


def get_secret(name: str, required: bool = True) -> str | None:
    """按 Colab Secret -> 环境变量 的顺序取凭证。

    取不到时：required=True 抛 RuntimeError（带排查提示），否则返回 None。
    """
    canonical = _canonical(name)

    value = _from_colab(canonical)
    source = "colab-secret"
    if not value:
        value = os.environ.get(canonical)
        source = "env"

    if not value:
        if required:
            hint = (
                f"取不到凭证 '{canonical}'。排查顺序：\n"
                f"  1. Colab: 左侧 🔑 图标里确认 Secret 名字是 '{canonical}'，"
                f"且该行右侧的 'Notebook access' 已打开（默认是关的）\n"
                f"  2. 本地: `source ~/.zprofile` 后再跑\n"
                f"  3. 检查是否被别名覆盖: 本模块也认 {[k for k, v in _ALIASES.items() if v == canonical]}"
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
