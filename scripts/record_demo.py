# -*- coding: utf-8 -*-
"""scripts/record_demo.py —— 自动录制产品演示视频（默认输出 demo_tour.mp4）

用法::

    python scripts/record_demo.py                  # 输出到项目根目录 demo_tour.mp4
    python scripts/record_demo.py --keep-webm      # 额外保留原始 webm
    python scripts/record_demo.py --out demo.mp4   # 指定输出文件名

工作原理
--------
1. 以子进程拉起 `streamlit run app.py`（随机空闲端口，headless）
2. Playwright 打开 Chromium 并开启**真实视频录制**（不是截图拼贴）
3. 按剧本巡视四个 Tab，并在页面内注入底部字幕条——字幕直接录进画面，无需后期合成
4. 用 ffmpeg 转码为 H.264 / yuv420p / faststart 的 MP4，微信、钉钉、浏览器均可直接播放

为什么用浏览器内字幕而不是后期烧字
----------------------------------
ffmpeg drawtext 在中文字体路径与转义上极易踩坑；而在页面里注入一个 fixed 定位的
div 更新文案，字幕天然与画面同帧、零转码损耗，脚本也更短。
"""
from __future__ import annotations

import argparse
import shutil
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")   # type: ignore[attr-defined]
except Exception:  # pragma: no cover
    pass

VIEWPORT = {"width": 1600, "height": 900}

# ---------------------------------------------------------------------------
# 剧本：(字幕文案, 停留毫秒, 附加动作)
# 附加动作：("tab", i) 切换 Tab ｜ ("scroll", px) 滚动 ｜ None 静止
# ---------------------------------------------------------------------------

SCRIPT = [
    ("OnPrem-RAG-Kit｜私有化 RAG 落地套件 · 检索优化与算力 TCO 评估", 2800, ("title", True)),
    ("Tab 1｜同一个刁钻问题：左侧 Naive RAG（固定窗口暴力切分） vs 右侧本方案 Hybrid RAG", 9000, None),
    ("表格行被完整回溯、编码精确命中，回答带引用角标与置信度评分", 7000, ("scroll", 620)),
    ("Tab 2｜121 题规模化基准：表格解析准确率 54.4% → 100%，答案准确性 61.1% → 93.4%", 10000, ("tab", 1)),
    ("按题目来源拆分：简单查表题会淹没真实差距，所以拆开报", 6000, ("scroll", 700)),
    ("Tab 3｜算力选型与 TCO 测算器：输入规模与密级，实时输出硬件配置与三年现金流", 8000, ("tab", 2)),
    ("36 个月累计成本曲线，自动标出「回本临界点」；一键导出《致管理层》ROI 评估报告", 9000, ("scroll", 760)),
    ("Tab 4｜vLLM vs Ollama 并发压测：并发 50 时吞吐 10.2 倍，首字延迟相差 28.5 倍", 9000, ("tab", 3)),
    ("零配置可用 · 全部指标可复现 · python scripts/smoke_test.py", 4000, ("scroll", -400)),
]

CAPTION_JS = """
(text) => {
  let el = document.getElementById('__caption');
  if (!el) {
    el = document.createElement('div');
    el.id = '__caption';
    document.body.appendChild(el);
  }
  el.style.cssText = [
    'position:fixed','left:0','right:0','bottom:0','z-index:2147483647',
    'padding:16px 28px','text-align:center','pointer-events:none',
    'font:600 17px/1.55 "Microsoft YaHei","PingFang SC","Segoe UI",sans-serif',
    'color:#0E1B2E','letter-spacing:.02em',
    'background:linear-gradient(90deg,rgba(247,249,252,.97),rgba(226,237,250,.97),rgba(247,249,252,.97))',
    'border-top:1px solid rgba(42,120,214,.45)',
    'box-shadow:0 -8px 26px rgba(16,42,80,.12)'
  ].join(';');
  el.textContent = text;
  return true;
}
"""

TITLE_JS = """
(text) => {
  let el = document.getElementById('__titlecard');
  if (!el) {
    el = document.createElement('div');
    el.id = '__titlecard';
    document.body.appendChild(el);
  }
  el.style.cssText = [
    'position:fixed','inset:0','z-index:2147483646','display:flex',
    'align-items:center','justify-content:center','text-align:center',
    'padding:0 8%','pointer-events:none','background:rgba(250,252,255,.95)',
    'backdrop-filter:blur(3px)','transition:opacity .5s ease',
    'font:800 34px/1.5 "Microsoft YaHei","PingFang SC","Segoe UI",sans-serif',
    'background-image:radial-gradient(900px 420px at 20% 0%,rgba(42,120,214,.16),rgba(250,252,255,.96))',
    'color:#0E1B2E'
  ].join(';');
  el.textContent = text;
  return true;
}
"""


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def wait_health(port: int, timeout: float = 90.0) -> bool:
    url = f"http://127.0.0.1:{port}/_stcore/health"
    end = time.time() + timeout
    while time.time() < end:
        try:
            with urllib.request.urlopen(url, timeout=3) as r:
                if r.status == 200:
                    return True
        except Exception:
            time.sleep(0.8)
    return False


def convert_to_mp4(src: Path, dst: Path) -> bool:
    """webm -> mp4（H.264 / yuv420p / faststart），失败则原样保留 webm。"""
    exe = shutil.which("ffmpeg")
    if not exe:
        try:
            import imageio_ffmpeg  # type: ignore

            exe = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            exe = None
    if not exe:
        print("[warn] 未找到 ffmpeg，跳过转码，保留 webm。")
        return False
    cmd = [exe, "-y", "-i", str(src), "-c:v", "libx264", "-preset", "slow",
           "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", str(dst)]
    # 必须钉死 encoding：Windows 上 text=True 默认按区域编码（GBK）解码子进程管道，
    # 而 ffmpeg 输出里含中文路径（UTF-8 字节）会让读取线程抛 UnicodeDecodeError
    proc = subprocess.run(cmd, capture_output=True, text=True,
                          encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        print("[warn] ffmpeg 转码失败：", proc.stderr[-600:])
        return False
    return True


def _launch_chromium(p):
    """启动无头 Chromium，兼容「只装了完整内核、没装 headless_shell」的环境。"""
    opts = dict(args=["--hide-scrollbars", "--force-device-scale-factor=1"])
    try:
        return p.chromium.launch(**opts)
    except Exception as e:                                    # pragma: no cover
        if "Executable doesn't exist" not in str(e):
            raise
        print("   [note] 未找到 headless_shell，改用完整 Chromium 内核")
        return p.chromium.launch(channel="chromium", **opts)


def record(out_name: str, keep_webm: bool) -> int:
    try:
        from playwright.sync_api import sync_playwright
    except Exception as exc:
        print(f"[error] 需要 playwright：pip install playwright && playwright install chromium\n  {exc}")
        return 2

    port = free_port()
    work = ROOT / ".video_tmp"
    work.mkdir(exist_ok=True)
    out_path = ROOT / out_name

    print("=" * 78)
    print("OnPrem-RAG-Kit :: 演示视频录制")
    print("=" * 78)
    print(f"[1/4] 启动 Streamlit（端口 {port}）…")
    server = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", str(ROOT / "app.py"),
         "--server.headless=true", f"--server.port={port}",
         "--server.fileWatcherType=none", "--browser.gatherUsageStats=false",
         "--server.address=127.0.0.1"],
        cwd=str(ROOT), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        if not wait_health(port):
            print("[error] Streamlit 未能启动（健康检查超时）")
            return 3
        print("[2/4] 服务就绪，开始录制…")

        with sync_playwright() as p:
            browser = _launch_chromium(p)
            ctx = browser.new_context(viewport=VIEWPORT, record_video_dir=str(work),
                                      record_video_size=VIEWPORT, locale="zh-CN")
            page = ctx.new_page()
            page.set_default_timeout(60000)
            page.goto(f"http://127.0.0.1:{port}", wait_until="load")
            page.wait_for_selector("text=OnPrem-RAG-Kit", timeout=90000)
            page.wait_for_timeout(3500)          # 等图表与卡片完成首屏渲染

            for i, (caption, hold_ms, action) in enumerate(SCRIPT):
                if action and action[0] == "title":
                    page.evaluate(TITLE_JS, caption)
                    page.evaluate(CAPTION_JS, "OnPrem-RAG-Kit")
                    page.wait_for_timeout(int(hold_ms))
                    page.evaluate("document.getElementById('__titlecard').style.display='none'")
                    continue
                if action and action[0] == "tab":
                    try:
                        # 1.5x 是 button[data-baseweb="tab"]，1.6x 起改为 div[role="tab"]
                        page.locator('[role="tab"]').nth(int(action[1])).click()
                    except Exception as exc:      # 切换失败不中断录制
                        print(f"   [warn] 切换 Tab {action[1]} 失败：{exc}")
                    page.wait_for_timeout(2600)
                elif action and action[0] == "scroll":
                    page.mouse.wheel(0, int(action[1]))
                # 字幕必须等画面就位后再出现：若先写字幕再切 Tab，会有约 2.6 秒
                # 「字幕讲的是下一个 Tab、画面还停在上一个」的错位，观感很伤
                page.evaluate(CAPTION_JS, caption)
                if i == len(SCRIPT) - 1:
                    page.evaluate("const t=document.getElementById('__caption'); if(t) t.style.opacity='0';")
                page.wait_for_timeout(int(hold_ms))

            video = page.video
            ctx.close()
            path = Path(video.path()) if video else None
            browser.close()
        print("[3/4] 录制完成，转码中…")
        if not path or not path.exists():
            print("[error] 未生成视频文件")
            return 4
        ok = convert_to_mp4(path, out_path)
        if not ok:
            out_path = ROOT / (Path(out_name).stem + ".webm")
            shutil.copy2(path, out_path)
        if keep_webm:
            shutil.copy2(path, work.parent / (Path(out_name).stem + ".webm"))
        size_mb = out_path.stat().st_size / 1e6
        print(f"[4/4] 已输出：{out_path}（{size_mb:.1f} MB）")
        print("=" * 78)
        return 0
    finally:
        server.terminate()
        try:
            server.wait(timeout=15)
        except Exception:
            server.kill()
        if not keep_webm:
            shutil.rmtree(work, ignore_errors=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="录制 OnPrem-RAG-Kit 演示视频")
    ap.add_argument("--out", default="demo_tour.mp4", help="输出文件名（默认 demo_tour.mp4，写入项目根目录）")
    ap.add_argument("--keep-webm", action="store_true", help="保留原始 webm 与临时目录")
    args = ap.parse_args()
    sys.exit(record(args.out, args.keep_webm))


if __name__ == "__main__":
    main()
