# -*- coding: utf-8 -*-
"""scripts/ui_shot.py —— 无损界面截图与样式核验（给改动做验收用）

用法::

    python scripts/ui_shot.py                 # 四个 Tab 各存一张 PNG 到 .shots/
    python scripts/ui_shot.py --tab 2         # 只存第 3 个 Tab（从 0 数）
    python scripts/ui_shot.py --assets        # 顺便覆盖 README 用的 assets/ui-tab*.png

为什么需要它
------------
演示视频经过 H.264 压缩，小字与发丝线会被明显损伤，无法用来判断
「样式是否真的生效、排版是否成立」。本脚本直接由浏览器输出无损 PNG，
并回读关键元素的计算样式，把"渲染正确"和"压缩失真"两件事彻底分开。
"""
from __future__ import annotations

import argparse
import json
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VIEWPORT = {"width": 1600, "height": 900}
TAB_LABELS = ["Tab1 架构对比", "Tab2 基准评测", "Tab3 TCO测算", "Tab4 压测"]

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")   # type: ignore[attr-defined]
except Exception:  # pragma: no cover
    pass

PROBE_JS = """
() => {
  const cs = getComputedStyle(document.documentElement);
  const out = {vars:{}, probe:{}};
  ['--paper','--hull','--deck','--ink','--ink-2','--ink-3',
   '--sig','--sig-text','--warn','--warn-text','--ok-text','--amber-text'].forEach(k => {
    out.vars[k] = cs.getPropertyValue(k).trim();
  });
  const pick = (sel, props) => {
    const el = document.querySelector(sel);
    if (!el) return null;
    const s = getComputedStyle(el);
    const o = {};
    props.forEach(p => o[p] = s.getPropertyValue(p));
    return o;
  };
  out.probe['rail-id'] = pick('.rail-id', ['font-size','color','font-weight']);
  out.probe['kpi-value'] = pick('.kpi-value', ['font-size','color']);
  out.probe['tab-selected'] = pick('[role="tab"][aria-selected="true"]',
                                   ['color','border-bottom-color','background-color']);
  out.probe['verdict-to'] = pick('.verdict-to', ['font-size','color']);
  out.probe['answer-box-left-rule'] = pick(
      'div[data-testid="stVerticalBlockBorderWrapper"]', ['border-left-color','border-left-width']);
  out.styleTagCount = document.querySelectorAll('style').length;
  out.textHead = (document.body.innerText || '').replace(/\\s+/g, ' ').slice(0, 240);
  return out;
}
"""


def launch_chromium(p):
    """启动无头 Chromium，兼容「只装了完整内核、没装 headless_shell」的环境。

    Playwright 新版默认 headless=True 会去找 chromium_headless_shell-<rev>，
    某些机器上只缓存了 chromium-<rev>（完整内核）。此时显式走
    channel="chromium" 使用完整内核的新无头模式，效果一致。
    """
    opts = dict(args=["--hide-scrollbars", "--force-device-scale-factor=1"])
    try:
        return p.chromium.launch(**opts)
    except Exception as e:                                    # pragma: no cover
        if "Executable doesn't exist" not in str(e):
            raise
        print("        [note] 未找到 headless_shell，改用完整 Chromium 内核")
        return p.chromium.launch(channel="chromium", **opts)


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


def main() -> None:
    ap = argparse.ArgumentParser(description="无损界面截图与样式核验")
    ap.add_argument("--tab", type=int, default=-1, help="只截某个 Tab（0-3），默认全部")
    ap.add_argument("--full", action="store_true", help="同时保存整页长图")
    ap.add_argument("--height", type=int, default=VIEWPORT["height"],
                    help=f"视口高度，默认 {VIEWPORT['height']}（Tab1 的答案框会被截断，可调高）")
    ap.add_argument("--assets", action="store_true",
                    help="把 Tab1/Tab2 同步覆盖到 assets/ui-tab*.png（README 展示用）")
    args = ap.parse_args()

    from playwright.sync_api import sync_playwright

    port = free_port()
    out_dir = ROOT / ".shots"
    out_dir.mkdir(exist_ok=True)

    print(f"[1/3] 启动 Streamlit（端口 {port}）…")
    server = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", str(ROOT / "app.py"),
         "--server.headless=true", f"--server.port={port}",
         "--server.fileWatcherType=none", "--browser.gatherUsageStats=false",
         "--server.address=127.0.0.1"],
        cwd=str(ROOT), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        if not wait_health(port):
            print("[error] Streamlit 未启动")
            sys.exit(3)
        print("[2/3] 渲染并就绪，开始截图…")
        with sync_playwright() as p:
            browser = launch_chromium(p)
            page = browser.new_context(viewport={"width": VIEWPORT["width"], "height": args.height},
                                       device_scale_factor=1,
                                       locale="zh-CN").new_page()
            page.set_default_timeout(60000)
            page.goto(f"http://127.0.0.1:{port}", wait_until="load")
            page.wait_for_selector("text=OnPrem-RAG-Kit", timeout=90000)
            page.wait_for_timeout(4000)

            targets = range(4) if args.tab < 0 else [args.tab]
            # Streamlit 1.5x 是 button[data-baseweb="tab"]，1.6x 起改为 div[role="tab"]。
            # 用 [role="tab"] 覆盖两代结构，避免换版本就截图失败。
            tabs = page.locator('[role="tab"]')
            if tabs.count() < 4:
                raise SystemExit(f"[error] 只找到 {tabs.count()} 个 Tab，页面结构异常")
            for i in targets:
                tabs.nth(i).click()
                page.wait_for_timeout(3200)
                f = out_dir / f"tab{i + 1}.png"
                page.screenshot(path=str(f))                      # 无损 PNG，视口内
                if args.full:
                    page.screenshot(path=str(out_dir / f"tab{i + 1}_full.png"), full_page=True)
                print(f"       已保存 {f.name}  ({f.stat().st_size / 1024:.0f} KB)  {TAB_LABELS[i]}")
                if args.assets and i in (0, 1):
                    dest = ROOT / "assets" / f"ui-tab{i + 1}.png"
                    dest.parent.mkdir(exist_ok=True)
                    dest.write_bytes(f.read_bytes())
                    print(f"       已同步 assets/{dest.name}")

            probe = page.evaluate(PROBE_JS)
            (out_dir / "style_probe.json").write_text(
                json.dumps(probe, ensure_ascii=False, indent=2), encoding="utf-8")
            print("[3/3] 样式核验结果：")
            print(json.dumps(probe, ensure_ascii=False, indent=2))
            browser.close()
    finally:
        server.terminate()
        try:
            server.wait(timeout=15)
        except Exception:
            server.kill()


if __name__ == "__main__":
    main()
