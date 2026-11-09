"""Mail the hourly protector-research report.

Body: the newest RESULTS.md section(s) since the last mail, branch commits since then, and
which research jobs are running. Figures under runs/figs/ newer than the last mail are attached.

SMTP credentials come from ~/.config/sample_plan_mail/env (MAIL_PROVIDER, MAIL_USER, MAIL_PASS,
MAIL_TO), the same file the sampling scripts use; nothing secret lives in this repo.

Usage: python tools/send_report.py [--dry-run] [--force]
Without --force, it skips mailing when nothing changed since the last mail.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import smtplib
import ssl
import subprocess
from email.header import Header
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.utils import formataddr, formatdate, make_msgid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
RESULTS = ROOT / "RESULTS.md"
FIGS = ROOT / "runs" / "figs"
STATE = ROOT / "runs" / ".last_report.json"
MAIL_ENV = Path.home() / ".config/sample_plan_mail/env"
MAX_ATTACH = 6


def sh(*cmd: str) -> str:
  return subprocess.run(cmd, cwd=REPO, capture_output=True, text=True).stdout.strip()


def load_state() -> dict:
  if STATE.exists():
    return json.loads(STATE.read_text())
  return {"time": 0.0, "results_len": 0, "head": ""}


def build(state: dict) -> tuple[str, str, list[Path], bool]:
  now = dt.datetime.now()
  text = RESULTS.read_text() if RESULTS.exists() else ""
  new_results = text[state["results_len"]:].strip() if len(text) >= state["results_len"] else text
  head = sh("git", "rev-parse", "HEAD")
  commits = (sh("git", "log", "--oneline", f"{state['head']}..HEAD", "--", "protector_research")
             if state["head"] else sh("git", "log", "--oneline", "-10", "--", "protector_research"))
  jobs = sh("bash", "-c", "pgrep -fa 'protector_research|fit_m2|fit_m3|drop_m3|run_robot' "
                          "| grep -v -e pgrep -e send_report || true")
  figs = sorted((p for p in FIGS.glob("*.png") if p.stat().st_mtime > state["time"]),
                key=lambda p: p.stat().st_mtime)[-MAX_ATTACH:] if FIGS.exists() else []
  changed = bool(new_results or commits or figs)

  lines = [f"护具仿真研究 小时报告  {now:%Y-%m-%d %H:%M}", ""]
  lines += ["## 本小时新增结果", new_results or "（RESULTS.md 无新增）", ""]
  lines += ["## 新提交（wczmiracle/mujoco 分支 protector-research）", commits or "（无）", ""]
  lines += ["## 正在运行的任务", jobs or "（无）", ""]
  if figs:
    lines += ["## 附图", *[f"- {p.name}" for p in figs], ""]
  lines += ["完整记录：/home/wang22/engineai/mujoco/protector_research/RESULTS.md"]
  subject = f"[护具研究] {now:%m-%d %H:%M} " + ("有新结果" if changed else "无新进展")
  return subject, "\n".join(lines), figs, changed


def send(subject: str, body: str, figs: list[Path]) -> None:
  env = {}
  for line in MAIL_ENV.read_text().splitlines():
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
      k, v = line.split("=", 1)
      env[k.strip()] = v.strip()
  host, port = {"qq": ("smtp.qq.com", 465), "126": ("smtp.126.com", 465)}[env["MAIL_PROVIDER"]]
  user = env["MAIL_USER"]
  to = [x.strip() for x in env["MAIL_TO"].split(",") if x.strip()]
  msg = MIMEMultipart()
  msg["Subject"] = Header(subject, "utf-8")
  msg["From"] = formataddr(("protector_research", user))
  msg["To"] = ", ".join(to)
  msg["Date"] = formatdate(localtime=True)
  msg["Message-ID"] = make_msgid(domain=user.split("@", 1)[-1])
  msg.attach(MIMEText(body, "plain", "utf-8"))
  for p in figs:
    msg.attach(MIMEApplication(p.read_bytes(), Name=p.name))
  with smtplib.SMTP_SSL(host, port, context=ssl.create_default_context(), timeout=60) as s:
    s.login(user, env["MAIL_PASS"])
    s.sendmail(user, to, msg.as_string())


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--dry-run", action="store_true")
  ap.add_argument("--force", action="store_true", help="mail even if nothing changed")
  args = ap.parse_args()
  state = load_state()
  subject, body, figs, changed = build(state)
  if args.dry_run:
    print(subject, "\n", body, "\nattach:", [p.name for p in figs])
    return
  if not changed and not args.force:
    print("no change since last report; not mailing")
    return
  send(subject, body, figs)
  STATE.parent.mkdir(parents=True, exist_ok=True)
  STATE.write_text(json.dumps({
      "time": dt.datetime.now().timestamp(),
      "results_len": len(RESULTS.read_text()) if RESULTS.exists() else 0,
      "head": sh("git", "rev-parse", "HEAD"),
  }))
  print("mailed:", subject)


if __name__ == "__main__":
  main()
