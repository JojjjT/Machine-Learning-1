# -*- coding: utf-8 -*-
"""Verifica que todas las referencias de index.html existan en disco (carpetas nuevas)."""
import sys, re, os
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
P = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # raiz del proyecto
h = open(os.path.join(P, "index.html"), encoding="utf-8").read()
refs = re.findall(r'(?:src|href)="([^"]+\.(?:png|pdf|html|json))"', h)
fallos = 0
for r in refs:
    if r.startswith("http"):
        continue
    ok = os.path.exists(os.path.join(P, r))
    print(f"  {'OK ' if ok else 'FALTA'} {r}")
    fallos += 0 if ok else 1
# tambien streamlit.html
h2 = open(os.path.join(P, "streamlit.html"), encoding="utf-8").read()
refs2 = [m for m in re.findall(r'url:\s*"\./([^"]+)"', h2)]
for r in refs2:
    ok = os.path.exists(os.path.join(P, r))
    print(f"  {'OK ' if ok else 'FALTA'} [streamlit] {r}")
    fallos += 0 if ok else 1
print(f"\nReferencias rotas: {fallos}")
