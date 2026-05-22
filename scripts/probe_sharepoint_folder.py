from __future__ import annotations

import re

import requests


URL = (
    "https://hkustconnect-my.sharepoint.com/personal/rzuo_connect_ust_hk/Documents/Forms/All.aspx"
    "?RootFolder=%2fpersonal%2frzuo_connect_ust_hk%2fDocuments%2fEMNLP24_Online%2fPhoenix-2014T%2fckpts"
    "&FolderCTID=0x0120007A850B972CF04E4382A2B21E4F0C8908&ga=1"
)


def main() -> None:
    text = requests.get(URL, timeout=60).text
    patterns = [
        r"download\.aspx[^\"'<> ]+",
        r"UniqueId[^\"'<> ]+",
        r"[A-Za-z0-9_\-]+\.ckpt",
        r"[A-Za-z0-9_\-]+\.vocab",
        r"[A-Za-z0-9_\-]+\.yaml",
        r"[A-Za-z0-9_\-]+\.json",
    ]
    for p in patterns:
        matches = sorted(set(re.findall(p, text, flags=re.IGNORECASE)))
        print(p, len(matches))
        for m in matches[:20]:
            print(" ", m)


if __name__ == "__main__":
    main()
