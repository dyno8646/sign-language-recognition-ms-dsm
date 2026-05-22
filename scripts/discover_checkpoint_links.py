from __future__ import annotations

import re

import requests


URL = "https://hkustconnect-my.sharepoint.com/:f:/g/personal/rzuo_connect_ust_hk/EidJXFxpyaNPho5SKtVHEJ8BHex8Gq62koL-RrNnqtF1PA?e=IGGpxU"


def main() -> None:
    text = requests.get(URL, timeout=30).text
    links = sorted(set(re.findall(r"https://[^\"'\\s<>]+", text)))
    interesting = [
        x
        for x in links
        if any(k in x.lower() for k in ["download", "ckpt", "vocab", "yaml", ".pt", ".pth", ".zip"])
    ]
    print("all_links", len(links))
    print("interesting", len(interesting))
    for link in interesting[:200]:
        print(link)


if __name__ == "__main__":
    main()
