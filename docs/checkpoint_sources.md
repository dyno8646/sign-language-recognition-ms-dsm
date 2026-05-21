# Pretrained Checkpoint Sources

## SLRT Online (Primary)

From upstream `third_party_SLRT/Online/CSLR/README.md`:

- Phoenix-2014T ISLR checkpoint folder:
  - [OneDrive](https://hkustconnect-my.sharepoint.com/:f:/g/personal/rzuo_connect_ust_hk/EidJXFxpyaNPho5SKtVHEJ8BHex8Gq62koL-RrNnqtF1PA?e=IGGpxU)
- CSL-Daily ISLR checkpoint folder:
  - [OneDrive](https://hkustconnect-my.sharepoint.com/:f:/g/personal/rzuo_connect_ust_hk/EhS5B3p9i3FNu5OpqFy3WyABkMMGg1VbAzMJrxjuFVOg6Q?e=c7OK0Z)

## NLA-SLR (Reference Only)

From upstream `third_party_SLRT/NLA-SLR/README.md`:

- WLASL/MSASL/NMF checkpoints are provided via OneDrive/Baidu links in the performance table.
- These are isolated-sign recognition checkpoints and are kept as reference for later comparisons, not as the primary online webcam path.

## Expected Local Placement

Put downloaded files in:

```text
checkpoints/slrt/
```

Required by runtime:

- one checkpoint (`*.ckpt` or `*.pth` or `*.pt`)
- one vocabulary file (`*.vocab` or `*.json`)
- optional config (`*.yaml`)

The runtime auto-discovers these files at startup and reports status through `/health`.
