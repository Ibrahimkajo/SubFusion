# LLVIP smoke-test example

`demo_input/llvip_010001.jpg` is a single horizontally paired LLVIP example:
the visible RGB image is on the left and the infrared image is on the right.
It is included only to let reviewers verify inference immediately after cloning
the repository.

The image remains subject to the official LLVIP non-commercial terms of use.
Do not redistribute it separately.  See the official [LLVIP repository](https://github.com/bupt-ai-cz/LLVIP)
and cite Jia *et al.*, *LLVIP: A Visible-infrared Paired Dataset for Low-light
Vision*, ICCVW 2021, when using it.

After installing the dependencies and downloading the released checkpoint, run:

```bash
python scripts/infer.py --input-dir examples/demo_input --output-dir outputs/demo
```

The fused result is written to `outputs/demo/llvip_010001.png`.
