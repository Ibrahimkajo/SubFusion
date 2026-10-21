# Examples

`demo_input/synthetic_street.png` is a licence-free synthetic smoke-test image.
Its left half is visible RGB and its right half is a co-registered thermal-style
image.  It does not originate from MSRS, LLVIP, M3FD, or another dataset.

After installing the dependencies, run the released checkpoint directly:

```bash
python scripts/infer.py --input-dir examples/demo_input --output-dir outputs/demo
```

The fused result is written to `outputs/demo/synthetic_street.png`.  To
regenerate the identical input image, use:

```bash
python scripts/make_demo_pair.py
```
