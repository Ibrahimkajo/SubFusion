# Third-party notices

The image-fusion metric implementations in `subfusion/metrics.py` document their public reference implementations in that file.  In particular, the implementations cite the MMIF-CDDFuse evaluator, the Yang Qy metric reference, and the `pytorch-msssim` package.

Datasets are not included, except for one LLVIP paired smoke-test image at
`examples/demo_input/llvip_010001.jpg`.  That file remains subject to the
[official LLVIP terms](https://github.com/bupt-ai-cz/LLVIP) and is supplied
solely to make the released inference path directly testable.  MSRS, LLVIP,
and M3FD must otherwise be obtained from their official distribution locations
under their respective terms.
