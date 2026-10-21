# Publishing this reviewer release on GitHub

1. Create a new, empty GitHub repository.  For double-blind review, use an anonymous organization/account and do not add identifying metadata.
2. Commit the code, configuration, documentation, and compact CSV/JSON summaries in this folder.
3. Do **not** commit `weights/subfusion_msrs_full_rank1_70e.pth` as an ordinary Git blob; GitHub's ordinary-file limit is 100 MB.  Upload it as a GitHub Release asset named exactly `subfusion_msrs_full_rank1_70e.pth`, or use Git LFS.
4. Verify the release-asset SHA-256 against `weights/README.md`.
5. Before changing repository visibility after review, select an author-approved licence and replace `LICENSE_STATUS.md`.

The local bundle includes the checkpoint so the reviewer archive can be tested before publishing.  `.gitignore` excludes the binary solely to prevent an accidental rejected push.
