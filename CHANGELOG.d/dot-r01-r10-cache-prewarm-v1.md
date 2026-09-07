# Resumable DOT R01–R10 cache prewarm

Add a request-triggered, checksum-bound workflow that materializes the official compressed `R01-10.zip` archive at the exact cache path used by the frozen R04–R10 CUT3R confirmation on `gpuserver6000`.

The workflow supports resumable transfer, verifies the publisher MD5 and byte count, and never enumerates or extracts archive members. It opens no images or markers, constructs no predictions, and performs no scientific evaluation. Its sole purpose is to let a technical rerun reuse an exact verified archive without changing any frozen scientific identity or decision rule.
