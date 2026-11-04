#!/usr/bin/env python3
"""Download only the official GoPro full ZIP member by HTTP byte ranges.

The URL comes directly from the pinned official Concertormer README. No
third-party mirror is accepted. ZIP CRC and the extracted SHA256 are recorded.
Use --validate to perform strict model loading after acquisition.
"""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import time
import urllib.request
import zipfile

OFFICIAL_URL = "https://drive.usercontent.google.com/download?id=1NvbGroZm4vVgvWJtmgW-6Fwd97PiukiL&export=download&confirm=t"
REPOSITORY = "https://github.com/setsunil/Concertormer"


class HTTPRangeFile(io.RawIOBase):
    def __init__(self, url, timeout=90):
        self.url, self.timeout, self.position = url, timeout, 0
        with urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=timeout) as r:
            self.length = int(r.headers["Content-Length"])
            self.metadata = dict(r.headers)
        self.downloaded = 0
        self.requests = 0
        self.cache_start = -1
        self.cache = b""

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=os.SEEK_SET):
        if whence == os.SEEK_SET:
            self.position = offset
        elif whence == os.SEEK_CUR:
            self.position += offset
        elif whence == os.SEEK_END:
            self.position = self.length + offset
        if self.position < 0:
            raise ValueError("Negative range position")
        return self.position

    def read(self, size=-1):
        if size < 0:
            size = self.length - self.position
        size = min(size, self.length - self.position)
        if size <= 0:
            return b""
        start = self.position
        end = start + size
        if not (self.cache_start <= start and end <= self.cache_start + len(self.cache)):
            fetch_end = min(self.length, max(end, start + (8 << 20)))
            request = urllib.request.Request(self.url, headers={"Range": f"bytes={start}-{fetch_end-1}"})
            for attempt in range(4):
                try:
                    with urllib.request.urlopen(request, timeout=self.timeout) as r:
                        if r.status != 206 or r.headers.get("Content-Range") != f"bytes {start}-{fetch_end-1}/{self.length}":
                            raise RuntimeError("Server did not honor the exact byte range; refusing full archive fallback")
                        data = r.read()
                    if len(data) != fetch_end-start:
                        raise RuntimeError("Incomplete byte-range response")
                    break
                except Exception:
                    if attempt == 3:
                        raise
                    time.sleep(2**attempt)
            self.cache_start, self.cache = start, data
            self.downloaded += len(data)
            self.requests += 1
        data = self.cache[start-self.cache_start:end-self.cache_start]
        self.position = end
        return data


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dest", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)
    p.add_argument("--validate", action="store_true")
    p.add_argument("--list-only", action="store_true")
    args = p.parse_args()
    started = time.time()
    remote = HTTPRangeFile(OFFICIAL_URL)
    with zipfile.ZipFile(remote) as archive:
        members = archive.infolist()
        names = [m.filename for m in members]
        candidates = [m for m in members if m.filename.endswith("GoPro/Concertormer.pth")]
        if len(candidates) != 1:
            raise RuntimeError(f"Expected exactly one official GoPro full member, got {names}")
        selected = candidates[0]
        report = dict(repository=REPOSITORY, source_url=OFFICIAL_URL,
                      archive_bytes=remote.length, archive_members=names,
                      selected_member=selected.filename, member_bytes=selected.file_size,
                      member_compressed_bytes=selected.compress_size, member_crc32=f"{selected.CRC:08x}",
                      full_archive_downloaded=False)
        if not args.list_only:
            args.dest.parent.mkdir(parents=True, exist_ok=True)
            temporary = args.dest.with_suffix(args.dest.suffix + ".partial")
            with archive.open(selected) as source, temporary.open("wb") as target:
                copied = 0
                while block := source.read(8 << 20):
                    target.write(block)
                    copied += len(block)
                    print(json.dumps({"downloaded_member_bytes": copied, "member_bytes": selected.file_size}), flush=True)
            if temporary.stat().st_size != selected.file_size:
                raise RuntimeError("Extracted member size mismatch")
            temporary.replace(args.dest)
            report.update(extracted_sha256=sha256(args.dest), zip_crc_verified=True,
                          extracted_path=str(args.dest.resolve()))
    report.update(transferred_bytes=remote.downloaded, range_requests=remote.requests,
                  elapsed_seconds=time.time()-started)
    if args.validate and not args.list_only:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from models import build_model
        model = build_model(official_checkpoint=args.dest)
        report["strict_load"] = model.pretrained_report
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
