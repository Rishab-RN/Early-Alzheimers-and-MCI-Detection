#!/usr/bin/env python3
"""Download one OASIS-3 T1w scan per selected NITRC-IR MRI session.

Run with --select-only to create selected_sessions.csv without logging in.
Credentials are requested interactively and are never saved to disk.
"""

from __future__ import annotations

import argparse
import csv
import getpass
import gzip
import re
import shutil
import sys
import tempfile
import time
import zipfile
from pathlib import Path, PurePosixPath
from urllib.parse import quote

import pandas as pd
import requests
from tqdm import tqdm


BASE_URL = "https://nitrc.org/ir"
PROJECT_ID = "OASIS3"
DEFAULT_CSV = "akhileshrvce_11_8_2026_13_30_10.csv"
TIMEOUT = (20, 120)  # connect, read (seconds)
MAX_ATTEMPTS = 3
RETRYABLE_STATUSES = {408, 429, 500, 502, 503, 504}
LOG_COLUMNS = ["MR ID", "scan ID", "status", "file path", "error message"]


class DownloadError(Exception):
    """A download or API error safe to report in a log."""


def response_preview(response: requests.Response, max_chars: int = 1000) -> str:
    """Show HTTP status and a bounded response body for API troubleshooting."""
    try:
        # Avoid reading a potentially enormous error response into memory.
        body = next(response.iter_content(chunk_size=max_chars), b"")
        body_text = body.decode(response.encoding or "utf-8", errors="replace")
    except requests.RequestException:
        body_text = "<response body unavailable>"
    return f"HTTP {response.status_code} {response.url}: {body_text[:max_chars]!r}"


def retry_delay(attempt: int) -> None:
    time.sleep(2 ** (attempt - 1))


class NITRCClient:
    def __init__(self, username: str, password: str):
        self.username = username
        self.password = password
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "OASIS3-T1w-downloader/1.0"})

    def close(self) -> None:
        self.session.close()

    def authenticate(self) -> None:
        """Create or refresh the session using the XNAT JSESSION endpoint."""
        url = f"{BASE_URL}/data/JSESSION"
        # Clear expired cookies before requesting a new token. Some XNAT
        # versions return the new ID only in the response body.
        self.session.cookies.clear()
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                with self.session.post(
                    url,
                    auth=(self.username, self.password),
                    timeout=TIMEOUT,
                    allow_redirects=False,
                ) as response:
                    if response.status_code == 200:
                        # Some XNAT installations send Set-Cookie; others return
                        # the JSESSIONID as plain text. Support both variants.
                        if not self.session.cookies.get("JSESSIONID"):
                            token = response.text.strip()
                            if not re.fullmatch(r"[A-Za-z0-9._=-]{8,256}", token):
                                raise DownloadError(
                                    "The JSESSION endpoint returned no usable session cookie "
                                    "or token (possible login-page redirect)."
                                )
                            self.session.cookies.set(
                                "JSESSIONID", token, domain=".nitrc.org", path="/"
                            )
                        return
                    # Never display the authentication response body: it might
                    # contain a token or other confidential information.
                    if response.status_code not in RETRYABLE_STATUSES:
                        raise DownloadError(
                            f"Authentication failed: HTTP {response.status_code} at {url}"
                        )
                    error = f"Authentication endpoint returned HTTP {response.status_code}"
            except requests.RequestException as exc:
                error = f"Authentication request failed: {type(exc).__name__}: {exc}"
            if attempt < MAX_ATTEMPTS:
                retry_delay(attempt)
        raise DownloadError(f"{error} (after {MAX_ATTEMPTS} attempts)")

    def get(self, path: str, *, params: dict[str, str], stream: bool = False):
        """GET with timeouts, retries, and automatic reauthentication on 401."""
        url = f"{BASE_URL}{path}"
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                response = self.session.get(
                    url, params=params, timeout=TIMEOUT, stream=stream
                )
                if response.status_code == 401:
                    response.close()
                    if attempt < MAX_ATTEMPTS:
                        self.authenticate()
                        continue
                    raise DownloadError(f"HTTP 401 {url}: unauthorized after reauthentication")
                if response.status_code >= 400:
                    detail = response_preview(response)
                    retry = response.status_code in RETRYABLE_STATUSES
                    response.close()
                    if retry and attempt < MAX_ATTEMPTS:
                        retry_delay(attempt)
                        continue
                    raise DownloadError(detail)
                return response  # caller must close the returned Response
            except requests.RequestException as exc:
                if attempt == MAX_ATTEMPTS:
                    raise DownloadError(
                        f"GET {url} failed after {MAX_ATTEMPTS} attempts: "
                        f"{type(exc).__name__}: {exc}"
                    ) from exc
                retry_delay(attempt)
        raise DownloadError(f"GET {url} failed after {MAX_ATTEMPTS} attempts")

    def list_scans(self, mr_id: str) -> list[dict]:
        # NITRC-IR only returns a flat ResultSet for the fully qualified
        # project/subject/experiment path. The shorter projects/.../experiments
        # path responds HTTP 200 with a *different* nested experiment document.
        subject = mr_id.split("_MR_", 1)[0]
        path = (
            f"/data/projects/{PROJECT_ID}/subjects/{quote(subject, safe='')}"
            f"/experiments/{quote(mr_id, safe='')}/scans"
        )
        with self.get(
            path, params={"format": "json", "columns": "ID,type,series_description"}
        ) as response:
            try:
                data = response.json()
                scans = data["ResultSet"]["Result"]
                if not isinstance(scans, list):
                    raise ValueError("ResultSet.Result is not a list")
                return scans
            except (ValueError, KeyError, TypeError) as exc:
                # An HTML login screen with HTTP 200 must not pass as JSON.
                raise DownloadError(
                    f"Unexpected scan-list response for {mr_id}: "
                    f"{response_preview(response)} ({exc})"
                ) from exc

    def download_zip(self, mr_id: str, scan_id: str, destination: Path) -> Path:
        subject = mr_id.split("_MR_", 1)[0]
        path = (
            f"/data/projects/{PROJECT_ID}/subjects/{quote(subject, safe='')}"
            f"/experiments/{quote(mr_id, safe='')}"
            f"/scans/{quote(scan_id, safe='')}/files"
        )
        destination.mkdir(parents=True, exist_ok=True)
        zip_path = destination / f"scan_{scan_id}.zip"
        part_path = destination / f"scan_{scan_id}.zip.part"
        # A stream error can happen after GET succeeds. Retry the entire transfer.
        for attempt in range(1, MAX_ATTEMPTS + 1):
            part_path.unlink(missing_ok=True)
            try:
                with self.get(path, params={"format": "zip"}, stream=True) as response:
                    content_type = response.headers.get("Content-Type", "").lower()
                    if "text/html" in content_type or "application/json" in content_type:
                        raise DownloadError(
                            "The ZIP endpoint returned an unexpected content type: "
                            + response_preview(response)
                        )
                    with part_path.open("wb") as output:
                        for chunk in response.iter_content(chunk_size=1024 * 1024):
                            if chunk:
                                output.write(chunk)
                if not zipfile.is_zipfile(part_path):
                    with part_path.open("rb") as bad_response:
                        preview = bad_response.read(500).decode("utf-8", errors="replace")
                    raise DownloadError(
                        f"GET {BASE_URL}{path}?format=zip returned a non-ZIP file. "
                        f"Response begins: {preview!r}"
                    )
                part_path.replace(zip_path)
                return zip_path
            except (requests.RequestException, OSError, zipfile.BadZipFile, DownloadError) as exc:
                part_path.unlink(missing_ok=True)
                if attempt == MAX_ATTEMPTS:
                    raise DownloadError(
                        f"ZIP download failed after {MAX_ATTEMPTS} attempts: {exc}"
                    ) from exc
                retry_delay(attempt)
        raise DownloadError("ZIP download failed")


def choose_t1w_scan(scans: list[dict]) -> str:
    t1w = [scan for scan in scans if str(scan.get("type", "")).strip() == "T1w"]
    if not t1w:
        raise DownloadError("No scan with type exactly 'T1w' found in XNAT")
    chosen = next(
        (
            scan
            for scan in t1w
            if "run-01" in str(scan.get("series_description") or "").lower()
        ),
        t1w[0],
    )
    scan_id = str(chosen.get("ID", "")).strip()
    if not scan_id:
        raise DownloadError("Selected T1w scan has no ID")
    return scan_id


def extract_zip_flat(zip_path: Path, destination: Path) -> list[Path]:
    """Safely extract a scan's files together into the session directory.

    XNAT ZIPs may contain directory prefixes. Flatten them so NIfTI and JSON
    sidecars end up together, and reject duplicate output names.
    """
    with tempfile.TemporaryDirectory(prefix="extract_", dir=destination) as temp:
        stage = Path(temp)
        seen = set()
        with zipfile.ZipFile(zip_path) as archive:
            for item in archive.infolist():
                if item.is_dir():
                    continue
                member = PurePosixPath(item.filename.replace("\\", "/"))
                if member.is_absolute() or ".." in member.parts:
                    raise DownloadError(f"Unsafe path in ZIP: {item.filename!r}")
                filename = member.name
                if not filename or filename in seen:
                    raise DownloadError(f"Empty or duplicate filename in ZIP: {filename!r}")
                seen.add(filename)
                with archive.open(item) as source, (stage / filename).open("wb") as out:
                    shutil.copyfileobj(source, out, length=1024 * 1024)
        nifti = list(stage.glob("*.nii.gz"))
        if not nifti:
            raise DownloadError("ZIP contained no .nii.gz file")
        for source in stage.iterdir():
            source.replace(destination / source.name)
        return [destination / item.name for item in nifti]


def select_sessions(csv_path: Path, all_sessions: bool) -> pd.DataFrame:
    columns = ["MR ID", "Date", "Subject", "Age", "Scanner", "Scans",
               "PUP Timecourses", "Freesurfers"]
    data = pd.read_csv(csv_path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    data.columns = data.columns.str.strip()
    missing = [column for column in columns if column not in data.columns]
    if missing:
        raise ValueError(f"Input CSV is missing columns: {', '.join(missing)}")
    for column in ("Subject", "MR ID", "Scans"):
        data[column] = data[column].str.strip()

    data = data[
        ~data["Subject"].str.startswith("0AS")
        & data["Scans"].str.contains("T1w", regex=False)
    ].copy()
    data["_day"] = pd.to_numeric(
        data["MR ID"].str.extract(r"_MR_d(\d+)$", expand=False), errors="coerce"
    )
    invalid = data[data["_day"].isna()]
    if not invalid.empty:
        examples = ", ".join(invalid["MR ID"].head(5))
        raise ValueError(f"Cannot parse day number from MR ID(s): {examples}")

    data = data.sort_values(["Subject", "_day", "MR ID"], kind="stable")
    if not all_sessions:
        data = data.drop_duplicates(subset=["Subject"], keep="first")
    return data[["Subject", "MR ID", "Age", "Scanner"]].reset_index(drop=True)


def append_log(log_path: Path, mr_id: str, scan_id: str, status: str,
               file_path: str = "", error_message: str = "") -> None:
    new_file = not log_path.exists() or log_path.stat().st_size == 0
    with log_path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        if new_file:
            writer.writerow(LOG_COLUMNS)
        writer.writerow([mr_id, scan_id, status, file_path, error_message])


def verify_niftis(session_folders: list[Path], report_path: Path) -> tuple[int, int]:
    """Open NIfTIs with nibabel and verify the entire gzip payload/CRC."""
    try:
        import nibabel as nib
    except ImportError as exc:
        raise DownloadError(
            "nibabel is required for verification. Install with "
            "'python -m pip install -r requirements.txt'."
        ) from exc

    seen = set()
    valid = corrupt = 0
    with report_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["file path", "status", "error message"])
        for folder in tqdm(session_folders, desc="Verifying NIfTI files", unit="session"):
            for path in folder.rglob("*.nii.gz"):
                if path in seen:
                    continue
                seen.add(path)
                try:
                    image = nib.load(str(path))
                    if not image.shape or any(dim <= 0 for dim in image.shape):
                        raise ValueError(f"Invalid image shape: {image.shape}")
                    # nib.load() is lazy; reading the entire gzip stream also
                    # catches truncated files and checksum errors in voxel data.
                    with gzip.open(path, "rb") as compressed:
                        while compressed.read(1024 * 1024):
                            pass
                    valid += 1
                    writer.writerow([str(path), "ok", ""])
                except Exception as exc:
                    corrupt += 1
                    message = f"{type(exc).__name__}: {exc}"
                    writer.writerow([str(path), "corrupt", message])
                    tqdm.write(f"CORRUPT: {path} - {message}")
    return valid, corrupt


def positive_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv", type=Path, default=Path(__file__).resolve().parents[2] / DEFAULT_CSV,
        help="Path to the original OASIS-3 sessions CSV",
    )
    parser.add_argument("--all-sessions", action="store_true",
                        help="Keep every T1w session rather than one per participant")
    parser.add_argument("--limit", type=positive_int, metavar="N",
                        help="Process only the first N selected sessions")
    parser.add_argument("--select-only", action="store_true",
                        help="Write selected_sessions.csv and exit without downloading")
    args = parser.parse_args()

    workdir = Path(__file__).resolve().parents[2]
    selected_path = workdir / "selected_sessions.csv"
    log_path = workdir / "download_log.csv"
    report_path = workdir / "verification_report.csv"
    output_root = workdir / "oasis3_t1w"

    try:
        selected = select_sessions(args.csv, args.all_sessions)
        selected.to_csv(selected_path, index=False)
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        print(f"CSV selection failed: {exc}", file=sys.stderr)
        return 1

    print(f"Selected {len(selected)} T1w sessions; wrote {selected_path}")
    if args.select_only:
        return 0

    if args.limit is not None:
        selected = selected.iloc[:args.limit]
    total = len(selected)
    if not total:
        print("No T1w sessions to process.")
        return 0

    # Collect authentication only if there is a session to download.
    pending = 0
    sessions_to_verify = []
    successes = failures = skipped = 0
    for row in selected.itertuples(index=False, name=None):
        subject, mr_id, _age, _scanner = row
        folder = output_root / subject / mr_id
        sessions_to_verify.append(folder)
        if any(folder.rglob("*.nii.gz")):
            skipped += 1
            append_log(log_path, mr_id, "", "skipped", str(folder), "Existing .nii.gz")
        else:
            pending += 1

    client = None
    if pending:
        username = input("NITRC username: ").strip()
        password = getpass.getpass("NITRC password: ")
        if not username or not password:
            print("Username and password are required.", file=sys.stderr)
            return 1
        client = NITRCClient(username, password)
        try:
            client.authenticate()
        except DownloadError as exc:
            print(f"Could not authenticate: {exc}", file=sys.stderr)
            client.close()
            return 1
        finally:
            # Never retain credentials longer than needed outside the client.
            del password

    try:
        for index, (subject, mr_id, _age, _scanner) in enumerate(
            tqdm(selected.itertuples(index=False, name=None), total=total,
                 desc="OASIS-3", unit="session"),
            start=1,
        ):
            folder = output_root / subject / mr_id
            if any(folder.rglob("*.nii.gz")):
                tqdm.write(f"[{index}/{total}] {mr_id}: skipped (already downloaded)")
                continue
            scan_id = ""
            zip_path = None
            try:
                assert client is not None
                scans = client.list_scans(mr_id)
                scan_id = choose_t1w_scan(scans)
                zip_path = client.download_zip(mr_id, scan_id, folder)
                nifti_paths = extract_zip_flat(zip_path, folder)
                successes += 1
                append_log(log_path, mr_id, scan_id, "downloaded",
                           str(nifti_paths[0]))
                tqdm.write(f"[{index}/{total}] {mr_id}: downloaded scan {scan_id}")
            except (DownloadError, OSError, zipfile.BadZipFile, EOFError) as exc:
                failures += 1
                message = str(exc)
                append_log(log_path, mr_id, scan_id, "failed", str(folder), message)
                tqdm.write(f"[{index}/{total}] {mr_id}: FAILED - {message}")
            finally:
                if zip_path is not None:
                    zip_path.unlink(missing_ok=True)
    finally:
        if client is not None:
            client.close()

    print("\nVerifying NIfTI files with nibabel...")
    try:
        valid, corrupt = verify_niftis(sessions_to_verify, report_path)
    except (DownloadError, OSError) as exc:
        print(f"Verification could not finish: {exc}", file=sys.stderr)
        return 1
    print(
        f"\nDone: {total} sessions; {successes} downloaded, {skipped} skipped, "
        f"{failures} failed. NIfTI verification: {valid} valid, {corrupt} corrupt."
    )
    print(f"Log: {log_path}\nVerification report: {report_path}")
    return 1 if failures or corrupt else 0


if __name__ == "__main__":
    raise SystemExit(main())
