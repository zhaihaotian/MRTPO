"""Download the Text-to-SQL sources into <root>/raw and check their hashes.

Layout written under <root>/raw:
  spider_data/                 Spider 1.0 (train, dev, test, databases, test_database)
  bird_dev_hf/dev_20240627/    BIRD dev (dev.json, dev_databases/)
  sqlr1_db_info/               SQL-R1 schema and value caches for Spider dev/test and BIRD dev
  variants/                    Spider-Syn dev, Spider-Realistic, Spider-DK (+ its 3 new databases)
  omnisql/                     OmniSQL chain-of-thought for the Spider training set (OPSD reference solutions)
"""
import argparse
import hashlib
import io
import shutil
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent

SPIDER = dict(url='https://drive.usercontent.google.com/download?id=1403EGqzIDoHMdQF4c9Bkyl7dZLZ5Wt6J'
                  '&export=download&confirm=t',
              md5='b039fdb9e1e3adc386a91c5b2211667e')
BIRD_DEV = dict(url='https://bird-bench.oss-cn-beijing.aliyuncs.com/dev.zip',
                md5='04b4af221c9186361f09b16abfd917ec')

SQL_R1 = 'https://raw.githubusercontent.com/DataArcTech/SQL-R1/ccfb511e81fdb3a8e87bd3da7f3d71882cedd532/db_info'
SPIDER_SYN = 'https://raw.githubusercontent.com/ygan/Spider-Syn/0b996a57b7e329c14c300a8fb13661da128a9ddc'
SPIDER_DK = 'https://raw.githubusercontent.com/ygan/Spider-DK/5e16fa1f8cdbb401f3b55d2afb5a7c7435360590'
SPIDER_REALISTIC = 'https://zenodo.org/api/records/5205322/files/spider-realistic.json/content'
OMNISQL_ZIP = 'https://huggingface.co/datasets/seeklhy/OmniSQL-datasets/resolve/e1db8bb10ef11f92eacd8d87801d716ba0e9e9be/data.zip'
OMNISQL_MEMBER = 'data/spider/train_spider_enhanced_with_cot.json'

# (path under raw/, source URL, sha256)
FILES = [
    ('sqlr1_db_info/spiderdev_db_id2db_info.json', f'{SQL_R1}/spiderdev_db_id2db_info.json',
     'bcd34831fc658d7bea69e16c8bd29a279ef0991b3a0228d9ca0404a106e50748'),
    ('sqlr1_db_info/spiderdev_db_id2sampled_db_values.json', f'{SQL_R1}/spiderdev_db_id2sampled_db_values.json',
     'e891fcb1fb6d0c7c656ce7f02a66179a768b239e98844bc42fbf1f16d4bdbbe8'),
    ('sqlr1_db_info/spidertest_db_id2db_info.json', f'{SQL_R1}/spidertest_db_id2db_info.json',
     'b084adc0f5cf18d24f132e7adeb33ff923472d147ea996b93c67c68791ed6a4d'),
    ('sqlr1_db_info/spidertest_db_id2sampled_db_values.json', f'{SQL_R1}/spidertest_db_id2sampled_db_values.json',
     '292ebd57ca9ddbb690d603566c24c697720ce3f4409abeedd0213b8d02f1b4dd'),
    ('sqlr1_db_info/bird_db_id2db_info.json', f'{SQL_R1}/bird_db_id2db_info.json',
     'cff487a7d16e70e323a23c85847a04b0038fee244eaa19a3c91b93ae41275d48'),
    ('sqlr1_db_info/bird_db_id2sampled_db_values.json', f'{SQL_R1}/bird_db_id2sampled_db_values.json',
     '3c951f15b156822a1aff0f318783cb7439379d446d2e56ef247ed1cb71cf4db3'),
    ('variants/spider_syn_dev.json', f'{SPIDER_SYN}/Spider-Syn/dev.json',
     'c70b19cd000376d46331dce518c2bc5a672950c9bf2ee6ade7de519e373ee38d'),
    ('variants/spider_realistic.json', SPIDER_REALISTIC,
     '6d16e1be16eb47909430da083e1e2c29936814f141c31c6de4137a43adbf5ffa'),
    ('variants/spider_dk.json', f'{SPIDER_DK}/Spider-DK.json',
     'f8590106c56e0850c2e1412179a78fdcf86f02226ab9f81a4c367d1ae8575397'),
    ('variants/spider_dk_tables.json', f'{SPIDER_DK}/tables.json',
     'aeaa9268303e7d5871c3e6bff4330b997fcdaea4414bba32d0160406527ffe35'),
    ('variants/dk_database/new_concert_singer/new_concert_singer.sqlite',
     f'{SPIDER_DK}/database/new_concert_singer/new_concert_singer.sqlite',
     '3640db4739de19f1cb81fa6f5abe938f22beebd51889036f4a51ecb43135eaa0'),
    ('variants/dk_database/new_orchestra/new_orchestra.sqlite',
     f'{SPIDER_DK}/database/new_orchestra/new_orchestra.sqlite',
     '8e452adcb335fad523a54d5ba4ebb8d9b0ad85e30e849fcc40ac5f0d956831a9'),
    ('variants/dk_database/new_pets_1/new_pets_1.sqlite',
     f'{SPIDER_DK}/database/new_pets_1/new_pets_1.sqlite',
     '270d319add83d7ced59db0119c71f3ab101ced02a77a5a4e58ac88fdaeadb13d'),
]
OMNISQL_SHA256 = '4a6f0d272c5e945060f8953deb766a78f5b1dd90855d732fe21225a4c341a144'


def digest(path, name):
    h = hashlib.new(name)
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def download(url, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.download')
    with urllib.request.urlopen(url, timeout=300) as response, open(temporary, 'wb') as out:
        shutil.copyfileobj(response, out, 1 << 20)
    temporary.replace(path)


def check(path, expected, name='sha256'):
    actual = digest(path, name)
    if actual != expected:
        raise SystemExit(f'{name} mismatch for {path}: {actual} != {expected}')
    print(f'ok  {path}  {name} {actual}')


def extract(archive, target):
    """Unzip without the macOS resource-fork entries."""
    with zipfile.ZipFile(archive) as z:
        z.extractall(target, [m for m in z.namelist() if not m.startswith('__MACOSX/') and '/.DS_Store' not in m])


class RemoteFile(io.RawIOBase):
    """Seekable read-only view of a remote file through HTTP range requests."""

    def __init__(self, url):
        with urllib.request.urlopen(urllib.request.Request(url, method='HEAD'), timeout=120) as response:
            self.url, self.size = response.geturl(), int(response.headers['Content-Length'])
        self.position = 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        self.position = offset if whence == 0 else self.position + offset if whence == 1 else self.size + offset
        return self.position

    def readinto(self, buffer):
        n = min(len(buffer), self.size - self.position)
        if n <= 0:
            return 0
        request = urllib.request.Request(self.url, headers={'Range': f'bytes={self.position}-{self.position + n - 1}'})
        with urllib.request.urlopen(request, timeout=300) as response:
            data = response.read()
        buffer[:len(data)] = data
        self.position += len(data)
        return len(data)


def fetch_spider(raw, spider_zip):
    if (raw / 'spider_data/database').is_dir() and (raw / 'spider_data/test_database').is_dir():
        print(f'ok  {raw / "spider_data"} (already extracted)')
        return
    archive = Path(spider_zip) if spider_zip else raw / 'spider_data.zip'
    if not archive.is_file():
        print(f'downloading Spider (206 MB) -> {archive}')
        download(SPIDER['url'], archive)
    check(archive, SPIDER['md5'], 'md5')
    extract(archive, raw)


def fetch_bird(raw):
    base = raw / 'bird_dev_hf'
    if (base / 'dev_20240627/dev_databases').is_dir():
        print(f'ok  {base / "dev_20240627"} (already extracted)')
        return
    archive = base / 'dev.zip'
    if not archive.is_file():
        print(f'downloading BIRD dev (346 MB) -> {archive}')
        download(BIRD_DEV['url'], archive)
    check(archive, BIRD_DEV['md5'], 'md5')
    extract(archive, base)
    extract(base / 'dev_20240627/dev_databases.zip', base / 'dev_20240627')


def fetch_files(raw):
    for relative, url, sha in FILES:
        path = raw / relative
        if not path.is_file():
            download(url, path)
        check(path, sha)


def fetch_omnisql(raw):
    """Read one member (33 MB) of OmniSQL's 22 GB data.zip with range requests."""
    path = raw / 'omnisql/train_spider_enhanced_with_cot.json'
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        remote = io.BufferedReader(RemoteFile(OMNISQL_ZIP), buffer_size=1 << 20)
        with zipfile.ZipFile(remote) as z, z.open(OMNISQL_MEMBER) as source, open(path, 'wb') as out:
            shutil.copyfileobj(source, out, 1 << 20)
    check(path, OMNISQL_SHA256)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--root', type=Path, default=ROOT, help='data root (default: this folder)')
    parser.add_argument('--spider-zip', help='use an already downloaded spider_data.zip')
    args = parser.parse_args()
    raw = args.root / 'raw'
    raw.mkdir(parents=True, exist_ok=True)
    fetch_files(raw)
    fetch_omnisql(raw)
    fetch_spider(raw, args.spider_zip)
    fetch_bird(raw)
    for name in ('spider_data/database', 'spider_data/test_database', 'bird_dev_hf/dev_20240627/dev_databases',
                 'variants/dk_database'):
        print(f'{name}: {len(list((raw / name).rglob("*.sqlite")))} sqlite files')


if __name__ == '__main__':
    main()
