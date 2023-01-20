"""
This file is part of nataili ("Homepage" = "https://github.com/Sygil-Dev/nataili").

Copyright 2022 hlky and Sygil-Dev
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
import hashlib
import json
import os
import sqlite3
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from nataili.util.logger import logger


class Cache:
    def __init__(self, cache_name, cache_subname=None, cache_parentname=None):
        """
        :param cache_name: Name of the cache
        :param cache_subname: Subfolder in the cache
        :param cache_parentname: Parent folder of the cache
        Examples:
        cache = Cache("test", "sub", "parent")
        path = self.path + "/parent/test/sub"

        cache = Cache("test", "sub")
        path = self.path + "/test/sub"

        cache = Cache("test")
        path = self.path + "/test"

        cache = Cache("test", cache_parentname="parent")
        path = self.path + "/parent/test"

        If cache file does not exist it is created
        If cache folder does not exist it is created
        """
        self.path = f"{Path.home()}/.cache/nataili/"
        if cache_parentname:
            self.path = os.path.join(self.path, cache_parentname)
        self.cache_dir = os.path.join(self.path, cache_name)
        if cache_subname:
            self.cache_dir = os.path.join(self.cache_dir, cache_subname)
        self.cache_db = os.path.join(self.cache_dir, "cache.db")
        logger.info(f"Cache file: {self.cache_db}")
        logger.info(f"Cache dir: {self.cache_dir}")
        os.makedirs(self.cache_dir, exist_ok=True)
        self.conn = sqlite3.connect(self.cache_db)
        self.cursor = self.conn.cursor()
        self.create_sqlite_db()

    def __del__(self):
        self.conn.close()

    def list_dir(self, input_directory, extensions=[".webp"]):
        """
        List all files in a directory
        :param input_directory: Directory to list
        :param extensions: List of extensions to filter for
        :return: List of files
        """
        files = []
        for file in tqdm(os.listdir(input_directory)):
            if os.path.splitext(file)[1] in extensions:
                files.append(os.path.splitext(file)[0])
        return files

    def get_all(self):
        """
        Get all entries from the cache
        :return: List of all entries
        """
        self.cursor.execute("SELECT file FROM cache")
        return [x[0] for x in self.cursor.fetchall()]

    def filter_list(self, input_list):
        """
        Filter a list
        :param input_list: List to filter
        :param filter_list: List to filter with
        :return: Filtered list
        """
        db_list = self.get_all()
        logger.info(f"Filtering {len(input_list)} files")
        logger.info(f"Filtering {len(db_list)} files")
        logger.info(f"Filtering {len(set(input_list) - set(db_list))} files")
        logger.info(f"First item in input_list: {input_list[0]}")
        logger.info(f"First item in db_list: {db_list[0]}")
        return list(set(input_list) - set(db_list))

    def hash_file(self, file_path):
        """
        Hash a file
        :param file_path: Path to the file
        :return: Hash of the file
        """
        with open(file_path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()

    def hash_pil_image(self, pil_image: Image.Image):
        """
        Hash a PIL image
        :param pil_image: PIL image
        :return: Hash of the PIL image
        """
        return hashlib.sha256(pil_image.tobytes()).hexdigest()

    def hash_pil_image_file(self, file_path):
        """
        Hash a PIL image
        :param file_path: Path to the file
        :return: Hash of the PIL image
        """
        pil_image = Image.open(file_path)
        return self.hash_pil_image(pil_image)

    def hash_files(self, files_list, input_directory, extensions=[".webp"]):
        """
        Hash all files in a directory
        :param input_directory: Directory to hash
        :return: List of hashes
        """
        pil_hashes = []
        file_hashes = []
        for file in tqdm(files_list):
            for extension in extensions:
                file = file + extension
                file_path = os.path.join(input_directory, file)
                file_hash = self.hash_file(file_path)
                pil_image_hash = self.hash_pil_image_file(file_path)
                pil_hashes.append(pil_image_hash)
                file_hashes.append(file_hash)
        return pil_hashes, file_hashes

    def rebuild_image_cache(self, input_directory):
        """
        Rebuild the image cache if it gets corrupted
        For file in input_directory: sha256 hash of file
        Check for npy with hash in cache_dir
        If npy exists: add to cache
        """
        self.create_sqlite_db()
        count = 0
        files = self.list_dir(input_directory, extensions=[".webp"])
        pil_hashes, file_hashes = self.hash_files(files, input_directory, extensions=[".webp"])
        cache_dir_files = self.list_dir(self.cache_dir, extensions=[".npy"])
        logger.info(f"Cache dir files: {len(cache_dir_files)}")
        logger.info(f"Files: {len(files)}")
        logger.info(f"PIL hashes: {len(pil_hashes)}")
        logger.info(f"File hashes: {len(file_hashes)}")
        # set()
        # cache_dir_files without pil_hashes and file_hashes = missing
        # to_readd = cache_dir_files - missing
        missing = set(cache_dir_files) - set(pil_hashes) - set(file_hashes)
        logger.info(f"Missing {len(missing)} files")
        to_readd = set(cache_dir_files) - missing
        logger.info(f"Readding {len(to_readd)} files")
        # files length and pil_hashes length and file_hashes and to_readd length should be the same
        # if not, then something is wrong
        # add every file to cache
        hashed_files = []
        for file, pil_hash, file_hash in zip(files, pil_hashes, file_hashes):
            hashed_files.append({"file": file, "hash": file_hash, "pil_hash": pil_hash})
        json.dump(hashed_files, open(os.path.join(self.cache_dir, "hashed_files.json"), "w"))
        self.populate_sqlite_db(hashed_files)
        logger.info(f"Rebuilt cache with {len(to_readd)} images")

    def create_sqlite_db(self):
        """
        Create a sqlite database from the cache
        """
        self.cursor.execute("CREATE TABLE IF NOT EXISTS cache (file text, hash text, pil_hash text)")
        self.conn.commit()

    def add_sqlite_row(self, file: str, hash: str, pil_hash: str, commit=True):
        """
        Add a row to the sqlite database
        """
        self.cursor.execute("INSERT INTO cache VALUES (?, ?, ?)", (file, hash, pil_hash))
        if commit:
            self.conn.commit()

    def populate_sqlite_db(self, list_of_files: list):
        """
        Populate the sqlite database from the cache
        """
        # Populate sqlite database
        for file in list_of_files:
            self.add_sqlite_row(file["file"], file["hash"], file["pil_hash"], commit=False)
        self.conn.commit()

    def key_exists(self, key):
        """
        Check if a key exists in the cache
        """
        query = "SELECT hash, pil_hash FROM cache WHERE file=?"
        self.cursor.execute(query, (key,))
        if self.cursor.fetchone():
            return True
        return False

    def get(self, file: str = None, file_hash: str = None, pil_hash: str = None, no_return=False):
        """
        Get a file from the cache
        """
        if not any([file, file_hash, pil_hash]):
            raise ValueError("At least one value must be provided to search the database")
        file = os.path.splitext(file)[0] if file else None
        query = "SELECT hash, pil_hash FROM cache WHERE "
        conditions = []
        values = []
        if file:
            conditions.append("file=?")
            values.append(file)
        if file_hash:
            conditions.append("hash=?")
            values.append(file_hash)
        if pil_hash:
            conditions.append("pil_hash=?")
            values.append(pil_hash)

        query += " OR ".join(conditions)

        self.cursor.execute(query, tuple(values))
        result = self.cursor.fetchone()
        if result:
            if no_return:
                return True
            file_hash, pil_hash = result
            if file_hash:
                file_path = os.path.join(self.cache_dir, file_hash + ".npy")
                if os.path.exists(file_path):
                    return file_path
            if pil_hash:
                file_path = os.path.join(self.cache_dir, pil_hash + ".npy")
                if os.path.exists(file_path):
                    return file_path
        return None
