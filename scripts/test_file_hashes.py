import os
import re

from nataili.model_manager.base import BaseModelManager

DIRECTORY = "test_imgs/"
FILEPATTERN = "^.*\.(md5|sha256)$"


class TestImageHashes:
    def __init__(self, filename: str, md5hash: str, sha256hash: str):
        self.filename = filename
        self.md5hash = md5hash
        self.sha256hash = sha256hash


TEST_HASHES = [
    TestImageHashes("bag.png",
                    md5hash="1a443b1f39203a9629c5ad4088915050",
                    sha256hash="667155baed8357af22adb5738a33e1063b58149bedb532b28a1a7295b28d371c"),
    TestImageHashes("bird.png",
                    md5hash="f58b5d829aaa23bc4efa0e4420c99760",
                    sha256hash="cad49fc7d3071b2bcd078bc8dde365f8fa62eaa6d43705fd50c212794a3aac35"),
]

# Calculate hashes, assert equals
for h in TEST_HASHES:
    calculated_md5sum = BaseModelManager.get_file_md5sum_hash(DIRECTORY + h.filename)
    calculated_sha256sum = BaseModelManager.get_file_sha256_hash(DIRECTORY + h.filename)
    assert calculated_md5sum == h.md5hash
    assert calculated_sha256sum == h.sha256hash

# Remove cached hash files
for f in os.listdir(DIRECTORY):
    if re.search(FILEPATTERN, f):
        os.remove(os.path.join(DIRECTORY, f))
