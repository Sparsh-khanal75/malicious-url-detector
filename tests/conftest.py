import random

import joblib
import pytest

from urlguard.model import UrlGuardModel

BENIGN_SITES = [
    "wikipedia.org/wiki/",
    "github.com/",
    "bbc.co.uk/news/",
    "python.org/doc/",
    "nasa.gov/missions/",
    "stackoverflow.com/questions/",
    "nytimes.com/2026/",
    "imdb.com/title/",
    "mit.edu/courses/",
    "reddit.com/r/",
]
BAD_PATTERNS = [
    "paypal.com.secure-{n}.tk/webscr?cmd=_login",
    "{n}.xyz/login/verify/account.php",
    "192.168.{n}.4/bank/confirm",
    "secure-update-{n}.info/signin/",
    "free-bonus-{n}.ml/claim.exe",
]


@pytest.fixture(scope="session")
def toy_data():
    rnd = random.Random(0)
    urls, y, groups = [], [], []
    for _ in range(300):
        site = rnd.choice(BENIGN_SITES)
        urls.append(site + f"page_{rnd.randint(1, 9999)}")
        y.append(0)
        groups.append(site.split("/")[0])
    for _ in range(150):
        pat = rnd.choice(BAD_PATTERNS)
        u = pat.format(n=rnd.randint(1, 250))
        urls.append(u)
        y.append(1)
        groups.append(u.split("/")[0])
    return urls, y, groups


@pytest.fixture(scope="session")
def toy_model(toy_data):
    urls, y, groups = toy_data
    return UrlGuardModel(max_features=5000).fit(urls, y, groups)


@pytest.fixture(scope="session")
def model_path(toy_model, tmp_path_factory):
    path = tmp_path_factory.mktemp("models") / "toy.joblib"
    joblib.dump(toy_model, path)
    return path
