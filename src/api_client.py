import json
from typing import Dict, Any, Iterable, List, Optional
import requests
from requests import Session
from urllib.parse import urljoin
from config import settings

class APIClient:
    def __init__(self, base_url: str, username: str, password: str):
        self.base_url = base_url.rstrip("/")
        self.username = username
        self.password = password
        self.s: Session = requests.Session()
        self.token: Optional[str] = None

    # Tenta vários formatos comuns de autenticação JWT em FastAPI
    def authenticate(self) -> str:
        trials = [
            # (path, payload, headers, use_form)
            ("/token", {"username": self.username, "password": self.password, "grant_type": "password"}, {"Content-Type": "application/x-www-form-urlencoded"}, True),
            ("/login", {"username": self.username, "password": self.password}, {"Content-Type": "application/json"}, False),
            ("/auth/login", {"username": self.username, "password": self.password}, {"Content-Type": "application/json"}, False),
            ("/jwt/login", {"username": self.username, "password": self.password}, {"Content-Type": "application/json"}, False),
        ]
        errors = []
        for path, payload, headers, use_form in trials:
            url = urljoin(self.base_url + "/", path.lstrip("/"))
            try:
                if use_form:
                    resp = self.s.post(url, data=payload, headers=headers, timeout=30)
                else:
                    resp = self.s.post(url, json=payload, headers=headers, timeout=30)
                if resp.status_code < 400:
                    data = resp.json()
                    token = data.get("access_token") or data.get("token") or data.get("access")
                    if token:
                        self.token = token
                        self.s.headers.update({"Authorization": f"Bearer {self.token}"})
                        return self.token
            except Exception as e:
                errors.append((path, str(e)))
        raise RuntimeError(f"Falha ao autenticar. Tentativas: {errors}")

    def get_openapi(self) -> Dict[str, Any]:
        # padrão do FastAPI
        for path in ["/openapi.json", "/docs/openapi.json"]:
            url = urljoin(self.base_url + "/", path.lstrip("/"))
            try:
                r = self.s.get(url, timeout=30)
                if r.status_code == 200:
                    return r.json()
            except Exception:
                pass
        return {}

    def list_get_endpoints(self) -> List[str]:
        spec = self.get_openapi()
        paths = spec.get("paths", {}) if spec else {}
        get_paths = []
        for p, ops in paths.items():
            if "get" in ops:
                get_paths.append(p)
        # Heurística: ignorar docs/openapi
        return [p for p in get_paths if not p.startswith("/docs") and not p.startswith("/openapi")]

    def get(self, path: str, params: Dict[str, Any] = None) -> Any:
        url = urljoin(self.base_url + "/", path.lstrip("/"))
        r = self.s.get(url, params=params or {}, timeout=60)
        r.raise_for_status()
        try:
            return r.json()
        except json.JSONDecodeError:
            return r.text

    def paginate_all(self, path: str, page_param: str = "page", limit_param: str = "limit", limit: int = 200) -> Iterable[Dict[str, Any]]:
        # Paginação genérica: tenta page/limit; se a API devolver 'next' no payload, usa também
        page = 1
        while True:
            data = self.get(path, params={page_param: page, limit_param: limit})
            # formatos comuns
            items = None
            if isinstance(data, dict):
                # tenta encontrar lista principal
                for k in ["results", "items", "data", "content", "rows"]:
                    if k in data and isinstance(data[k], list):
                        items = data[k]
                        break
                if items is None and isinstance(data.get(path.strip("/").split("/")[-1], None), list):
                    items = data[path.strip("/").split("/")[-1]]
                if items is None and isinstance(data.get("0", None), list):  # fallback esquisito
                    items = data["0"]
                if items is None and isinstance(data, dict):
                    # pode ser já a lista
                    if all(isinstance(v, (str, int, float, bool, type(None))) for v in data.values()):
                        items = [data]
            elif isinstance(data, list):
                items = data
            else:
                items = []

            for it in items or []:
                yield it

            has_next = False
            if isinstance(data, dict):
                nxt = data.get("next") or data.get("next_page") or data.get("links", {}).get("next")
                if nxt:
                    has_next = True
                    page += 1
            if not has_next and (not items or len(items) < limit):
                break
            if not has_next:
                page += 1
