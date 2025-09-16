import requests, json

BASE = 'http://ec2-54-233-36-108.sa-east-1.compute.amazonaws.com:8000'
USER = 'kaizen-poke'
PASS = '7`d$t>/ov%ZL8;g~*?Ei&07'

s = requests.Session()
s.trust_env = False  # ignora proxies/sessões do SO
url = f"{BASE}/login"

payload = {"username": USER, "password": PASS}
r = s.post(url, json=payload, timeout=20)

print("== REQUEST ==")
print("URL:", r.request.url)
print("Headers:", dict(r.request.headers))
print("Body:", r.request.body.decode() if isinstance(r.request.body, (bytes, bytearray)) else r.request.body)

print("\n== RESPONSE ==")
print("Status:", r.status_code)
print("Headers:", dict(r.headers))
print("Body:", r.text)

# Se der 200, prossegue:
if r.ok:
    tok = r.json().get("access_token")
    print("\nToken prefix:", tok[:20])
    s.headers.update({"Authorization": f"Bearer {tok}"})
    t = s.get(f"{BASE}/pokemon", timeout=20)
    print("\nGET /pokemon ->", t.status_code, t.text[:200])
