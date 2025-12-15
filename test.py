import requests
r = requests.get("http://127.0.0.1:9621/graphs",
                 params={"label":"羟碳钠铍石","max_depth":2,"max_nodes":200}, timeout=30)
print(r.status_code)
print(r.text)