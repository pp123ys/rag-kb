import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from ragkb.mcp_server.server import build_server

server = build_server()

print("=== 1. search: A-100 单价 ===")
r = server._search("A-100 型号的单价是多少", top_k=3)
for item in r["results"]:
    print(f"  [{item['source']}] v{item['version']}: {item['text'][:60]}")

print()
print("=== 2. search: 保修期 ===")
r = server._search("保修期多久", top_k=3)
for item in r["results"]:
    print(f"  [{item['source']}] v{item['version']}: {item['text'][:60]}")

print()
print("=== 3. search: 知识库外问题（防幻觉）===")
r = server._search("量子计算加密算法原理", top_k=3)
print("  empty_reason =", r.get("empty_reason"), "| results =", r["results"])

print()
print("=== 4. retrieve_table: demo.xlsx-报价表 ===")
t = server._retrieve_table(table_id="demo.xlsx-报价表")
for row in (t["rows"] or [])[:3]:
    print(" ", row)

print()
print("=== 5. search: C-300（Excel 表格语义检索）===")
r = server._search("C-300 型号库存多少", top_k=3)
for item in r["results"]:
    print(f"  [{item['source']}] v{item['version']}: {item['text'][:60]}")

print()
print("=== 6. list_versions ===")
v = server._list_versions("c4961702-b305-4c19-a59a-9992f96a2479")
print(" ", v)
