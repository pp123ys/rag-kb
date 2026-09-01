import sqlite3

con = sqlite3.connect("data/ragkb.db")
cur = con.cursor()
print("== tables_index 前 8 行 ==")
cols = [d[0] for d in cur.execute("SELECT * FROM tables_index LIMIT 1").description]
print("columns:", cols)
for row in cur.execute("SELECT * FROM tables_index LIMIT 8").fetchall():
    print(row)
print("== document_versions 前 5 行 ==")
for row in cur.execute("SELECT * FROM document_versions LIMIT 5").fetchall():
    print(row)
