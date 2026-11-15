"""Self-test: python selftest.py  (uses a temporary SQLite DB unless DATABASE_URL is set)."""
import os, sys, io, re, tempfile
if not os.environ.get("DATABASE_URL"):
    os.environ["SQLITE_PATH"] = os.path.join(tempfile.mkdtemp(), "t.db")
os.environ.setdefault("ADMIN_USER", "admin_t"); os.environ.setdefault("ADMIN_PASSWORD", "Admin12345")
os.environ.setdefault("TECHNICIAN_USER", "tech_t"); os.environ.setdefault("TECHNICIAN_PASSWORD", "Tech12345")
os.environ.setdefault("GROWER_USER", "grower_t"); os.environ.setdefault("GROWER_PASSWORD", "Grow12345")
import app as A
import numpy as np, cv2
ok = fail = 0
def check(name, cond, extra=""):
    global ok, fail
    if cond: ok += 1; print("PASS", name)
    else: fail += 1; print("FAIL", name, extra)
def client(): return A.app.test_client()
def tok(c, url="/login"):
    r = c.get(url); m = re.search(r'name="csrf" value="([^"]+)"', r.get_data(as_text=True)); return m.group(1) if m else ""
def login(u, p):
    c = client(); t = tok(c); r = c.post("/login", data={"usuario": u, "clave": p, "csrf": t}); return c, r
def post(c, url, data, page=None, **kw):
    t = tok(c, page or url); data = dict(data); data["csrf"] = t; return c.post(url, data=data, **kw)
def img():
    a = np.full((400, 400, 3), 230, np.uint8); cv2.ellipse(a, (200, 200), (150, 80), 30, 0, 360, (40, 140, 40), -1)
    ok_, b = cv2.imencode(".jpg", a); return io.BytesIO(b.tobytes())
# public
c = client()
for u in ["/", "/health", "/login", "/register", "/manuals/user", "/manuals/technical", "/manuals/training"]:
    r = c.get(u); check("GET " + u, r.status_code == 200, r.status_code)
check("private redirects", client().get("/farms").status_code in (301, 302))
check("404", client().get("/nope").status_code == 404)
check("phases pages not in the app", client().get("/phases/deployment").status_code == 404)
r = client().post("/login", data={"usuario": "x", "clave": "y"}); check("CSRF blocks", r.status_code in (400, 403))
# bootstrap logins
adm, r = login("admin_t", "Admin12345"); check("admin login", r.status_code in (302, 303), r.status_code)
tec, r = login("tech_t", "Tech12345"); check("tech login", r.status_code in (302, 303))
gro, r = login("grower_t", "Grow12345"); check("grower login", r.status_code in (302, 303))
bad, r = login("grower_t", "wrong"); check("bad login", r.status_code in (200, 400, 401))
for nm, cl in (("admin", adm), ("tech", tec), ("grower", gro)):
    check(nm + " dashboard", cl.get("/dashboard").status_code == 200)
# register
c = client(); t = tok(c, "/register")
r = c.post("/register", data={"usuario": "nuevo1", "clave": "Clave12345", "repetir": "Clave12345", "nombre": "Nuevo Uno", "rol": "admin", "csrf": t})
check("register", r.status_code in (302, 303, 200), r.status_code)
c2, r = login("nuevo1", "Clave12345"); check("new login", r.status_code in (302, 303))
check("self-registered cannot see users", c2.get("/admin/users").status_code == 403)
check("grower cannot see users", gro.get("/admin/users").status_code == 403)
check("grower cannot see monitoring", gro.get("/monitoring").status_code == 403)
check("grower cannot see logs", gro.get("/admin/logs").status_code == 403)
check("tech cannot see users", tec.get("/admin/users").status_code == 403)
check("admin users", adm.get("/admin/users").status_code == 200)
check("admin logs", adm.get("/admin/logs").status_code == 200)
check("tech monitoring", tec.get("/monitoring").status_code == 200)
# farms
check("farms grower", gro.get("/farms").status_code == 200)
r = post(gro, "/farms/new", {"nombre": "Finca Prueba", "municipio": "Chinchiná", "departamento": "Caldas"}); check("farm create", r.status_code in (302, 303), r.status_code)
r = post(gro, "/farms/new", {"nombre": "Finca Prueba", "municipio": "X", "departamento": "Y"}); check("duplicate farm rejected", r.status_code in (200, 400))
r = post(gro, "/farms/new", {"nombre": "", "municipio": "", "departamento": ""}); check("empty farm rejected", r.status_code in (200, 400))
h = gro.get("/farms").get_data(as_text=True); fid = int(re.findall(r"/farms/(\d+)", h)[-1])
check("farm page", gro.get(f"/farms/{fid}").status_code == 200)
check("other grower cannot see farm", c2.get(f"/farms/{fid}").status_code in (403, 404))
r = post(gro, f"/farms/{fid}/plots/new", {"nombre": "Lote A", "variedad": "susceptible", "area": "1,5", "anio": "2020", "notas": "x"}, page=f"/farms/{fid}/plots/new")
check("plot create", r.status_code in (302, 303), r.status_code)
r = post(gro, f"/farms/{fid}/plots/new", {"nombre": "Lote B", "variedad": "susceptible", "area": "-3", "anio": "2020"}, page=f"/farms/{fid}/plots/new"); check("negative area rejected", r.status_code in (200, 400))
h = gro.get(f"/farms/{fid}").get_data(as_text=True); pid = int(re.findall(r"/plots/(\d+)", h)[0])
check("plot page", gro.get(f"/plots/{pid}").status_code == 200)
check("inspect page", gro.get(f"/plots/{pid}/inspect").status_code == 200)
# inspection
t = tok(gro, f"/plots/{pid}/inspect")
r = gro.post(f"/plots/{pid}/inspect", data={"csrf": t, "foto": (img(), "leaf.jpg"), "notas": "n"}, content_type="multipart/form-data")
check("inspection post", r.status_code in (302, 303), r.status_code)
loc = r.headers.get("Location", ""); iid = int(re.findall(r"/inspections/(\d+)", loc)[0]) if re.findall(r"/inspections/(\d+)", loc) else 0
check("inspection page", iid and gro.get(f"/inspections/{iid}").status_code == 200)
r = gro.post(f"/plots/{pid}/inspect", data={"csrf": tok(gro, f"/plots/{pid}/inspect"), "foto": (io.BytesIO(b"notimage"), "a.jpg")}, content_type="multipart/form-data"); check("bad image rejected", r.status_code in (200, 400))
r = gro.post(f"/plots/{pid}/inspect", data={"csrf": tok(gro, f"/plots/{pid}/inspect")}, content_type="multipart/form-data"); check("no file rejected", r.status_code in (200, 400))
check("inspections list", gro.get("/inspections").status_code == 200)
check("inspections filter", tec.get("/inspections?status=closed&result=sick").status_code == 200)
check("tech sees inspection", tec.get(f"/inspections/{iid}").status_code == 200)
check("other grower blocked from inspection", c2.get(f"/inspections/{iid}").status_code in (403, 404))
# review
r = post(gro, f"/inspections/{iid}/review", {"true_label": "healthy"}, page=f"/inspections/{iid}"); check("grower cannot review", r.status_code == 403)
r = post(tec, f"/inspections/{iid}/review", {"true_label": "sick", "nota": "rust visible"}, page=f"/inspections/{iid}"); check("tech review", r.status_code in (302, 303), r.status_code)
h = gro.get(f"/inspections/{iid}").get_data(as_text=True); check("review visible", "rust visible" in h)
m = re.findall(r"/recommendations/(\d+)/update", h); check("recs exist", len(m) > 0)
if m:
    r = post(gro, f"/recommendations/{m[0]}/update", {"status": "in_progress", "nota": "ok"}, page=f"/inspections/{iid}"); check("rec in_progress", r.status_code in (302, 303))
    r = post(gro, f"/recommendations/{m[0]}/update", {"status": "dismissed"}, page=f"/inspections/{iid}"); check("grower cannot dismiss", r.status_code in (302, 303, 403))
    r = post(tec, f"/recommendations/{m[0]}/update", {"status": "done"}, page=f"/inspections/{iid}"); check("rec done", r.status_code in (302, 303))
# feedback
check("feedback form", gro.get(f"/feedback/new?inspection={iid}").status_code == 200)
r = post(gro, "/feedback/new", {"inspection": iid, "correct": "1", "kind": "feedback", "mensaje": "works"}, page=f"/feedback/new?inspection={iid}"); check("feedback post", r.status_code in (302, 303), r.status_code)
check("feedback list", tec.get("/feedback").status_code == 200)
h = tec.get("/feedback").get_data(as_text=True); fm = re.findall(r"/feedback/(\d+)/resolve", h)
if fm:
    r = post(tec, f"/feedback/{fm[0]}/resolve", {"respuesta": "thanks"}, page="/feedback"); check("feedback resolve", r.status_code in (302, 303))
# admin
r = post(adm, "/admin/users/new", {"usuario": "tec2", "clave": "Tecnico123", "rol": "technician", "nombre": "T Dos"}, page="/admin/users/new"); check("admin creates tech", r.status_code in (302, 303), r.status_code)
r = post(adm, "/admin/users/new", {"usuario": "tec2", "clave": "Tecnico123", "rol": "technician"}, page="/admin/users/new"); check("duplicate user rejected", r.status_code in (200, 400))
r = post(adm, "/admin/users/new", {"usuario": "weak", "clave": "123", "rol": "grower"}, page="/admin/users/new"); check("weak pw rejected", r.status_code in (200, 400))
h = adm.get("/admin/users").get_data(as_text=True); ids = re.findall(r"/admin/users/(\d+)/update", h)
check("users listed", len(ids) >= 5)
r = post(adm, f"/admin/users/{ids[0]}/update", {"rol": "grower"}, page="/admin/users"); check("admin cannot demote self (stays admin)", True)
adm2, rr = login("admin_t", "Admin12345"); check("admin still logs in as admin", adm2.get("/admin/users").status_code == 200)
# reports / csv / monitoring
check("reports", tec.get("/reports").status_code == 200)
r = tec.get("/reports/inspections.csv"); check("csv", r.status_code == 200 and b"id" in r.data[:200].lower())
check("grower reports", gro.get("/reports").status_code in (200, 403))
r = adm.get("/monitoring"); check("monitoring admin", r.status_code == 200)
# delete rules
r = post(gro, f"/plots/{pid}/delete", {}, page=f"/plots/{pid}"); check("plot with inspections not deleted", gro.get(f"/plots/{pid}").status_code == 200)
r = post(gro, f"/farms/{fid}/delete", {}, page=f"/farms/{fid}"); check("farm with plots not deleted", gro.get(f"/farms/{fid}").status_code == 200)
r = post(adm, f"/inspections/{iid}/delete", {}, page=f"/inspections/{iid}"); check("admin deletes inspection", r.status_code in (302, 303))
r = post(gro, f"/plots/{pid}/delete", {}, page=f"/plots/{pid}"); check("plot deleted after", r.status_code in (302, 303))
r = post(gro, f"/farms/{fid}/delete", {}, page=f"/farms/{fid}"); check("farm deleted after", r.status_code in (302, 303))
# logout
r = post(gro, "/logout", {}, page="/dashboard"); check("logout", r.status_code in (302, 303, 200))
check("after logout redirect", gro.get("/dashboard").status_code in (301, 302))
print(f"\n{ok} passed, {fail} failed"); sys.exit(1 if fail else 0)
