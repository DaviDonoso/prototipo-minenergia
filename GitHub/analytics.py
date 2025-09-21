# analytics.py
import pandas as pd

TRIM_CIERRE = {1:"marzo", 2:"junio", 3:"septiembre", 4:"diciembre"}

def _apply_scope(df, scope):
    """scope: dict opcional con filtros: anio, capitulo, programa, subtitulo_range, incluir_ingresos"""
    out = df.copy()
    if scope.get("anio"):
        years = scope["anio"]; out = out[out["anio"].isin(years)]
    if scope.get("capitulo"):
        out = out[out["capitulo"].str.lower().isin([x.lower() for x in scope["capitulo"]])]
    if scope.get("programa"):
        out = out[out["programa"].str.lower().isin([x.lower() for x in scope["programa"]])]
    if not scope.get("incluir_ingresos", False):
        out = out[(out["tipo_mov"].isna()) | (out["tipo_mov"] != "INGRESO")]  # por defecto excluye ingresos
    if scope.get("subtitulo_range"):
        lo, hi = scope["subtitulo_range"]
        out = out[(out["subtitulo"]>=lo) & (out["subtitulo"]<=hi)]
    return out

def totales_anuales(df, scope):
    d = _apply_scope(df, scope)
    # usar solo cierre de año (diciembre) si hay periodos acumulados
    d_year = d[(d["mes_cierre"]=="diciembre")]
    if d_year.empty:
        # fallback: suma todo el año (no ideal pero sirve si no hay cierre explícito)
        d_year = d
    agg = d_year.groupby("anio", as_index=False)["monto"].sum().rename(columns={"monto":"total_anual"})
    return agg.sort_values("anio")

# analytics.py

def totales_trimestrales(df, scope):
    d = _apply_scope(df, scope)
    acc = []
    for q, mes in TRIM_CIERRE.items():
        tmp = d[d["mes_cierre"]==mes].groupby(["anio"], as_index=False)["monto"].sum()
        tmp["quarter"] = q
        tmp = tmp.rename(columns={"monto":"acumulado"})
        acc.append(tmp)
    acc = pd.concat(acc, ignore_index=True) if acc else pd.DataFrame(columns=["anio","quarter","acumulado"])

    q1 = acc[acc["quarter"]==1][["anio","acumulado"]].rename(columns={"acumulado":"Q1"})
    q2 = acc[acc["quarter"]==2][["anio","acumulado"]].rename(columns={"acumulado":"_Q2"})
    q3 = acc[acc["quarter"]==3][["anio","acumulado"]].rename(columns={"acumulado":"_Q3"})
    q4 = acc[acc["quarter"]==4][["anio","acumulado"]].rename(columns={"acumulado":"_Q4"})
    res = q1.merge(q2, on="anio", how="outer").merge(q3, on="anio", how="outer").merge(q4, on="anio", how="outer")

    res["Q2"] = (res["_Q2"] - res["Q1"]).clip(lower=0)
    res["Q3"] = (res["_Q3"] - res["_Q2"]).clip(lower=0)
    res["Q4"] = (res["_Q4"] - res["_Q3"]).clip(lower=0)
    res = res.drop(columns=["_Q2","_Q3","_Q4"]).fillna(0)

    # ✅ move this INSIDE the function
    res["alerta_datos_faltantes"] = res.apply(
        lambda r: any(v == 0 for v in [r["Q1"], r["Q2"], r["Q3"], r["Q4"]]),
        axis=1
    )

    return res.sort_values("anio")

def serie_mensual(df, scope):
    d = _apply_scope(df, scope)
    order = {m:i for i,m in enumerate(["enero","febrero","marzo","abril","mayo","junio","julio","agosto","septiembre","octubre","noviembre","diciembre"], start=1)}
    d = d[d["mes_cierre"].isin(order.keys())]
    d["mes_num"] = d["mes_cierre"].map(order)
    agg = d.groupby(["anio","mes_num","mes_cierre"], as_index=False)["monto"].sum().sort_values(["anio","mes_num"])
    return agg.rename(columns={"monto":"acumulado_mes"})

def desglose_por_denominacion(df, scope, top=20, periodo="anual"):
    d = _apply_scope(df, scope)
    if periodo=="anual":
        d = d[d["mes_cierre"]=="diciembre"] if "mes_cierre" in d.columns else d
        grp = d.groupby(["anio","denominacion"], as_index=False)["monto"].sum()
        grp = grp.sort_values(["anio","monto"], ascending=[True, False])
        return grp.groupby("anio").head(top)
    elif periodo=="q4":
        d = d[d["mes_cierre"]=="diciembre"]
        return d.groupby(["anio","denominacion"], as_index=False)["monto"].sum().sort_values(["anio","monto"], ascending=[True, False]).groupby("anio").head(top)
    else:
        # genérico
        return d.groupby(["anio","denominacion"], as_index=False)["monto"].sum().sort_values(["anio","monto"], ascending=[True, False]).groupby("anio").head(top)

def _apply_scope(df, scope):
    out = df.copy()
    if scope.get("anio"):
        out = out[out["anio"].isin(scope["anio"])]
    if scope.get("capitulo"):
        out = out[out["capitulo"].str.lower().isin([x.lower() for x in scope["capitulo"]])]
        # 🔒 si no se pidió programa explícito, quedar solo con el nivel capítulo (programa vacío)
        if not scope.get("programa"):
            out = out[(out["programa"] == "") | (out["programa"].isna())]
    if scope.get("programa"):
        out = out[out["programa"].str.lower().isin([x.lower() for x in scope["programa"]])]
    if not scope.get("incluir_ingresos", False):
        out = out[(out["tipo_mov"].isna()) | (out["tipo_mov"] != "INGRESO")]
    if scope.get("subtitulo_range"):
        lo, hi = scope["subtitulo_range"]
        out = out[(out["subtitulo"]>=lo) & (out["subtitulo"]<=hi)]
    return out
