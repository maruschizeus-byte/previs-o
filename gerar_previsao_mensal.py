#!/usr/bin/env python3
"""Previsão dos próximos 3 meses (tendência) e anomalia prevista, por área.

Para cada área (Brasil, 5 regiões, 27 estados e, se pedido, grupos como o
AMAGGI) gera, para os 3 meses e para o trimestre:
  chuva prevista (mm), temperatura média prevista (°C),
  anomalia de chuva (mm) e anomalia de temperatura (°C).

Fonte: ECMWF SEAS5 no Copernicus (C3S). A previsão e a anomalia saem da mesma
rodada e da mesma grade. A anomalia é em relação ao normal do próprio modelo.

Precisa de conta gratuita no Copernicus e da chave (API token):
  1. https://cds.climate.copernicus.eu  -> criar conta e entrar
  2. aceitar as licenças dos dois conjuntos (aba "Download", fim da página):
     seasonal-monthly-single-levels e seasonal-postprocessed-single-levels
  3. copiar o "API Token" da página do seu perfil
  GitHub: segredo CDSAPI_KEY. Computador: arquivo cds_chave.txt ao lado deste script.

Uso típico:
  python gerar_previsao_mensal.py --kml-estados estados.kml --saida previsao_mensal
"""
import argparse
import calendar
import datetime as dt
import json
import os
import re
import shutil
import sys
import time

import numpy as np

import gerar_previsao_regiao as g

CDS_URL = "https://cds.climate.copernicus.eu/api"
SISTEMA_PADRAO = "51"          # ECMWF SEAS5 no Copernicus
DIA_PUBLICACAO = 13            # o C3S publica a rodada do mês por volta do dia 13
AREA_CDS = [8, -76, -36, -32]  # N, O, S, L: Brasil com folga
MESES = ["janeiro", "fevereiro", "março", "abril", "maio", "junho", "julho", "agosto",
         "setembro", "outubro", "novembro", "dezembro"]
MESES_CURTOS = ["jan", "fev", "mar", "abr", "mai", "jun", "jul", "ago", "set", "out", "nov", "dez"]

# ------------------------------------------------------------------ escalas
FAIXAS_CHUVA_MES = [  # mm no mês
    (0,   25,  "#ffffff", "#bdbdbd"),
    (25,  50,  "#c7e9c0", "#41ab5d"),
    (50,  100, "#9ecae1", "#2171b5"),
    (100, 200, "#fff23f", "#f07a00"),
    (200, 300, "#ff4a1a", "#7a0000"),
    (300, 400, "#8a5a3c", "#caa07a"),
    (400, 600, "#c8bce0", "#7b4fb0"),
]
FAIXAS_CHUVA_TRI = [(a * 3, b * 3, c1, c2) for a, b, c1, c2 in FAIXAS_CHUVA_MES]  # mm no trimestre
CHUVA_ACIMA = "#e83ce8"
FAIXAS_ANOM_CHUVA_MES = [  # mm abaixo (marrom) / acima (verde-azulado) do normal
    (-200, -100, "#8c510a", "#bf812d"),
    (-100, -50,  "#bf812d", "#dfc27d"),
    (-50,  -10,  "#dfc27d", "#f6e8c3"),
    (-10,  10,   "#f5f5f5", "#f5f5f5"),
    (10,   50,   "#c7eae5", "#80cdc1"),
    (50,   100,  "#80cdc1", "#35978f"),
    (100,  200,  "#35978f", "#01665e"),
]
FAIXAS_ANOM_CHUVA_TRI = [(a * 2, b * 2, c1, c2) for a, b, c1, c2 in FAIXAS_ANOM_CHUVA_MES]
ANOM_CHUVA_ABAIXO, ANOM_CHUVA_ACIMA = "#543005", "#003c30"
FAIXAS_ANOM_TEMP = [  # °C abaixo (azul) / acima (vermelho) do normal
    (-4,    -2,    "#2166ac", "#4393c3"),
    (-2,    -1,    "#4393c3", "#92c5de"),
    (-1,    -0.25, "#92c5de", "#d1e5f0"),
    (-0.25, 0.25,  "#f7f7f7", "#f7f7f7"),
    (0.25,  1,     "#fddbc7", "#f4a582"),
    (1,     2,     "#f4a582", "#d6604d"),
    (2,     4,     "#d6604d", "#b2182b"),
]
ANOM_TEMP_ABAIXO, ANOM_TEMP_ACIMA = "#053061", "#67001f"

# produto -> (título, faixas do mês, faixas do trimestre, cor acima, cor abaixo, extend)
PRODUTOS = {
    "chuva": ("Chuva prevista (mm)", FAIXAS_CHUVA_MES, FAIXAS_CHUVA_TRI, CHUVA_ACIMA, None, "max"),
    "temp": ("Temperatura média prevista (°C)", g.FAIXAS_TEMP, g.FAIXAS_TEMP, g.TEMP_ACIMA, g.TEMP_ABAIXO, "both"),
    "anom_chuva": ("Anomalia de chuva prevista (mm)", FAIXAS_ANOM_CHUVA_MES, FAIXAS_ANOM_CHUVA_TRI,
                   ANOM_CHUVA_ACIMA, ANOM_CHUVA_ABAIXO, "both"),
    "anom_temp": ("Anomalia de temperatura prevista (°C)", FAIXAS_ANOM_TEMP, FAIXAS_ANOM_TEMP,
                  ANOM_TEMP_ACIMA, ANOM_TEMP_ABAIXO, "both"),
}

# conjunto do Copernicus -> (variáveis pedidas, tipo de produto)
PEDIDOS = {
    "seasonal-monthly-single-levels": (["total_precipitation", "2m_temperature"], "monthly_mean"),
    "seasonal-postprocessed-single-levels": (["total_precipitation_anomalous_rate_of_accumulation",
                                              "2m_temperature_anomaly"], "ensemble_mean"),
}


class SemDados(RuntimeError):
    """A rodada pedida ainda não foi publicada no Copernicus."""


# ------------------------------------------------------------------ datas
def somar_meses(d, n):
    m = d.month - 1 + n
    return dt.date(d.year + m // 12, m % 12 + 1, 1)


def rotulo_mes(d):
    return f"{MESES[d.month - 1]}/{d.year}"


def rotulo_trimestre(meses):
    a, b = meses[0], meses[-1]
    ano = f"/{a.year}" if a.year == b.year else f"/{a.year}–{b.year}"
    return f"{MESES_CURTOS[a.month - 1]} a {MESES_CURTOS[b.month - 1]}{ano}"


def rodadas_candidatas(hoje, primeiro_mes):
    """A rodada do mês só existe a partir do dia ~13; antes, usa a do mês anterior."""
    atual = dt.date(hoje.year, hoje.month, 1)
    anterior = somar_meses(atual, -1)
    if primeiro_mes < atual:  # pedido de meses passados: rodada daquele mês
        return [primeiro_mes]
    return [atual, anterior] if hoje.day >= DIA_PUBLICACAO else [anterior]


# ------------------------------------------------------------------ Copernicus
def chave_cds(pasta):
    chave = os.environ.get("CDSAPI_KEY", "").strip()
    if not chave:
        for local in (os.path.join(pasta, "cds_chave.txt"), "cds_chave.txt"):
            if os.path.isfile(local):
                chave = open(local, encoding="utf-8").read().strip()
                break
    return chave or None


def explicar_erro_cds(e):
    msg = str(e)
    baixo = msg.lower()
    if "401" in msg or "authenticat" in baixo or "invalid api key" in baixo or "unauthor" in baixo:
        return ("a chave do Copernicus não foi aceita. Confira o API Token na página do seu perfil em "
                "cds.climate.copernicus.eu e copie de novo para o segredo CDSAPI_KEY ou o cds_chave.txt.")
    if "licen" in baixo or "terms" in baixo:
        return ("falta aceitar a licença do conjunto no site do Copernicus. Abra os dois conjuntos "
                "(seasonal-monthly-single-levels e seasonal-postprocessed-single-levels), vá na aba "
                "Download, role até o fim e aceite os termos.")
    return msg[:500]


def baixar_cds(cliente, conjunto, rodada, lead, sistema, destino):
    variaveis, produto = PEDIDOS[conjunto]
    pedido = {"originating_centre": "ecmwf", "system": sistema, "variable": variaveis,
              "product_type": [produto], "year": [str(rodada.year)], "month": [f"{rodada.month:02d}"],
              "leadtime_month": [str(lead)], "area": AREA_CDS, "data_format": "grib"}
    temporario = destino + ".baixando"
    for tentativa in range(3):
        try:
            cliente.retrieve(conjunto, pedido, temporario)
            os.replace(temporario, destino)
            return
        except Exception as e:  # noqa: BLE001 - o cdsapi usa exceções genéricas
            baixo = str(e).lower()
            if "no data" in baixo or "not available" in baixo or "no matching" in baixo or "404" in baixo:
                raise SemDados(str(e)[:300]) from e
            if "401" in baixo or "licen" in baixo or "terms" in baixo or "authenticat" in baixo or tentativa == 2:
                raise RuntimeError(explicar_erro_cds(e)) from e
            print(f"    (Copernicus: {str(e)[:160]}; nova tentativa em {30 * (tentativa + 1)} s)")
            time.sleep(30 * (tentativa + 1))


def ler_grib(caminho):
    """Lê um GRIB do Copernicus e devolve (lons, lats, {nome_curto: array 2D}).

    Cada arquivo tem um só mês; qualquer dimensão além de lat/lon (membros,
    rodada, passo) vira média.
    """
    import warnings
    import cfgrib
    warnings.filterwarnings("ignore", category=FutureWarning)  # aviso interno do cfgrib/xarray
    saida, lons, lats = {}, None, None
    for ds in cfgrib.open_datasets(caminho, backend_kwargs={"indexpath": ""}):
        for nome, da in ds.data_vars.items():
            lat_n = next((c for c in ("latitude", "lat") if c in da.dims), None)
            lon_n = next((c for c in ("longitude", "lon") if c in da.dims), None)
            if not lat_n or not lon_n:
                continue
            extras = [d for d in da.dims if d not in (lat_n, lon_n)]
            campo = da.mean(extras) if extras else da
            la = np.asarray(ds[lat_n].values, dtype=float)
            lo = np.asarray(ds[lon_n].values, dtype=float)
            arr = np.asarray(campo.transpose(lat_n, lon_n).values, dtype=np.float64)
            lo = np.where(lo > 180, lo - 360, lo)
            ordem = np.argsort(lo)
            lo, arr = lo[ordem], arr[:, ordem]
            curto = str(da.attrs.get("GRIB_shortName") or nome)
            saida[curto] = {"dados": arr, "unidade": str(da.attrs.get("GRIB_units", "")),
                            "nome": str(da.attrs.get("GRIB_name", nome))}
            lons, lats = lo, la
    if not saida:
        raise ValueError(f"nenhum campo lido de {os.path.basename(caminho)}")
    return lons, lats, saida


def _escolher(campos, candidatos, palavra):
    for c in candidatos:
        if c in campos:
            return campos[c]
    for c, v in campos.items():
        if palavra in v["nome"].lower():
            return v
    raise ValueError(f"não achei {palavra} entre {', '.join(campos)}")


def para_mm_no_mes(campo, mes):
    """Taxa (m/s) ou total (m) do mês -> mm no mês."""
    unidade = campo["unidade"].replace(" ", "")
    dados = campo["dados"]
    if "s**-1" in unidade or "s-1" in unidade or "/s" in unidade:
        return dados * 1000 * 86400 * calendar.monthrange(mes.year, mes.month)[1]
    return dados * 1000


def para_celsius(campo, anomalia=False):
    dados = campo["dados"]
    if not anomalia and np.nanmean(dados) > 150:  # veio em kelvin
        return dados - 273.15
    return dados


# ------------------------------------------------------------------ áreas
def _ler_opcional(caminho, leitor):
    if not os.path.isfile(caminho) and not os.path.isabs(caminho):
        caminho = os.path.join(os.path.dirname(os.path.abspath(__file__)), caminho)
    try:
        return leitor(caminho) if os.path.isfile(caminho) else []
    except (ValueError, OSError) as e:
        print(f"  AVISO: não consegui ler {caminho}: {e}")
        return []


def montar_alvos(args, fundo, mesos):
    alvos = []
    cidades = _ler_opcional(args.cidades, g.ler_cidades)          # uma por mesorregião
    municipios = _ler_opcional(args.municipios, g.ler_municipios)  # para achar as capitais
    print(f"Cidades: {len(cidades)} principais, {len(municipios)} municípios")
    caixas = [g.bbox_enquadramento(r) for r in fundo]
    lo0, la0 = min(c[0] for c in caixas), min(c[1] for c in caixas)
    lo1, la1 = max(c[2] for c in caixas), max(c[3] for c in caixas)
    alvos.append(("brasil", None, (lo0 - 0.5, lo1 + 0.5, la0 - 0.5, la1 + 0.5), []))
    por_uf = {g.uf_sigla(r, fundo): r for r in fundo}
    m = args.margem
    for nome_reg, ufs in g.REGIOES_BRASIL.items():
        presentes = [u for u in ufs if u in por_uf]
        if not presentes:
            continue
        cx = [g.bbox_enquadramento(por_uf[u]) for u in presentes]
        reg = {"nome": nome_reg, "tipo": "regiao", "ufs": presentes, "campos": {}, "chaves": [],
               "arquivo": por_uf[presentes[0]]["arquivo"], "nome_mapa": f"Região {nome_reg}",
               "poligonos": [p for u in presentes for p in por_uf[u]["poligonos"]],
               "rotulos": [(u, *g.ponto_para_rotulo(por_uf[u])) for u in presentes]}
        alvos.append((os.path.join("regioes", g._pasta_segura(nome_reg)), reg,
                      (min(c[0] for c in cx) - m, max(c[2] for c in cx) + m,
                       min(c[1] for c in cx) - m, max(c[3] for c in cx) + m), g.capitais_da_regiao(presentes, municipios)))
    for r in sorted(fundo, key=lambda r: g.UF_NOMES.get(g.uf_sigla(r, fundo), r["nome"])):
        b = g.bbox_enquadramento(r)
        alvos.append((g.subdir_do_alvo(r, fundo), r, (b[0] - m, b[2] + m, b[1] - m, b[3] + m),
                      g.cidades_do_estado(r, cidades, municipios, fundo)))
    if args.grupos:  # grupos de pontos (ex.: AMAGGI), só o mapa do estado
        for gr in g.ler_grupos(args.grupos).values():
            estado = por_uf.get(gr["uf"])
            if estado is None:
                print(f"  AVISO: grupo {gr['nome']}: estado {gr['uf']} não está no KML; pulando.")
                continue
            logo_grupo = None
            if gr["logo"]:
                caminho = gr["logo"] if os.path.isfile(gr["logo"]) else os.path.join(
                    os.path.dirname(os.path.abspath(args.grupos)), gr["logo"])
                if os.path.isfile(caminho):
                    logo_grupo = (caminho, gr["logo_pos"])
            contornos = [r for r in mesos if g.uf_sigla(r, fundo) == gr["uf"]] if gr["mesorregioes"] else []
            reg = dict(estado, nome=gr["nome"], tipo="grupo", grupo=gr["nome"],
                       campos=dict(estado["campos"], sigla=gr["uf"]), pontos=gr["pontos"],
                       contornos=contornos, rotulos_contornos=[], logo_grupo=logo_grupo,
                       nome_mapa=f"{gr['nome']} — {g.UF_NOMES[gr['uf']]}")
            b = g.bbox_enquadramento(estado)
            alvos.append((g._pasta_segura(gr["nome"]), reg, (b[0] - m, b[2] + m, b[1] - m, b[3] + m), []))
    return alvos


def alvo_curto(regiao, fundo):
    p = g.prefixo_png(regiao, fundo, "x", 0)
    return p[len("ecmwf_"):-len("_x_0d")]


def nome_arquivo(regiao, fundo, produto, periodo):
    return f"sazonal_{alvo_curto(regiao, fundo)}_{produto}_{periodo}.png"


# ------------------------------------------------------------------ geração em paralelo
_TRABALHO = {}


def _iniciar(estado):
    global _TRABALHO
    import matplotlib
    matplotlib.use("Agg")
    _TRABALHO = estado


def _figura(tarefa):
    i_alvo, chave = tarefa
    e = _TRABALHO
    subdir, regiao, extent, cids = e["alvos"][i_alvo]
    produto, periodo, rotulo, dados, trimestre = e["campos"][chave]
    titulo, f_mes, f_tri, acima, abaixo, extend = PRODUTOS[produto]
    arquivo = nome_arquivo(regiao, e["fundo"], produto, periodo)
    g.plotar(e["lons"], e["lats"], dados, titulo, f"{rotulo} · tendência",
             os.path.join(e["saida"], subdir, arquivo), f_tri if trimestre else f_mes,
             cor_acima=acima, cor_abaixo=abaixo, extend=extend, extent=extent, regiao=regiao,
             fundo=e["fundo"], cidades=cids, logo=e["logo"])
    return os.path.join(e["saida"], subdir), arquivo


# ------------------------------------------------------------------ principal
def main():
    ap = argparse.ArgumentParser(description="Previsão dos próximos 3 meses e anomalia prevista (ECMWF SEAS5 via Copernicus)")
    ap.add_argument("--kml-estados", default="estados.kml", help="KML dos estados")
    ap.add_argument("--kml-meso", default="mesorregioes.kml", help="KML das mesorregiões (contorno dos grupos)")
    ap.add_argument("--saida", default="previsao_mensal", help="pasta de saída")
    ap.add_argument("--cache", default=".cache_c3s", help="pasta dos arquivos baixados do Copernicus")
    ap.add_argument("--logo", default=None, help="logo principal (ex.: logo.png)")
    ap.add_argument("--grupos", default=None, help="grupos.json para incluir grupos (ex.: AMAGGI)")
    ap.add_argument("--cidades", default=g.CIDADES_PADRAO, help="CSV das cidades principais (mapas de estado)")
    ap.add_argument("--municipios", default=g.MUNICIPIOS_PADRAO, help="CSV das sedes dos municípios (capitais)")
    ap.add_argument("--mes-inicio", default=None, help="primeiro dos 3 meses (AAAA-MM). Padrão: mês atual")
    ap.add_argument("--sistema", default=SISTEMA_PADRAO, help=f"sistema do ECMWF no Copernicus (padrão {SISTEMA_PADRAO})")
    ap.add_argument("--sem-anomalia", action="store_true", help="só a previsão, sem os mapas de anomalia")
    ap.add_argument("--margem", type=float, default=1.0, help="margem em graus em volta de cada área")
    ap.add_argument("--processos", type=int, default=0, help="mapas ao mesmo tempo (0 = automático)")
    ap.add_argument("--somente-estrutura", action="store_true", help="só confere arquivos e pastas")
    args = ap.parse_args()

    hoje = dt.datetime.now(g.BRASILIA).date()
    if args.mes_inicio:
        try:
            a, mm = (int(x) for x in re.split(r"[-/]", args.mes_inicio.strip()))
            primeiro = dt.date(a, mm, 1)
        except ValueError:
            sys.exit(f"ERRO: --mes-inicio deve ser AAAA-MM (ex.: 2026-10), veio '{args.mes_inicio}'")
    else:
        primeiro = dt.date(hoje.year, hoje.month, 1)
    meses = [somar_meses(primeiro, k) for k in range(3)]
    print(f"Meses: {', '.join(rotulo_mes(m_) for m_ in meses)} | hoje (Brasília): {hoje.isoformat()}")

    try:
        _, fundo, _ = g.carregar_geografia([args.kml_estados], [args.kml_estados], [])
    except (ValueError, OSError) as e:
        sys.exit(f"ERRO no KML de estados: {e}")
    mesos = []
    if args.grupos and os.path.isfile(args.kml_meso):
        _, _, mesos = g.carregar_geografia([args.kml_meso], [args.kml_estados], [args.kml_meso])
    try:
        alvos = montar_alvos(args, fundo, mesos)
    except (ValueError, OSError) as e:
        sys.exit(f"ERRO nos grupos: {e}")
    alvos = g.preparar_pastas(alvos, args.saida, fundo)
    por_tipo = {}
    for _, r, _, _ in alvos:
        t = "brasil" if r is None else r["tipo"]
        por_tipo[t] = por_tipo.get(t, 0) + 1
    print("Áreas: " + ", ".join(f"{n} {t}" for t, n in por_tipo.items()) + f" (total {len(alvos)})")
    if args.somente_estrutura:
        print("Somente estrutura: nenhum download ou mapa foi gerado.")
        return

    chave = chave_cds(os.path.dirname(os.path.abspath(__file__)))
    if not chave:
        sys.exit("ERRO: falta a chave do Copernicus. No GitHub, crie o segredo CDSAPI_KEY; no computador, "
                 "ponha o API Token num arquivo cds_chave.txt nesta pasta. Veja o começo deste script.")
    import cdsapi
    cliente = cdsapi.Client(url=CDS_URL, key=chave, quiet=True, progress=False)

    # Rodada mais recente publicada: a do mês (a partir do dia ~13) ou a do mês anterior.
    conjuntos = list(PEDIDOS) if not args.sem_anomalia else ["seasonal-monthly-single-levels"]
    arquivos, rodada = {}, None
    for candidata in rodadas_candidatas(hoje, primeiro):
        leads = [(m_.year - candidata.year) * 12 + m_.month - candidata.month + 1 for m_ in meses]
        if min(leads) < 1 or max(leads) > 6:
            continue
        pasta = os.path.join(args.cache, candidata.strftime("%Y-%m"), f"sistema{args.sistema}")
        os.makedirs(pasta, exist_ok=True)
        print(f"Rodada do Copernicus de {rotulo_mes(candidata)} (meses à frente: {', '.join(map(str, leads))})")
        try:
            for conjunto in conjuntos:
                for mes, lead in zip(meses, leads):
                    destino = os.path.join(pasta, f"{conjunto}_{lead}.grib")
                    if not os.path.isfile(destino):
                        print(f"  baixando {conjunto}, {rotulo_mes(mes)} ...")
                        baixar_cds(cliente, conjunto, candidata, lead, args.sistema, destino)
                    arquivos[(conjunto, mes)] = destino
            rodada = candidata
            break
        except SemDados as e:
            print(f"  ainda não publicada ({str(e)[:120]}); tentando a rodada anterior.")
            arquivos = {}
        except RuntimeError as e:
            sys.exit(f"ERRO no Copernicus: {e}")
    if rodada is None:
        sys.exit("ERRO: não encontrei no Copernicus uma rodada que cubra esses meses. O C3S publica por volta "
                 f"do dia {DIA_PUBLICACAO}; se o sistema do ECMWF mudou, ajuste --sistema.")

    # Campos: mm no mês, °C, e as anomalias; o trimestre soma a chuva e faz a média da temperatura.
    campos, lons, lats = {}, None, None
    acumulado = {"chuva": [], "temp": [], "anom_chuva": [], "anom_temp": []}
    for mes in meses:
        periodo, rotulo = mes.strftime("%Y-%m"), rotulo_mes(mes)
        lons, lats, f = ler_grib(arquivos[("seasonal-monthly-single-levels", mes)])
        valores = {"chuva": para_mm_no_mes(_escolher(f, ["tprate", "tp", "mtpr"], "precipitation"), mes),
                   "temp": para_celsius(_escolher(f, ["2t", "t2m"], "temperature"))}
        if not args.sem_anomalia:
            _, _, fa = ler_grib(arquivos[("seasonal-postprocessed-single-levels", mes)])
            valores["anom_chuva"] = para_mm_no_mes(_escolher(fa, ["tpara", "tpa"], "precipitation"), mes)
            valores["anom_temp"] = para_celsius(_escolher(fa, ["2ta", "t2a"], "temperature"), anomalia=True)
        for produto, dados in valores.items():
            campos[f"{produto}|{periodo}"] = (produto, periodo, rotulo, dados.astype(np.float32), False)
            acumulado[produto].append(dados)
    tri_periodo = f"{meses[0]:%Y-%m}_a_{meses[-1]:%Y-%m}"
    for produto, lista in acumulado.items():
        if lista:
            soma = produto in ("chuva", "anom_chuva")
            dados = np.sum(lista, axis=0) if soma else np.mean(lista, axis=0)
            campos[f"{produto}|tri"] = (produto, f"tri_{tri_periodo}", rotulo_trimestre(meses),
                                        dados.astype(np.float32), True)

    print(f"Grade: {len(lons)} x {len(lats)} pontos, {abs(lons[1] - lons[0]):g}° | rodada {rotulo_mes(rodada)}")
    logo = args.logo if args.logo and os.path.isfile(args.logo) else None
    tarefas = [(i, k) for i in range(len(alvos)) for k in campos]
    estado = {"alvos": alvos, "fundo": fundo, "campos": campos, "lons": lons, "lats": lats,
              "saida": args.saida, "logo": logo}
    n = g.numero_processos(args.processos, len(tarefas))
    print(f"A gerar {len(tarefas)} mapa(s): {len(alvos)} áreas × {len(campos)} mapas | {n} ao mesmo tempo")
    inicio, gerados = time.time(), []
    if n == 1:
        _iniciar(estado)
        gerados = [_figura(t) for t in tarefas]
    else:
        from concurrent.futures import ProcessPoolExecutor
        try:
            with ProcessPoolExecutor(max_workers=n, initializer=_iniciar, initargs=(estado,)) as pool:
                for k, r in enumerate(pool.map(_figura, tarefas, chunksize=4), 1):
                    gerados.append(r)
                    if k % 100 == 0:
                        print(f"    {k}/{len(tarefas)} mapas | {time.time() - inicio:.0f}s")
        except Exception as e:  # noqa: BLE001
            sys.exit(f"ERRO ao gerar os mapas: {type(e).__name__}: {e}")
    print(f"{len(gerados)} mapas em {time.time() - inicio:.0f}s")

    # Tira os mapas de meses anteriores das mesmas áreas.
    novos = {}
    for pasta, arq in gerados:
        novos.setdefault(pasta, set()).add(arq)
    removidos = 0
    for pasta, arqs in novos.items():
        for arq in os.listdir(pasta):
            if arq.startswith("sazonal_") and arq.endswith(".png") and arq not in arqs:
                os.remove(os.path.join(pasta, arq))
                removidos += 1
    if removidos:
        print(f"Removidos {removidos} mapas de meses anteriores.")
    resumo = {"fonte": f"ECMWF SEAS5 (sistema {args.sistema}) via Copernicus C3S", "rodada": rodada.strftime("%Y-%m"),
              "meses": [m_.strftime("%Y-%m") for m_ in meses], "anomalia": not args.sem_anomalia,
              "gerado_em": dt.datetime.now(g.BRASILIA).isoformat(timespec="seconds"),
              "areas": {os.path.relpath(p, args.saida).replace(os.sep, "/"): sorted(a) for p, a in novos.items()}}
    with open(os.path.join(args.saida, "previsao_mensal.json"), "w", encoding="utf-8") as fp:
        json.dump(resumo, fp, ensure_ascii=False, indent=1)
    print(f"Pronto: {args.saida}")


if __name__ == "__main__":
    main()
