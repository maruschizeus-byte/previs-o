#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gera figuras de previsão do ECMWF (Open Data) no estilo do painel "Total (mm)"
— colorbar à direita, título e período —, FOCANDO a imagem em um ESTADO ou
MESORREGIÃO lido de arquivo(s) KML.

Variáveis (escolha com --vars):
  chuva : precipitação — gera ACUMULADO (desde a rodada) e do DIA (desacumulado)
  tmin  : temperatura MÍNIMA do dia a 2 m (°C)   — útil p/ geada
  tmax  : temperatura MÁXIMA do dia a 2 m (°C)
  nuvem : cobertura total de nuvens (%)

Estratégia: para cada horizonte (dia), o dado do ECMWF é baixado UMA vez e
depois recortado para TODAS as regiões pedidas. Mín e máx de temperatura saem
dos mesmos sub-passos de 2t (baixados uma vez).

USO
    pip install ecmwf-opendata xarray cfgrib numpy matplotlib
    # (sistema) eccodes p/ o cfgrib ler GRIB:  apt-get install libeccodes0

    python gerar_previsao_regiao.py --regioes "SP;MG;3105" \
        --kml estados.kml mesorregioes.kml --vars chuva tmin tmax nuvem
    python gerar_previsao_regiao.py --regiao SP --kml estados.kml --vars chuva
    python gerar_previsao_regiao.py --brasil --kml estados.kml --dias 1 2 3

A busca da região casa (sem acento/caixa) contra o <name> do Placemark e
contra qualquer campo de ExtendedData (SimpleData/Data) — então nome, sigla
ou CÓDIGO IBGE funcionam se estiverem no KML.
"""

import os
import sys
import time
import argparse
import unicodedata
import datetime as dt
import xml.etree.ElementTree as ET

import numpy as np

# =========================================================================
# CONFIGURAÇÕES
# =========================================================================
STEPS_PADRAO = [24, 48, 72, 96, 120, 144, 168]  # 1..7 dias, em horas

# Domínio máximo baixado do ECMWF (América do Sul). O recorte da REGIÃO é só no
# desenho; sempre baixamos o domínio todo para reaproveitar o dado entre regiões.
LON_MIN, LON_MAX = -82.0, -30.0
LAT_MIN, LAT_MAX = -60.0, 15.0

MESES_PT = ["jan", "fev", "mar", "abr", "mai", "jun",
            "jul", "ago", "set", "out", "nov", "dez"]

SUBNIVEIS = 6  # sub-faixas por faixa de cor (suaviza o degradê)

# ---- Paletas por variável: (limite_inf, limite_sup, cor_inicial, cor_final).
#      Dentro da faixa o degradê é linear; a borda (5, 10, 25...) é o "salto".

# Chuva (mm)
FAIXAS_CHUVA = [
    (0,   5,   "#ffffff", "#8f8f8f"),
    (5,   10,  "#7be07b", "#238b3a"),
    (10,  25,  "#63b1ff", "#123fb0"),
    (25,  50,  "#fff23f", "#f07a00"),
    (50,  75,  "#ff4a1a", "#7a0000"),
    (75,  100, "#8a5a3c", "#caa07a"),
    (100, 150, "#c8bce0", "#8f7fc0"),
    (150, 200, "#a020b0", "#e83ce8"),
]
CHUVA_ACIMA = "#f3ddf3"

# Temperatura (°C) — frio (roxo/azul) -> calor (laranja/vermelho)
FAIXAS_TEMP = [
    (-5,  0,  "#5e4fa2", "#3288bd"),
    (0,   5,  "#3288bd", "#66c2a5"),
    (5,   10, "#66c2a5", "#abdda4"),
    (10,  15, "#abdda4", "#e6f598"),
    (15,  20, "#e6f598", "#fee08b"),
    (20,  25, "#fee08b", "#fdae61"),
    (25,  30, "#fdae61", "#f46d43"),
    (30,  35, "#f46d43", "#d53e4f"),
    (35,  40, "#d53e4f", "#9e0142"),
]
TEMP_ACIMA = "#5a0028"
TEMP_ABAIXO = "#40004b"

# Cobertura de nuvens (%) — céu limpo (branco) -> encoberto (cinza-azulado)
FAIXAS_NUVEM = [
    (0,   20,  "#ffffff", "#cfdae6"),
    (20,  40,  "#cfdae6", "#9fb2c6"),
    (40,  60,  "#9fb2c6", "#7288a0"),
    (60,  80,  "#7288a0", "#4c6076"),
    (80,  100, "#4c6076", "#2b3a4a"),
]


# =========================================================================
# PALETA / COLORMAP
# =========================================================================
def _hex_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))


def construir_colormap(faixas, cor_acima=None, cor_abaixo=None, subniveis=SUBNIVEIS):
    """(levels, cmap, norm, ticks) a partir das faixas. Cada faixa vira
    'subniveis' degraus com cor interpolada; os rótulos são as bordas das
    faixas (igualmente espaçados no colorbar)."""
    from matplotlib.colors import ListedColormap, BoundaryNorm
    levels, cores = [], []
    for (a, b, c0, c1) in faixas:
        r0, g0, bl0 = _hex_rgb(c0)
        r1, g1, bl1 = _hex_rgb(c1)
        for k in range(subniveis):
            lv0 = a + (b - a) * (k / subniveis)
            if not levels or abs(levels[-1] - lv0) > 1e-9:
                levels.append(lv0)
            fm = (k + 0.5) / subniveis
            cores.append((r0 + (r1 - r0) * fm,
                          g0 + (g1 - g0) * fm,
                          bl0 + (bl1 - bl0) * fm))
        levels.append(b)
    levels = np.array(sorted(set(levels)), dtype="float64")
    cmap = ListedColormap(cores)
    if cor_acima:
        cmap.set_over(_hex_rgb(cor_acima))
    if cor_abaixo:
        cmap.set_under(_hex_rgb(cor_abaixo))
    norm = BoundaryNorm(levels, cmap.N)
    ticks = [a for (a, _b, _c0, _c1) in faixas] + [faixas[-1][1]]
    return levels, cmap, norm, ticks


# =========================================================================
# KML: leitura de polígonos e seleção da região
# =========================================================================
def _sem_acento(s):
    s = unicodedata.normalize("NFKD", s or "")
    s = "".join(c for c in s if not unicodedata.combining(c))
    return s.strip().lower()


def _parse_coords(texto):
    pontos = []
    for token in (texto or "").replace("\n", " ").split():
        partes = token.split(",")
        if len(partes) >= 2:
            try:
                pontos.append((float(partes[0]), float(partes[1])))
            except ValueError:
                pass
    return pontos


def ler_kml(caminho):
    """Lista de regiões: {'nome','chaves'(set),'poligonos'(list),'arquivo'}."""
    tree = ET.parse(caminho)
    root = tree.getroot()

    def tag(e):
        return e.tag.split("}")[-1]

    def achar(e, nome):
        return [x for x in e.iter() if tag(x) == nome]

    regioes = []
    for pm in achar(root, "Placemark"):
        nome = ""
        for filho in list(pm):
            if tag(filho) == "name" and filho.text:
                nome = filho.text.strip()
                break
        chaves = set()
        campos = {}
        if nome:
            chaves.add(_sem_acento(nome))
        for sd in achar(pm, "SimpleData"):
            if sd.text:
                chaves.add(_sem_acento(sd.text))
                nm = sd.get("name")
                if nm:
                    campos[_sem_acento(nm)] = sd.text.strip()
        for d in achar(pm, "Data"):
            nm = d.get("name")
            val = None
            for v in d:
                if tag(v) == "value" and v.text:
                    val = v.text.strip()
            if val:
                chaves.add(_sem_acento(val))
                if nm:
                    campos[_sem_acento(nm)] = val

        poligonos = []
        for poly in achar(pm, "Polygon"):
            for ob in achar(poly, "outerBoundaryIs"):
                for coords in achar(ob, "coordinates"):
                    pts = _parse_coords(coords.text)
                    if len(pts) >= 3:
                        poligonos.append(pts)
        if not poligonos:
            for coords in achar(pm, "coordinates"):
                pts = _parse_coords(coords.text)
                if len(pts) >= 3:
                    poligonos.append(pts)

        if poligonos:
            regioes.append({"nome": nome or "(sem nome)", "chaves": chaves,
                            "campos": campos, "poligonos": poligonos,
                            "arquivo": os.path.basename(caminho)})
    return regioes


def ler_kmls(caminhos):
    todas = []
    for c in caminhos:
        lidas = ler_kml(c)
        print(f"KML: {len(lidas)} região(ões) em {c}")
        todas.extend(lidas)
    return todas


def selecionar_regiao(regioes, alvo):
    a = _sem_acento(alvo)
    exatos = [r for r in regioes if a in r["chaves"]]
    if exatos:
        if len(exatos) > 1:
            fontes = ", ".join(f"{r['nome']} ({r['arquivo']})" for r in exatos)
            print(f"  AVISO: '{alvo}' casou com {len(exatos)} regiões: {fontes}. "
                  f"Usando a primeira. Para desambiguar, use o código IBGE.")
        return exatos[0]
    for r in regioes:
        if any(k.startswith(a) or a.startswith(k) for k in r["chaves"] if k):
            return r
    for r in regioes:
        if any(a in k for k in r["chaves"] if k):
            return r
    return None


def bbox_regiao(regiao):
    xs, ys = [], []
    for poly in regiao["poligonos"]:
        for (lon, lat) in poly:
            xs.append(lon); ys.append(lat)
    return min(xs), min(ys), max(xs), max(ys)


def bbox_uniao(regioes):
    """bbox que cobre todas as regiões (usado p/ enquadrar a vista Brasil)."""
    xs, ys = [], []
    for r in regioes:
        for poly in r["poligonos"]:
            for (lon, lat) in poly:
                xs.append(lon); ys.append(lat)
    if not xs:
        return None
    return min(xs), min(ys), max(xs), max(ys)


# ---- UF: código IBGE / nome -> sigla, para montar as pastas por estado ----
UF_COD2SIGLA = {
    11: "RO", 12: "AC", 13: "AM", 14: "RR", 15: "PA", 16: "AP", 17: "TO",
    21: "MA", 22: "PI", 23: "CE", 24: "RN", 25: "PB", 26: "PE", 27: "AL",
    28: "SE", 29: "BA", 31: "MG", 32: "ES", 33: "RJ", 35: "SP",
    41: "PR", 42: "SC", 43: "RS", 50: "MS", 51: "MT", 52: "GO", 53: "DF",
}
UF_NOME2SIGLA = {
    "rondonia": "RO", "acre": "AC", "amazonas": "AM", "roraima": "RR",
    "para": "PA", "amapa": "AP", "tocantins": "TO", "maranhao": "MA",
    "piaui": "PI", "ceara": "CE", "rio grande do norte": "RN", "paraiba": "PB",
    "pernambuco": "PE", "alagoas": "AL", "sergipe": "SE", "bahia": "BA",
    "minas gerais": "MG", "espirito santo": "ES", "rio de janeiro": "RJ",
    "sao paulo": "SP", "parana": "PR", "santa catarina": "SC",
    "rio grande do sul": "RS", "mato grosso do sul": "MS", "mato grosso": "MT",
    "goias": "GO", "distrito federal": "DF",
}


def _centro_bbox(regiao):
    lo0, la0, lo1, la1 = bbox_regiao(regiao)
    return (lo0 + lo1) / 2.0, (la0 + la1) / 2.0


def uf_sigla(regiao, estados=None):
    """Descobre a sigla da UF de uma região, em cascata:
    1) campo de sigla no KML; 2) nome do estado; 3) código IBGE (2 primeiros
    dígitos); 4) geometria: centro dentro de qual estado do fundo."""
    c = regiao.get("campos", {})
    for k in ("sigla_uf", "sigla", "uf", "cd_uf_sigla"):
        v = c.get(k, "").strip()
        if len(v) == 2 and v.isalpha():
            return v.upper()
    for k in ("nm_uf", "nome_uf", "estado", "nm_estado", "name_uf"):
        s = UF_NOME2SIGLA.get(_sem_acento(c.get(k, "")))
        if s:
            return s
    for k in ("cd_uf", "cd_geocuf", "cd_meso", "geocodigo", "codigo",
              "cd_geocme", "cd_mesorregiao", "cd_rgint"):
        v = c.get(k, "").strip()
        if len(v) >= 2 and v[:2].isdigit():
            s = UF_COD2SIGLA.get(int(v[:2]))
            if s:
                return s
    # nome da própria região casa com um estado?
    s = UF_NOME2SIGLA.get(_sem_acento(regiao.get("nome", "")))
    if s:
        return s
    # geometria: centro da região dentro de um estado do fundo
    if estados:
        clon, clat = _centro_bbox(regiao)
        for st in estados:
            if ponto_na_regiao(clon, clat, st):
                return uf_sigla(st) or None
    return None


def _pasta_segura(nome):
    """Nome de pasta legível e seguro (troca / por -, tira pontas ruins)."""
    s = (nome or "").replace("/", "-").replace("\\", "-").strip().strip(".")
    return s or "sem_nome"


def subdir_do_alvo(regiao, estados):
    """Caminho relativo do alvo: 'brasil', ou '<UF>' p/ estado, ou
    '<UF>/<mesorregião>' p/ mesorregião."""
    if regiao is None:
        return "brasil"
    ids_estados = {id(r) for r in (estados or [])}
    if id(regiao) in ids_estados:               # é um estado
        return uf_sigla(regiao, estados) or _pasta_segura(regiao["nome"])
    uf = uf_sigla(regiao, estados) or "SEM_UF"  # é mesorregião
    return os.path.join(uf, _pasta_segura(regiao["nome"]))


# =========================================================================
# CIDADES (pontos de referência)
# =========================================================================
def ler_cidades(caminho):
    """Lê pontos de cidade de um CSV (nome,lat,lon) ou de um KML de pontos.
    Retorna lista [(nome, lat, lon)]."""
    if caminho.lower().endswith(".kml"):
        return _ler_cidades_kml(caminho)
    return _ler_cidades_csv(caminho)


def _ler_cidades_kml(caminho):
    tree = ET.parse(caminho)
    root = tree.getroot()

    def tag(e):
        return e.tag.split("}")[-1]

    def achar(e, nome):
        return [x for x in e.iter() if tag(x) == nome]

    cidades = []
    for pm in achar(root, "Placemark"):
        nome = ""
        for filho in list(pm):
            if tag(filho) == "name" and filho.text:
                nome = filho.text.strip()
                break
        for pt in achar(pm, "Point"):
            for coords in achar(pt, "coordinates"):
                pts = _parse_coords(coords.text)
                if pts:
                    lon, lat = pts[0]
                    cidades.append((nome or "(cidade)", lat, lon))
    return cidades


def _ler_cidades_csv(caminho):
    import csv
    cidades = []
    with open(caminho, encoding="utf-8-sig") as fp:
        amostra = fp.read(2048)
        fp.seek(0)
        delim = ";" if amostra.count(";") > amostra.count(",") else ","
        leitor = csv.reader(fp, delimiter=delim)
        linhas = [ln for ln in leitor if ln]
    if not linhas:
        return cidades

    # detecta cabeçalho e a ordem das colunas
    cab = [c.strip().lower() for c in linhas[0]]
    def _idx(cands):
        for i, c in enumerate(cab):
            if c in cands:
                return i
        return None
    i_nome = _idx({"nome", "name", "cidade", "municipio", "município"})
    i_lat = _idx({"lat", "latitude", "y"})
    i_lon = _idx({"lon", "lng", "long", "longitude", "x"})
    tem_cab = None not in (i_lat, i_lon)
    dados = linhas[1:] if tem_cab else linhas
    if not tem_cab:
        i_nome, i_lat, i_lon = 0, 1, 2  # ordem assumida: nome,lat,lon

    for ln in dados:
        try:
            nome = ln[i_nome].strip() if i_nome is not None and i_nome < len(ln) else "(cidade)"
            lat = float(str(ln[i_lat]).replace(",", "."))
            lon = float(str(ln[i_lon]).replace(",", "."))
            cidades.append((nome, lat, lon))
        except (ValueError, IndexError):
            continue
    return cidades


def _parse_cidade_cli(txt):
    """'Nome,lat,lon' -> (nome, lat, lon). Nome pode ter vírgula: os DOIS
    últimos campos são lat e lon."""
    partes = [p.strip() for p in txt.split(",")]
    if len(partes) < 3:
        raise ValueError(f"--cidade inválido: '{txt}' (use Nome,lat,lon)")
    lon = float(partes[-1].replace(",", "."))
    lat = float(partes[-2].replace(",", "."))
    nome = ",".join(partes[:-2]).strip() or "(cidade)"
    return (nome, lat, lon)


def ponto_na_regiao(lon, lat, regiao):
    """True se (lon,lat) cai dentro de algum polígono da região."""
    from matplotlib.path import Path
    for poly in regiao["poligonos"]:
        if len(poly) >= 3 and Path(poly).contains_point((lon, lat)):
            return True
    return False


# =========================================================================
# ECMWF: download e processamento (genérico por parâmetro)
# =========================================================================
# Fontes do ECMWF Open Data, em ordem de tentativa. Os espelhos em nuvem
# (aws/azure) não têm o limite de 500 conexões do portal principal e evitam o
# erro 429. 'ecmwf' fica por último, como reserva. Ajustável por --fonte.
FONTES = ["aws", "azure", "ecmwf"]


def baixar(param, step, grib_file, data_rodada=None, hora_rodada=0):
    """Baixa um passo tentando cada fonte em ordem; se uma falhar (429, etc.),
    passa para a próxima em vez de insistir no mesmo endpoint congestionado."""
    from ecmwf.opendata import Client
    kw = dict(type="fc", stream="oper", param=param, step=step, target=grib_file)
    if data_rodada is not None:
        kw["date"] = data_rodada.strftime("%Y%m%d")
        kw["time"] = hora_rodada
    ultimo = None
    for fonte in FONTES:
        try:
            Client(source=fonte).retrieve(**kw)
            return
        except Exception as e:
            ultimo = e
            print(f"    (fonte '{fonte}' falhou: {e}; tentando a próxima)")
    raise ultimo if ultimo else RuntimeError("nenhuma fonte disponível")


def ler_grib(grib_file, param, fator=1.0, offset=0.0):
    """Lê o GRIB, ajusta longitude p/ -180..180, recorta o domínio e converte
    (arr*fator+offset). Retorna (lons, lats, arr) com norte no topo."""
    import xarray as xr
    ds = xr.open_dataset(grib_file, engine="cfgrib")
    da = ds[param] if param in ds else ds[list(ds.data_vars)[0]]
    if float(da.longitude.max()) > 180:
        da = da.assign_coords(
            longitude=(((da.longitude + 180) % 360) - 180)
        ).sortby("longitude")
    rec = da.sel(latitude=slice(LAT_MAX, LAT_MIN),
                 longitude=slice(LON_MIN, LON_MAX))
    lats = np.asarray(rec.latitude.values, dtype="float64")
    lons = np.asarray(rec.longitude.values, dtype="float64")
    arr = np.asarray(rec.values, dtype="float32") * fator + offset
    if lats[0] < lats[-1]:
        lats = lats[::-1]
        arr = arr[::-1, :]
    return lons, lats, arr


def _valid_utc(grib_file, param):
    try:
        import xarray as xr
        ds = xr.open_dataset(grib_file, engine="cfgrib")
        da = ds[param] if param in ds else ds[list(ds.data_vars)[0]]
        v = da.coords.get("valid_time")
        if v is None:
            return None
        val = v.values.reshape(-1)[0] if getattr(v.values, "ndim", 0) else v.values
        return np.datetime64(val).astype("datetime64[s]").astype(dt.datetime)
    except Exception:
        return None


def _remover(*caminhos):
    for base in caminhos:
        for f in (base, base + ".idx"):
            if os.path.exists(f):
                try:
                    os.remove(f)
                except OSError:
                    pass


# ---- cache em disco: evita rebaixar o mesmo dado no mesmo dia ----
def _dt_iso(v):
    return v.strftime("%Y-%m-%dT%H:%M:%S") if v is not None else ""


def _iso_dt(s):
    s = str(s)
    if not s:
        return None
    try:
        return dt.datetime.fromisoformat(s)
    except ValueError:
        return None


def _cache_path(cache_dir, var, dia, step):
    return os.path.join(cache_dir, f"{var}_{dia}_s{step}.npz")


def _cache_load(cache_dir, var, dia, step):
    if not cache_dir:
        return None
    cp = _cache_path(cache_dir, var, dia, step)
    if os.path.exists(cp):
        try:
            return dict(np.load(cp, allow_pickle=False))
        except Exception:
            return None
    return None


def _cache_save(cache_dir, var, dia, step, **arrays):
    if not cache_dir:
        return
    try:
        os.makedirs(cache_dir, exist_ok=True)
        np.savez(_cache_path(cache_dir, var, dia, step), **arrays)
    except Exception as e:
        print(f"    (aviso: não consegui gravar cache {var} {step}h: {e})")


def obter_chuva(step, tmp_prefix, data_rodada=None, hora_rodada=0,
                cache_dir=None, dia=None):
    """(lons, lats, acumulado_mm, diario_mm, valido). Acumulado = tp no passo;
    diário = tp(step) - tp(step-24); no dia 1 o anterior é 0."""
    c = _cache_load(cache_dir, "chuva", dia, step)
    if c is not None:
        print(f"    (cache: chuva {step}h)")
        return c["lons"], c["lats"], c["acum"], c["diario"], _iso_dt(c["valido"])
    g = f"{tmp_prefix}_tp_{step}h.grib2"
    baixar("tp", step, g, data_rodada, hora_rodada)
    lons, lats, acum = ler_grib(g, "tp", 1000.0, 0.0)  # m -> mm
    valido = _valid_utc(g, "tp")
    if step > 24:
        g2 = f"{tmp_prefix}_tp_{step-24}h.grib2"
        baixar("tp", step - 24, g2, data_rodada, hora_rodada)
        _, _, acum_ant = ler_grib(g2, "tp", 1000.0, 0.0)
        diario = np.clip(acum - acum_ant, 0, None)
        _remover(g2)
    else:
        diario = np.clip(acum, 0, None)
    _remover(g)
    _cache_save(cache_dir, "chuva", dia, step, lons=lons, lats=lats,
                acum=acum, diario=diario, valido=_dt_iso(valido))
    return lons, lats, acum, diario, valido


def obter_nuvem(step, tmp_prefix, data_rodada=None, hora_rodada=0,
                cache_dir=None, dia=None):
    """(lons, lats, nuvem_%, valido). tcc é fração instantânea 0..1."""
    c = _cache_load(cache_dir, "nuvem", dia, step)
    if c is not None:
        print(f"    (cache: nuvem {step}h)")
        return c["lons"], c["lats"], c["nuvem"], _iso_dt(c["valido"])
    g = f"{tmp_prefix}_tcc_{step}h.grib2"
    baixar("tcc", step, g, data_rodada, hora_rodada)
    lons, lats, arr = ler_grib(g, "tcc", 100.0, 0.0)
    valido = _valid_utc(g, "tcc")
    _remover(g)
    arr = np.clip(arr, 0, 100)
    _cache_save(cache_dir, "nuvem", dia, step, lons=lons, lats=lats,
                nuvem=arr, valido=_dt_iso(valido))
    return lons, lats, arr, valido


def obter_temp(step, tmp_prefix, data_rodada=None, hora_rodada=0,
               cache_dir=None, dia=None):
    """(lons, lats, tmin_C, tmax_C, valido). Mín e máx do dia a partir dos
    sub-passos de 2t (3 em 3 h até 144 h, 6 em 6 h depois). Baixa cada
    sub-passo UMA vez e atualiza mín e máx juntos."""
    c = _cache_load(cache_dir, "temp", dia, step)
    if c is not None:
        print(f"    (cache: temperatura {step}h)")
        return c["lons"], c["lats"], c["tmin"], c["tmax"], _iso_dt(c["valido"])
    passo = 3 if step <= 144 else 6
    ini = step - 24 + passo
    sub_steps = list(range(ini, step + 1, passo))
    if step not in sub_steps:
        sub_steps.append(step)
    lons = lats = tmin = tmax = valido = None
    for s in sub_steps:
        if s <= 0:
            continue
        g = f"{tmp_prefix}_2t_{s}h.grib2"
        try:
            baixar("2t", s, g, data_rodada, hora_rodada)
            lo, la, arr = ler_grib(g, "2t", 1.0, -273.15)  # K -> °C
            lons, lats = lo, la
            if s == step:
                valido = _valid_utc(g, "2t")
            tmin = arr if tmin is None else np.fmin(tmin, arr)
            tmax = arr if tmax is None else np.fmax(tmax, arr)
        except Exception as e:
            print(f"    (sub-passo 2t {s}h indisponível: {e})")
        finally:
            _remover(g)
    if tmin is None:
        raise RuntimeError("nenhum sub-passo de 2t disponível")
    _cache_save(cache_dir, "temp", dia, step, lons=lons, lats=lats,
                tmin=tmin, tmax=tmax, valido=_dt_iso(valido))
    return lons, lats, tmin, tmax, valido


# =========================================================================
# FIGURA
# =========================================================================
def _caminho_poligono(regiao):
    from matplotlib.path import Path
    verts, codes = [], []
    for poly in regiao["poligonos"]:
        for i, (lon, lat) in enumerate(poly):
            verts.append((lon, lat))
            codes.append(Path.MOVETO if i == 0 else Path.LINETO)
        codes[-1] = Path.CLOSEPOLY
    return Path(verts, codes)


def _add_logo(fig, ax, caminho, pos, escala, alpha):
    """Sobrepõe a logo (PNG) num canto do mapa, preservando a proporção.
    'escala' = largura da logo como fração da largura do eixo."""
    import matplotlib.image as mpimg
    from matplotlib.offsetbox import OffsetImage, AnnotationBbox
    try:
        img = mpimg.imread(caminho)
    except Exception as e:
        print(f"    (aviso: não consegui ler a logo {caminho}: {e})")
        return
    # largura desejada (px) = fração * largura do eixo (px)
    ax_w_in = fig.get_size_inches()[0] * ax.get_position().width
    alvo_px = max(escala * ax_w_in * fig.dpi, 1.0)
    zoom = alvo_px / img.shape[1]
    oi = OffsetImage(img, zoom=zoom, alpha=alpha)
    cantos = {
        "inferior-esquerda": (0.02, 0.02, (0.0, 0.0)),
        "inferior-direita":  (0.98, 0.02, (1.0, 0.0)),
        "superior-esquerda": (0.02, 0.98, (0.0, 1.0)),
        "superior-direita":  (0.98, 0.98, (1.0, 1.0)),
    }
    x, y, ba = cantos.get(pos, cantos["inferior-direita"])
    ab = AnnotationBbox(oi, (x, y), xycoords="axes fraction",
                        box_alignment=ba, frameon=False, pad=0.0, zorder=10)
    ax.add_artist(ab)


def plotar(lons, lats, dados, titulo, periodo_txt, png_path, faixas,
           cor_acima=None, cor_abaixo=None, extend="max",
           extent=None, regiao=None, fundo=None, recortar=False, cidades=None,
           logo=None, logo_pos="inferior-direita", logo_escala=0.16, logo_alpha=1.0):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    levels, cmap, norm, ticks = construir_colormap(faixas, cor_acima, cor_abaixo)
    lon2d, lat2d = np.meshgrid(lons, lats)

    fig, ax = plt.subplots(figsize=(9, 9), dpi=130)
    cs = ax.contourf(lon2d, lat2d, dados, levels=levels, cmap=cmap, norm=norm,
                     extend=extend, antialiased=True)

    if recortar and regiao is not None:
        clip = _caminho_poligono(regiao)
        patch = mpatches.PathPatch(clip, transform=ax.transData,
                                   facecolor="none", edgecolor="none")
        ax.add_patch(patch)
        if hasattr(cs, "collections"):
            for col in cs.collections:
                col.set_clip_path(patch)
        else:
            cs.set_clip_path(patch)

    # fundo: contorno fino dos ESTADOS (contexto). As outras mesorregiões não
    # entram aqui — só o fundo de estados e a região escolhida em destaque.
    if fundo:
        for r in fundo:
            for poly in r["poligonos"]:
                xy = np.array(poly)
                ax.plot(xy[:, 0], xy[:, 1], color="black", linewidth=0.5, alpha=0.55)
    if regiao is not None:
        for poly in regiao["poligonos"]:
            xy = np.array(poly)
            ax.plot(xy[:, 0], xy[:, 1], color="black", linewidth=1.6)

    # pontos de cidade (referência p/ localizar) — sempre por cima do resto
    if cidades:
        import matplotlib.patheffects as pe
        halo = [pe.withStroke(linewidth=2.4, foreground="white")]
        for (nome, lat, lon) in cidades:
            ax.plot(lon, lat, marker="o", markersize=5.5, markerfacecolor="black",
                    markeredgecolor="white", markeredgewidth=0.9, zorder=6)
            t = ax.annotate(nome, (lon, lat), xytext=(5, 4),
                            textcoords="offset points", fontsize=11.5,
                            fontweight="bold", color="black", zorder=7)
            t.set_path_effects(halo)

    if extent is not None:
        ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3])
    else:
        ax.set_xlim(float(lons.min()), float(lons.max()))
        ax.set_ylim(float(lats.min()), float(lats.max()))

    lat_med = np.deg2rad((ax.get_ylim()[0] + ax.get_ylim()[1]) / 2.0)
    ax.set_aspect(1.0 / max(np.cos(lat_med), 1e-3))

    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_linewidth(1.0)

    ax.set_title(titulo, loc="left", fontsize=16, fontweight="bold", pad=8)
    ax.set_title(periodo_txt, loc="right", fontsize=16, fontweight="bold",
                 color="blue", pad=8)

    cb = fig.colorbar(cs, ax=ax, fraction=0.046, pad=0.02, ticks=ticks, extend=extend)
    cb.ax.tick_params(labelsize=15)
    cb.set_ticklabels(["%g" % t for t in ticks])

    if logo:
        _add_logo(fig, ax, logo, logo_pos, logo_escala, logo_alpha)

    fig.savefig(png_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# =========================================================================
# PERÍODO / RÓTULOS
# =========================================================================
def _fmt_dia(d):
    return f"{d.day}/{MESES_PT[d.month - 1]}/{str(d.year)[2:]}"


def periodo_acumulado(base, dias):
    ini = base
    fim = base + dt.timedelta(days=dias)
    yy = str(fim.year)[2:]
    if ini.year == fim.year and ini.month == fim.month:
        return f"{ini.day} a {fim.day}/{MESES_PT[fim.month - 1]}/{yy}"
    if ini.year == fim.year:
        return (f"{ini.day}/{MESES_PT[ini.month - 1]} a "
                f"{fim.day}/{MESES_PT[fim.month - 1]}/{yy}")
    return f"{_fmt_dia(ini)} a {_fmt_dia(fim)}"


def periodo_diario(base, dias):
    return _fmt_dia(base + dt.timedelta(days=dias))


def _base_do_valido(valido, dias, rodada_hoje):
    return (valido - dt.timedelta(days=dias)).date() if valido is not None else rodada_hoje


# =========================================================================
# MAIN
# =========================================================================
VARS_VALIDAS = ["chuva", "tmin", "tmax", "nuvem"]


def main():
    # saída sem buffer: no GitHub Actions o log passa a aparecer em tempo real
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="Previsão ECMWF por estado/mesorregião")
    ap.add_argument("--regiao", nargs="*", default=[],
                    help="uma ou mais regiões (nome/sigla/código). Ex.: --regiao SP MG")
    ap.add_argument("--regioes", default=None,
                    help="regiões separadas por ; (bom p/ nomes com espaço)")
    ap.add_argument("--sem-brasil", action="store_true",
                    help="NÃO gerar o Brasil inteiro (por padrão ele é sempre gerado)")
    ap.add_argument("--cache", default=".cache_ecmwf",
                    help="pasta de cache do dia (evita rebaixar). Padrão: .cache_ecmwf")
    ap.add_argument("--sem-cache", action="store_true",
                    help="desliga o cache (baixa sempre)")
    ap.add_argument("--kml", required=True, nargs="+",
                    help="um ou mais KML (ex.: estados.kml mesorregioes.kml)")
    ap.add_argument("--fundo", nargs="*", default=None,
                    help="KML(s) usados como contorno de fundo. Padrão: os que "
                         "tiverem 'estado' no nome do arquivo.")
    ap.add_argument("--fonte", nargs="+", default=None,
                    choices=["aws", "azure", "ecmwf"],
                    help="fonte(s) do ECMWF, em ordem de tentativa "
                         "(padrão: aws azure ecmwf — espelhos primeiro, evita 429)")
    ap.add_argument("--vars", nargs="+", default=VARS_VALIDAS, choices=VARS_VALIDAS,
                    help="variáveis a gerar (padrão: todas)")
    ap.add_argument("--dias", nargs="+", type=int, default=[1, 2, 3, 4, 5, 6, 7],
                    help="horizontes em dias (1..7)")
    ap.add_argument("--saida", default="saida_previsao", help="pasta de saída")
    ap.add_argument("--margem", type=float, default=1.0,
                    help="folga em graus ao redor da região no recorte da imagem")
    ap.add_argument("--recortar", action="store_true",
                    help="limita o preenchimento ao polígono da região")
    ap.add_argument("--todas-meso", action="store_true",
                    help="gera TODAS as mesorregiões (as regiões que não são fundo), "
                         "em pastas <saida>/<UF>/<mesorregião>/")
    ap.add_argument("--todos-estados", action="store_true",
                    help="gera também cada estado como figura própria, em <saida>/<UF>/")
    ap.add_argument("--cidades", default=None,
                    help="arquivo de cidades (CSV nome,lat,lon ou KML de pontos); "
                         "mostra as que caem dentro da região focada")
    ap.add_argument("--cidade", action="append", default=[],
                    help="ponto avulso 'Nome,lat,lon' (repetível); sempre desenhado")
    ap.add_argument("--logo", default=None, help="PNG de logo para sobrepor ao mapa")
    ap.add_argument("--logo-pos", default="inferior-direita",
                    choices=["inferior-esquerda", "inferior-direita",
                             "superior-esquerda", "superior-direita"],
                    help="canto da logo (padrão: inferior-direita)")
    ap.add_argument("--logo-escala", type=float, default=0.16,
                    help="largura da logo como fração da largura do mapa (0..1)")
    ap.add_argument("--logo-alpha", type=float, default=1.0,
                    help="opacidade da logo (0..1)")
    args = ap.parse_args()

    vars_sel = list(dict.fromkeys(args.vars))  # únicas, mantendo ordem

    global FONTES
    if args.fonte:
        FONTES = list(dict.fromkeys(args.fonte))
    print(f"Fontes ECMWF (ordem de tentativa): {', '.join(FONTES)}")

    pedidos = list(args.regiao)
    if args.regioes:
        pedidos += [p.strip() for p in args.regioes.split(";") if p.strip()]
    if not pedidos and args.sem_brasil and not args.todas_meso and not args.todos_estados:
        sys.exit("Nada a gerar: sem regiões, --sem-brasil e sem --todas-meso/--todos-estados.")

    existentes = [c for c in args.kml if os.path.exists(c)]
    for c in [c for c in args.kml if not os.path.exists(c)]:
        print(f"  AVISO: KML não encontrado, ignorando: {c}")
    if not existentes:
        sys.exit(f"Nenhum KML encontrado entre [{', '.join(args.kml)}] "
                 f"(pasta atual: {os.getcwd()}). Confira o caminho relativo "
                 f"à raiz do repositório.")

    regioes = ler_kmls(existentes)
    if not regioes:
        sys.exit(f"Nenhum polígono lido de {', '.join(existentes)}")
    print(f"Total: {len(regioes)} região(ões) disponível(is) para busca")

    # camada de FUNDO (contexto): só os estados. Por padrão, os arquivos com
    # 'estado' no nome; ou os informados em --fundo. A busca continua usando
    # TODAS as regiões (estados + mesorregiões).
    if args.fundo:
        bases_fundo = {os.path.basename(c) for c in args.fundo}
        fundo_regioes = [r for r in regioes if r["arquivo"] in bases_fundo]
    else:
        fundo_regioes = [r for r in regioes if "estado" in _sem_acento(r["arquivo"])]
    if not fundo_regioes:
        print("  AVISO: nenhum KML de fundo identificado — usando todos como fundo. "
              "(informe --fundo estados.kml para desenhar só os estados)")
        fundo_regioes = regioes
    else:
        arqs = sorted({r["arquivo"] for r in fundo_regioes})
        print(f"Fundo (contorno de estados): {', '.join(arqs)} "
              f"({len(fundo_regioes)} polígono(s))")

    # cidades de referência
    cidades_arquivo = []
    if args.cidades:
        if os.path.exists(args.cidades):
            cidades_arquivo = ler_cidades(args.cidades)
            print(f"Cidades: {len(cidades_arquivo)} ponto(s) de {args.cidades}")
        else:
            print(f"  AVISO: arquivo de cidades não encontrado: {args.cidades}")
    cidades_cli = []
    for txt in args.cidade:
        try:
            cidades_cli.append(_parse_cidade_cli(txt))
        except ValueError as e:
            print(f"  AVISO: {e}")

    # logo (opcional): resolve uma vez; se faltar, segue sem
    logo = None
    if args.logo:
        if os.path.exists(args.logo):
            logo = args.logo
            print(f"Logo: {args.logo} ({args.logo_pos})")
        else:
            print(f"  AVISO: logo não encontrada, seguindo sem: {args.logo}")

    alvos = []  # (subdir, regiao|None, extent|None, cidades)

    def _alvo_de_regiao(r):
        lo0, la0, lo1, la1 = bbox_regiao(r)
        m = args.margem
        ext = (lo0 - m, lo1 + m, la0 - m, la1 + m)
        cids = list(cidades_cli)
        cids += [c for c in cidades_arquivo if ponto_na_regiao(c[2], c[1], r)]
        sub = subdir_do_alvo(r, fundo_regioes)
        return (sub, r, ext, cids)

    if not args.sem_brasil:
        # enquadra o Brasil no bbox dos estados (fundo) + margem, em vez do
        # domínio inteiro que é baixado (que vai muito além do Brasil).
        bb = bbox_uniao(fundo_regioes)
        ext_br = None
        if bb:
            lo0, la0, lo1, la1 = bb
            mb = max(args.margem, 1.5)
            ext_br = (lo0 - mb, lo1 + mb, la0 - mb, la1 + mb)
        alvos.append(("brasil", None, ext_br, list(cidades_cli)))
        print("Alvo: Brasil inteiro (sempre)")

    for pedido in pedidos:
        r = selecionar_regiao(regioes, pedido)
        if r is None:
            exemplos = ", ".join(sorted(x["nome"] for x in regioes)[:12])
            print(f"  AVISO: região '{pedido}' não encontrada, pulando. Ex.: {exemplos} ...")
            continue
        sub, r, ext, cids = _alvo_de_regiao(r)
        alvos.append((sub, r, ext, cids))
        print(f"Alvo: {r['nome']} -> {sub}  cidades={len(cids)}")

    if args.todos_estados:
        print(f"Todos os estados: {len(fundo_regioes)}")
        for r in fundo_regioes:
            sub, r, ext, cids = _alvo_de_regiao(r)
            alvos.append((sub, r, ext, cids))

    if args.todas_meso:
        # mesorregiões = tudo que NÃO é fundo (estado)
        ids_fundo = {id(r) for r in fundo_regioes}
        mesos = [r for r in regioes if id(r) not in ids_fundo]
        print(f"Todas as mesorregiões: {len(mesos)}")
        sem_uf = 0
        for r in mesos:
            sub, r, ext, cids = _alvo_de_regiao(r)
            if sub.startswith("SEM_UF" + os.sep) or sub.startswith("SEM_UF/"):
                sem_uf += 1
            alvos.append((sub, r, ext, cids))
        if sem_uf:
            print(f"  AVISO: {sem_uf} mesorregião(ões) sem UF identificada "
                  f"(foram para a pasta SEM_UF/).")

    if not alvos:
        sys.exit("Nenhum alvo válido — nada a gerar.")

    os.makedirs(args.saida, exist_ok=True)
    cache_dir = None if args.sem_cache else args.cache
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
    dia_cache = dt.datetime.utcnow().date().isoformat()   # dado "do dia" (UTC)
    tmp = os.path.join(cache_dir or args.saida, "_tmp")
    rodada_hoje = dt.date.today()

    if cache_dir:
        print(f"Cache: {cache_dir} (dia {dia_cache}) — não rebaixa o que já tem")

    # =====================================================================
    # FASE 1 — DOWNLOAD: baixa tudo primeiro (1x por variável/dia). Toda a
    # rede acontece aqui; os campos ficam em memória para a fase de corte.
    # =====================================================================
    print(f"Variáveis: {', '.join(vars_sel)} | dias: {sorted(set(args.dias))} | "
          f"alvos: {len(alvos)}")
    print("=== Fase 1: download do ECMWF ===")
    dias_dados = {}  # dias -> {"lons","lats","produtos":[...]}
    for dias in sorted(set(args.dias)):
        if dias < 1 or dias > 7:
            print(f"  (pulando dia {dias}: fora de 1..7)")
            continue
        step = dias * 24
        lons = lats = None
        produtos = []  # (tipo, campo, titulo, periodo, faixas, c_acima, c_abaixo, extend)

        if "chuva" in vars_sel:
            print(f"[{dias}d] chuva (tp)...")
            try:
                lo, la, acum, diario, valido = obter_chuva(step, tmp, cache_dir=cache_dir, dia=dia_cache)
                lons, lats = lo, la
                base = _base_do_valido(valido, dias, rodada_hoje)
                produtos.append(("chuva_acumulado", acum, "Acumulado (mm)",
                                 periodo_acumulado(base, dias),
                                 FAIXAS_CHUVA, CHUVA_ACIMA, None, "max"))
                produtos.append(("chuva_dia", diario, "Total (mm)",
                                 periodo_diario(base, dias),
                                 FAIXAS_CHUVA, CHUVA_ACIMA, None, "max"))
            except Exception as e:
                print(f"  ERRO chuva {dias}d: {e}")

        if "tmin" in vars_sel or "tmax" in vars_sel:
            print(f"[{dias}d] temperatura (2t, sub-passos)...")
            try:
                lo, la, tmin, tmax, valido = obter_temp(step, tmp, cache_dir=cache_dir, dia=dia_cache)
                lons, lats = lo, la
                base = _base_do_valido(valido, dias, rodada_hoje)
                if "tmin" in vars_sel:
                    produtos.append(("tmin", tmin, "Temp. mínima (°C)",
                                     periodo_diario(base, dias),
                                     FAIXAS_TEMP, TEMP_ACIMA, TEMP_ABAIXO, "both"))
                if "tmax" in vars_sel:
                    produtos.append(("tmax", tmax, "Temp. máxima (°C)",
                                     periodo_diario(base, dias),
                                     FAIXAS_TEMP, TEMP_ACIMA, TEMP_ABAIXO, "both"))
            except Exception as e:
                print(f"  ERRO temperatura {dias}d: {e}")

        if "nuvem" in vars_sel:
            print(f"[{dias}d] nuvem (tcc)...")
            try:
                lo, la, nuvem, valido = obter_nuvem(step, tmp, cache_dir=cache_dir, dia=dia_cache)
                lons, lats = lo, la
                base = _base_do_valido(valido, dias, rodada_hoje)
                produtos.append(("nuvem", nuvem, "Nuvens (%)",
                                 periodo_diario(base, dias),
                                 FAIXAS_NUVEM, None, None, "neither"))
            except Exception as e:
                print(f"  ERRO nuvem {dias}d: {e}")

        if produtos and lons is not None:
            dias_dados[dias] = {"lons": lons, "lats": lats, "produtos": produtos}
        else:
            print(f"  (dia {dias}: sem dados — não entra na fase de corte)")

    if not dias_dados:
        sys.exit("Nada foi baixado — nenhuma figura a gerar (ver erros acima).")

    # =====================================================================
    # FASE 2 — CORTE: gera as figuras a partir dos dados já baixados.
    # Nenhum acesso à rede aqui.
    # =====================================================================
    print("=== Fase 2: recortes e figuras (sem rede) ===")
    total_esperado = sum(len(alvos) * len(dias_dados[d]["produtos"]) for d in dias_dados)
    print(f"A gerar ~{total_esperado} figura(s) ({len(alvos)} alvo(s) x "
          f"{len(dias_dados)} dia(s)).")
    inicio = time.time()
    total = 0
    for dias in sorted(dias_dados):
        d = dias_dados[dias]
        lons, lats, produtos = d["lons"], d["lats"], d["produtos"]
        for (subdir, regiao, extent, cids) in alvos:
            outdir = os.path.join(args.saida, subdir)
            os.makedirs(outdir, exist_ok=True)
            for (tipo, campo, titulo, per, faixas, c_a, c_b, ext_cb) in produtos:
                png = os.path.join(outdir, f"ecmwf_{tipo}_{dias}d.png")
                plotar(lons, lats, campo, titulo, per, png, faixas,
                       cor_acima=c_a, cor_abaixo=c_b, extend=ext_cb,
                       extent=extent, regiao=regiao, fundo=fundo_regioes,
                       recortar=args.recortar, cidades=cids,
                       logo=logo, logo_pos=args.logo_pos,
                       logo_escala=args.logo_escala, logo_alpha=args.logo_alpha)
                total += 1
                if total % 50 == 0 or total == total_esperado:
                    seg = time.time() - inicio
                    taxa = total / seg if seg > 0 else 0
                    restam = (total_esperado - total) / taxa if taxa > 0 else 0
                    print(f"    {total}/{total_esperado} figuras "
                          f"| {seg:.0f}s decorridos | ~{restam:.0f}s restantes "
                          f"| {taxa:.1f} fig/s")
        print(f"  dia {dias}d concluído")
    print(f"Fase 2 concluída: {total} figura(s) em {time.time() - inicio:.0f}s.")

    for pasta in {args.saida, cache_dir or args.saida}:
        for fn in os.listdir(pasta):
            if fn.startswith("_tmp"):
                _remover(os.path.join(pasta, fn))
    print("Pronto.")


if __name__ == "__main__":
    main()
