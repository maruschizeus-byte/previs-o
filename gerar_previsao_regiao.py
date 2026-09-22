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
        if nome:
            chaves.add(_sem_acento(nome))
        for sd in achar(pm, "SimpleData"):
            if sd.text:
                chaves.add(_sem_acento(sd.text))
        for val in achar(pm, "value"):
            if val.text:
                chaves.add(_sem_acento(val.text))

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
                            "poligonos": poligonos,
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
def baixar(param, step, grib_file, data_rodada=None, hora_rodada=0):
    from ecmwf.opendata import Client
    client = Client(source="ecmwf")
    kw = dict(type="fc", stream="oper", param=param, step=step, target=grib_file)
    if data_rodada is not None:
        kw["date"] = data_rodada.strftime("%Y%m%d")
        kw["time"] = hora_rodada
    client.retrieve(**kw)


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


def obter_chuva(step, tmp_prefix, data_rodada=None, hora_rodada=0):
    """(lons, lats, acumulado_mm, diario_mm, valido). Acumulado = tp no passo;
    diário = tp(step) - tp(step-24); no dia 1 o anterior é 0."""
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
    return lons, lats, acum, diario, valido


def obter_nuvem(step, tmp_prefix, data_rodada=None, hora_rodada=0):
    """(lons, lats, nuvem_%, valido). tcc é fração instantânea 0..1."""
    g = f"{tmp_prefix}_tcc_{step}h.grib2"
    baixar("tcc", step, g, data_rodada, hora_rodada)
    lons, lats, arr = ler_grib(g, "tcc", 100.0, 0.0)
    valido = _valid_utc(g, "tcc")
    _remover(g)
    return lons, lats, np.clip(arr, 0, 100), valido


def obter_temp(step, tmp_prefix, data_rodada=None, hora_rodada=0):
    """(lons, lats, tmin_C, tmax_C, valido). Mín e máx do dia a partir dos
    sub-passos de 2t (3 em 3 h até 144 h, 6 em 6 h depois). Baixa cada
    sub-passo UMA vez e atualiza mín e máx juntos."""
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


def plotar(lons, lats, dados, titulo, periodo_txt, png_path, faixas,
           cor_acima=None, cor_abaixo=None, extend="max",
           extent=None, regiao=None, fundo=None, recortar=False, cidades=None):
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

    ax.set_title(titulo, loc="left", fontsize=20, fontweight="bold", pad=8)
    ax.set_title(periodo_txt, loc="right", fontsize=20, fontweight="bold",
                 color="blue", pad=8)

    cb = fig.colorbar(cs, ax=ax, fraction=0.046, pad=0.02, ticks=ticks, extend=extend)
    cb.ax.tick_params(labelsize=15)
    cb.set_ticklabels(["%g" % t for t in ticks])

    fig.savefig(png_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# =========================================================================
# PERÍODO / RÓTULOS
# =========================================================================
def _fmt_dia(d):
    return f"{d.day}/{MESES_PT[d.month - 1]}/{str(d.year)[2:]}"


def periodo_acumulado(base, dias):
    return f"{_fmt_dia(base)} a {_fmt_dia(base + dt.timedelta(days=dias))}"


def periodo_diario(base, dias):
    return _fmt_dia(base + dt.timedelta(days=dias))


def _base_do_valido(valido, dias, rodada_hoje):
    return (valido - dt.timedelta(days=dias)).date() if valido is not None else rodada_hoje


# =========================================================================
# MAIN
# =========================================================================
VARS_VALIDAS = ["chuva", "tmin", "tmax", "nuvem"]


def main():
    ap = argparse.ArgumentParser(description="Previsão ECMWF por estado/mesorregião")
    ap.add_argument("--regiao", nargs="*", default=[],
                    help="uma ou mais regiões (nome/sigla/código). Ex.: --regiao SP MG")
    ap.add_argument("--regioes", default=None,
                    help="regiões separadas por ; (bom p/ nomes com espaço)")
    ap.add_argument("--brasil", action="store_true", help="inclui o Brasil inteiro")
    ap.add_argument("--kml", required=True, nargs="+",
                    help="um ou mais KML (ex.: estados.kml mesorregioes.kml)")
    ap.add_argument("--fundo", nargs="*", default=None,
                    help="KML(s) usados como contorno de fundo. Padrão: os que "
                         "tiverem 'estado' no nome do arquivo.")
    ap.add_argument("--vars", nargs="+", default=VARS_VALIDAS, choices=VARS_VALIDAS,
                    help="variáveis a gerar (padrão: todas)")
    ap.add_argument("--dias", nargs="+", type=int, default=[1, 2, 3, 4, 5, 6, 7],
                    help="horizontes em dias (1..7)")
    ap.add_argument("--saida", default="saida_previsao", help="pasta de saída")
    ap.add_argument("--margem", type=float, default=1.0,
                    help="folga em graus ao redor da região no recorte da imagem")
    ap.add_argument("--recortar", action="store_true",
                    help="limita o preenchimento ao polígono da região")
    ap.add_argument("--cidades", default=None,
                    help="arquivo de cidades (CSV nome,lat,lon ou KML de pontos); "
                         "mostra as que caem dentro da região focada")
    ap.add_argument("--cidade", action="append", default=[],
                    help="ponto avulso 'Nome,lat,lon' (repetível); sempre desenhado")
    args = ap.parse_args()

    vars_sel = list(dict.fromkeys(args.vars))  # únicas, mantendo ordem

    pedidos = list(args.regiao)
    if args.regioes:
        pedidos += [p.strip() for p in args.regioes.split(";") if p.strip()]
    if not pedidos and not args.brasil:
        sys.exit("Informe ao menos uma região (--regiao / --regioes) ou --brasil.")

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

    alvos = []  # (sufixo, regiao|None, extent|None, cidades)
    if args.brasil:
        alvos.append(("brasil", None, None, list(cidades_cli)))
        print("Alvo: Brasil inteiro (sem recorte)")
    for pedido in pedidos:
        r = selecionar_regiao(regioes, pedido)
        if r is None:
            exemplos = ", ".join(sorted(x["nome"] for x in regioes)[:12])
            print(f"  AVISO: região '{pedido}' não encontrada, pulando. Ex.: {exemplos} ...")
            continue
        lo0, la0, lo1, la1 = bbox_regiao(r)
        m = args.margem
        ext = (lo0 - m, lo1 + m, la0 - m, la1 + m)
        suf = _sem_acento(r["nome"]).replace(" ", "_").replace("/", "-")
        # cidades do alvo: as avulsas (sempre) + as do arquivo dentro da região
        cids = list(cidades_cli)
        cids += [c for c in cidades_arquivo if ponto_na_regiao(c[2], c[1], r)]
        alvos.append((suf, r, ext, cids))
        print(f"Alvo: {r['nome']}  bbox={lo0:.2f},{la0:.2f}..{lo1:.2f},{la1:.2f}"
              f"  cidades={len(cids)}")
    if not alvos:
        sys.exit("Nenhum alvo válido — nada a gerar.")

    os.makedirs(args.saida, exist_ok=True)
    tmp = os.path.join(args.saida, "_tmp")
    rodada_hoje = dt.date.today()

    print(f"Variáveis: {', '.join(vars_sel)} | dias: {sorted(set(args.dias))} | "
          f"alvos: {len(alvos)}")

    for dias in sorted(set(args.dias)):
        if dias < 1 or dias > 7:
            print(f"  (pulando dia {dias}: fora de 1..7)")
            continue
        step = dias * 24
        lons = lats = None
        # produtos = (tipo, campo, titulo, periodo, faixas, cor_acima, cor_abaixo, extend)
        produtos = []

        if "chuva" in vars_sel:
            print(f"[{dias}d] baixando CHUVA (tp) 1x...")
            try:
                lo, la, acum, diario, valido = obter_chuva(step, tmp)
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
            print(f"[{dias}d] baixando TEMPERATURA (2t, sub-passos) 1x...")
            try:
                lo, la, tmin, tmax, valido = obter_temp(step, tmp)
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
            print(f"[{dias}d] baixando NUVEM (tcc) 1x...")
            try:
                lo, la, nuvem, valido = obter_nuvem(step, tmp)
                lons, lats = lo, la
                base = _base_do_valido(valido, dias, rodada_hoje)
                produtos.append(("nuvem", nuvem, "Nuvens (%)",
                                 periodo_diario(base, dias),
                                 FAIXAS_NUVEM, None, None, "neither"))
            except Exception as e:
                print(f"  ERRO nuvem {dias}d: {e}")

        if not produtos or lons is None:
            print(f"  (dia {dias}: nada gerado)")
            continue

        # mesmo dado, vários recortes
        for (suf, regiao, extent, cids) in alvos:
            for (tipo, campo, titulo, per, faixas, c_a, c_b, ext_cb) in produtos:
                png = os.path.join(args.saida, f"ecmwf_{tipo}_{suf}_{dias}d.png")
                plotar(lons, lats, campo, titulo, per, png, faixas,
                       cor_acima=c_a, cor_abaixo=c_b, extend=ext_cb,
                       extent=extent, regiao=regiao, fundo=fundo_regioes,
                       recortar=args.recortar, cidades=cids)
                print(f"  gerado: {png}")

    for fn in os.listdir(args.saida):
        if fn.startswith("_tmp"):
            _remover(os.path.join(args.saida, fn))
    print("Pronto.")


if __name__ == "__main__":
    main()
