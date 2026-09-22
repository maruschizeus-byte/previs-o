#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gera figuras de previsão de chuva do ECMWF (Open Data) no estilo do painel
"Total (mm)" — colorbar à direita, título e período —, com a opção de FOCAR
a imagem em um ESTADO ou MESORREGIÃO lido de um arquivo KML.

Para cada horizonte (1..7 dias) gera duas figuras:
  - chuva ACUMULADA desde a rodada  ("Acumulado (mm)")
  - chuva DO DIA (desacumulada)      ("Total (mm)")

Diferença para o gerar_previsao_ecmwf.py: aquele produz PNGs transparentes
para sobrepor no Leaflet; ESTE produz a figura completa (com eixo, borda e
colorbar) para baixar/compartilhar/imprimir, recortada na região escolhida.

USO
    pip install ecmwf-opendata xarray cfgrib numpy matplotlib
    # (sistema) eccodes p/ o cfgrib ler GRIB: apt-get install libeccodes0

    # Foco em um estado (casa por nome ou sigla no KML):
    python gerar_previsao_regiao.py --regiao SP --kml estados.kml
    python gerar_previsao_regiao.py --regiao "Minas Gerais" --kml estados.kml

    # Foco em uma mesorregião (aponte o KML das mesorregiões):
    python gerar_previsao_regiao.py --regiao "Triângulo Mineiro/Alto Paranaíba" \
        --kml mesorregioes.kml

    # Brasil inteiro (sem recorte de região):
    python gerar_previsao_regiao.py --brasil --kml estados.kml

    # Só alguns horizontes, e recortando o preenchimento ao polígono da região:
    python gerar_previsao_regiao.py --regiao GO --kml estados.kml --dias 1 2 3 --recortar

O KML pode ter várias regiões (Placemarks). O nome é casado, sem acento e sem
caixa, contra: o <name> do Placemark, e quaisquer campos de ExtendedData
(SimpleData/Data) — então "SP", "São Paulo", "35" etc. funcionam se estiverem lá.
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

# Domínio máximo baixado do ECMWF (América do Sul). O recorte da REGIÃO é feito
# depois, só no desenho — sempre baixamos o domínio todo para reaproveitar o
# mesmo GRIB entre regiões e evitar recortes na borda.
LON_MIN, LON_MAX = -82.0, -30.0
LAT_MIN, LAT_MAX = -60.0, 15.0

MESES_PT = ["jan", "fev", "mar", "abr", "mai", "jun",
            "jul", "ago", "set", "out", "nov", "dez"]

# ---- Paleta de chuva (mesma leitura visual do painel de referência) --------
# Cada faixa: (limite_inf, limite_sup, cor_inicial, cor_final). Dentro da faixa
# a cor faz um degradê linear entre inicial e final; nas bordas das faixas
# (5, 10, 25, ...) fica o "salto" de cor característico do mapa de precipitação.
FAIXAS_CHUVA = [
    (0,   5,   "#ffffff", "#8f8f8f"),  # 0–5   branco -> cinza (chuvisco)
    (5,   10,  "#7be07b", "#238b3a"),  # 5–10  verde claro -> verde escuro
    (10,  25,  "#63b1ff", "#123fb0"),  # 10–25 azul claro -> azul forte
    (25,  50,  "#fff23f", "#f07a00"),  # 25–50 amarelo -> laranja
    (50,  75,  "#ff4a1a", "#7a0000"),  # 50–75 vermelho -> vermelho escuro
    (75,  100, "#8a5a3c", "#caa07a"),  # 75–100 marrom -> bege
    (100, 150, "#c8bce0", "#8f7fc0"),  # 100–150 lavanda
    (150, 200, "#a020b0", "#e83ce8"),  # 150–200 roxo -> magenta
]
COR_ACIMA = "#f3ddf3"     # > 200 mm (ponta do colorbar)
SUBNIVEIS = 6             # sub-faixas por faixa (suaviza o degradê)


# =========================================================================
# PALETA / COLORMAP
# =========================================================================
def _hex_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))


def construir_colormap():
    """Monta (levels, cmap, norm, ticks) a partir de FAIXAS_CHUVA.

    Cada faixa é subdividida em SUBNIVEIS degraus com cor interpolada. Como
    toda faixa recebe o mesmo número de degraus, os rótulos 0,5,10,...,200
    ficam igualmente espaçados no colorbar (igual ao painel de referência).
    """
    from matplotlib.colors import ListedColormap, BoundaryNorm

    levels = []
    cores = []
    for (a, b, c0, c1) in FAIXAS_CHUVA:
        r0, g0, bl0 = _hex_rgb(c0)
        r1, g1, bl1 = _hex_rgb(c1)
        for k in range(SUBNIVEIS):
            f0 = k / SUBNIVEIS
            f1 = (k + 1) / SUBNIVEIS
            lv0 = a + (b - a) * f0
            if not levels or abs(levels[-1] - lv0) > 1e-9:
                levels.append(lv0)
            # cor no meio do degrau
            fm = (f0 + f1) / 2.0
            cores.append((r0 + (r1 - r0) * fm,
                          g0 + (g1 - g0) * fm,
                          bl0 + (bl1 - bl0) * fm))
        levels.append(b)
    levels = sorted(set(levels))
    levels = np.array(levels, dtype="float64")

    cmap = ListedColormap(cores)
    cmap.set_over(_hex_rgb(COR_ACIMA))
    cmap.set_under("#ffffff")
    norm = BoundaryNorm(levels, cmap.N)
    ticks = [a for (a, _b, _c0, _c1) in FAIXAS_CHUVA] + [FAIXAS_CHUVA[-1][1]]
    return levels, cmap, norm, ticks


# =========================================================================
# KML: leitura de polígonos e seleção da região
# =========================================================================
def _sem_acento(s):
    s = unicodedata.normalize("NFKD", s or "")
    s = "".join(c for c in s if not unicodedata.combining(c))
    return s.strip().lower()


def _parse_coords(texto):
    """'-53.1,-22.6,0 -50,-25 ...' -> [(lon,lat), ...]."""
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
    """Lê um KML e devolve lista de regiões:
        {"nome": str, "chaves": set[str], "poligonos": [[(lon,lat),...], ...]}

    'chaves' reúne todos os nomes pelos quais a região pode ser buscada:
    o <name> e todo valor de ExtendedData (SimpleData/Data).
    """
    tree = ET.parse(caminho)
    root = tree.getroot()

    def tag(e):
        return e.tag.split("}")[-1]  # ignora namespace

    def achar(e, nome):
        return [x for x in e.iter() if tag(x) == nome]

    regioes = []
    for pm in achar(root, "Placemark"):
        # nome principal
        nome = ""
        for filho in list(pm):
            if tag(filho) == "name" and filho.text:
                nome = filho.text.strip()
                break
        # chaves de busca: nome + todos os valores de ExtendedData
        chaves = set()
        if nome:
            chaves.add(_sem_acento(nome))
        for sd in achar(pm, "SimpleData"):
            if sd.text:
                chaves.add(_sem_acento(sd.text))
        for val in achar(pm, "value"):  # <Data><value>...</value></Data>
            if val.text:
                chaves.add(_sem_acento(val.text))

        # geometrias (Polygon simples e MultiGeometry): pega os outerBoundary
        poligonos = []
        for poly in achar(pm, "Polygon"):
            for ob in achar(poly, "outerBoundaryIs"):
                for coords in achar(ob, "coordinates"):
                    pts = _parse_coords(coords.text)
                    if len(pts) >= 3:
                        poligonos.append(pts)
        if not poligonos:  # fallback: qualquer <coordinates> de anel
            for coords in achar(pm, "coordinates"):
                pts = _parse_coords(coords.text)
                if len(pts) >= 3:
                    poligonos.append(pts)

        if poligonos:
            regioes.append({"nome": nome or "(sem nome)",
                            "chaves": chaves, "poligonos": poligonos,
                            "arquivo": os.path.basename(caminho)})
    return regioes


def ler_kmls(caminhos):
    """Lê vários KMLs e devolve a lista unida de regiões. Permite ter, no
    mesmo comando, o KML de estados E o de mesorregiões: a busca varre os dois."""
    todas = []
    for c in caminhos:
        lidas = ler_kml(c)
        print(f"KML: {len(lidas)} região(ões) em {c}")
        todas.extend(lidas)
    return todas


def selecionar_regiao(regioes, alvo):
    """Acha a região cujo conjunto de chaves casa com 'alvo' (sem acento/caixa).
    Tenta match exato; se não houver, tenta 'começa com' e depois 'contém'."""
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
    return min(xs), min(ys), max(xs), max(ys)  # lon_min, lat_min, lon_max, lat_max


# =========================================================================
# ECMWF: download e processamento da precipitação (tp)
# =========================================================================
def baixar_tp(step, grib_file, data_rodada=None, hora_rodada=0):
    """Baixa 'tp' (precipitação total acumulada, em metros) de um passo."""
    from ecmwf.opendata import Client
    client = Client(source="ecmwf")
    kw = dict(type="fc", stream="oper", param="tp", step=step, target=grib_file)
    if data_rodada is not None:
        kw["date"] = data_rodada.strftime("%Y%m%d")
        kw["time"] = hora_rodada
    client.retrieve(**kw)


def ler_grib_tp(grib_file):
    """Lê o GRIB de tp, ajusta longitude p/ -180..180 e recorta o domínio.
    Retorna (lons[1d], lats[1d], arr[2d] em mm) já em ordem lat decrescente."""
    import xarray as xr
    ds = xr.open_dataset(grib_file, engine="cfgrib")
    da = ds["tp"] if "tp" in ds else ds[list(ds.data_vars)[0]]
    if float(da.longitude.max()) > 180:
        da = da.assign_coords(
            longitude=(((da.longitude + 180) % 360) - 180)
        ).sortby("longitude")
    rec = da.sel(latitude=slice(LAT_MAX, LAT_MIN),
                 longitude=slice(LON_MIN, LON_MAX))
    lats = np.asarray(rec.latitude.values, dtype="float64")
    lons = np.asarray(rec.longitude.values, dtype="float64")
    arr = np.asarray(rec.values, dtype="float32") * 1000.0  # m -> mm
    if lats[0] < lats[-1]:  # garante norte no topo
        lats = lats[::-1]
        arr = arr[::-1, :]
    return lons, lats, arr


def _valid_utc(grib_file):
    try:
        import xarray as xr
        ds = xr.open_dataset(grib_file, engine="cfgrib")
        v = ds["tp"].coords.get("valid_time")
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
    """Devolve (lons, lats, acumulado_mm, diario_mm, valido_utc) para o passo.

    - acumulado: tp direto no passo (chuva desde a rodada até o dia).
    - diario:    tp(step) - tp(step-24). No dia 1 (step 24), o anterior é 0.
    """
    g_atual = f"{tmp_prefix}_{step}h.grib2"
    baixar_tp(step, g_atual, data_rodada, hora_rodada)
    lons, lats, acum = ler_grib_tp(g_atual)
    valido = _valid_utc(g_atual)

    if step > 24:
        g_ant = f"{tmp_prefix}_{step-24}h.grib2"
        baixar_tp(step - 24, g_ant, data_rodada, hora_rodada)
        _, _, acum_ant = ler_grib_tp(g_ant)
        diario = np.clip(acum - acum_ant, 0, None)
        _remover(g_ant)
    else:
        diario = np.clip(acum, 0, None)

    _remover(g_atual)
    return lons, lats, acum, diario, valido


# =========================================================================
# FIGURA
# =========================================================================
def _caminho_poligono(regiao):
    """matplotlib.path.Path com todos os anéis da região (para recorte/borda)."""
    from matplotlib.path import Path
    verts, codes = [], []
    for poly in regiao["poligonos"]:
        for i, (lon, lat) in enumerate(poly):
            verts.append((lon, lat))
            codes.append(Path.MOVETO if i == 0 else Path.LINETO)
        codes[-1] = Path.CLOSEPOLY
    return Path(verts, codes)


def plotar(lons, lats, dados_mm, titulo, periodo_txt, png_path,
           extent=None, regiao=None, todas_regioes=None, recortar=False):
    """Desenha o painel de chuva e salva em png_path.

    extent = (lon_min, lon_max, lat_min, lat_max). Se None, usa todo o array.
    regiao = região selecionada (borda em destaque). todas_regioes = contexto
    (bordas finas). recortar=True limita o preenchimento ao polígono da região.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    levels, cmap, norm, ticks = construir_colormap()

    lon2d, lat2d = np.meshgrid(lons, lats)

    fig, ax = plt.subplots(figsize=(9, 9), dpi=130)
    cs = ax.contourf(lon2d, lat2d, dados_mm, levels=levels,
                     cmap=cmap, norm=norm, extend="max", antialiased=True)

    # recorte opcional do preenchimento ao polígono da região
    if recortar and regiao is not None:
        clip = _caminho_poligono(regiao)
        import matplotlib.patches as mpatches
        patch = mpatches.PathPatch(clip, transform=ax.transData,
                                   facecolor="none", edgecolor="none")
        ax.add_patch(patch)
        # matplotlib >=3.8: ContourSet é um único artista (sem .collections)
        if hasattr(cs, "collections"):
            for col in cs.collections:
                col.set_clip_path(patch)
        else:
            cs.set_clip_path(patch)

    # bordas de contexto (todas as regiões do KML), finas
    if todas_regioes:
        for r in todas_regioes:
            for poly in r["poligonos"]:
                xy = np.array(poly)
                ax.plot(xy[:, 0], xy[:, 1], color="black",
                        linewidth=0.5, alpha=0.55)

    # borda da região escolhida, em destaque
    if regiao is not None:
        for poly in regiao["poligonos"]:
            xy = np.array(poly)
            ax.plot(xy[:, 0], xy[:, 1], color="black", linewidth=1.6)

    if extent is not None:
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
    else:
        ax.set_xlim(float(lons.min()), float(lons.max()))
        ax.set_ylim(float(lats.min()), float(lats.max()))

    # aspecto ~geográfico (equiretangular corrigido pela latitude média)
    lat_med = np.deg2rad((ax.get_ylim()[0] + ax.get_ylim()[1]) / 2.0)
    ax.set_aspect(1.0 / max(np.cos(lat_med), 1e-3))

    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True); s.set_linewidth(1.0)

    # títulos no estilo da referência
    ax.set_title(titulo, loc="left", fontsize=20, fontweight="bold", pad=8)
    ax.set_title(periodo_txt, loc="right", fontsize=20,
                 fontweight="bold", color="blue", pad=8)

    # colorbar vertical à direita
    cb = fig.colorbar(cs, ax=ax, fraction=0.046, pad=0.02,
                      ticks=ticks, extend="max")
    cb.ax.tick_params(labelsize=15)
    cb.set_ticklabels([str(int(t)) for t in ticks])

    fig.savefig(png_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# =========================================================================
# PERÍODO / RÓTULOS
# =========================================================================
def _fmt_dia(d):
    return f"{d.day}/{MESES_PT[d.month - 1]}/{str(d.year)[2:]}"


def periodo_acumulado(rodada_dia, dias):
    """Rótulo do acumulado: da rodada até o fim do dia N."""
    ini = rodada_dia
    fim = rodada_dia + dt.timedelta(days=dias)
    return f"{_fmt_dia(ini)} a {_fmt_dia(fim)}"


def periodo_diario(rodada_dia, dias):
    """Rótulo da chuva do dia N (o próprio dia)."""
    d = rodada_dia + dt.timedelta(days=dias)
    return _fmt_dia(d)


# =========================================================================
# MAIN
# =========================================================================
def main():
    ap = argparse.ArgumentParser(description="Previsão ECMWF por estado/mesorregião")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--regiao", help="nome ou sigla da região (ex.: SP, 'Minas Gerais')")
    g.add_argument("--brasil", action="store_true", help="Brasil inteiro, sem recorte")
    ap.add_argument("--kml", required=True, nargs="+",
                    help="um ou mais KML (ex.: estados.kml mesorregioes.kml)")
    ap.add_argument("--dias", nargs="+", type=int, default=[1, 2, 3, 4, 5, 6, 7],
                    help="horizontes em dias (1..7)")
    ap.add_argument("--saida", default="saida_previsao", help="pasta de saída")
    ap.add_argument("--margem", type=float, default=1.0,
                    help="folga em graus ao redor da região no recorte da imagem")
    ap.add_argument("--recortar", action="store_true",
                    help="limita o preenchimento ao polígono da região")
    ap.add_argument("--somente", choices=["dia", "acumulado"], default=None,
                    help="gera só um dos dois tipos")
    args = ap.parse_args()

    faltando = [c for c in args.kml if not os.path.exists(c)]
    if faltando:
        sys.exit(f"KML não encontrado: {', '.join(faltando)}")

    regioes = ler_kmls(args.kml)
    if not regioes:
        sys.exit(f"Nenhum polígono lido de {', '.join(args.kml)}")
    print(f"Total: {len(regioes)} região(ões) disponível(is) para busca")

    regiao = None
    extent = None
    if args.regiao:
        regiao = selecionar_regiao(regioes, args.regiao)
        if regiao is None:
            exemplos = ", ".join(sorted(r["nome"] for r in regioes)[:12])
            sys.exit(f"Região '{args.regiao}' não encontrada. Ex.: {exemplos} ...")
        lo0, la0, lo1, la1 = bbox_regiao(regiao)
        m = args.margem
        extent = (lo0 - m, lo1 + m, la0 - m, la1 + m)
        print(f"Região: {regiao['nome']}  bbox={lo0:.2f},{la0:.2f}..{lo1:.2f},{la1:.2f}")
    else:
        print("Modo Brasil inteiro (sem recorte de região)")

    os.makedirs(args.saida, exist_ok=True)
    tmp_prefix = os.path.join(args.saida, "_tmp_tp")

    # rodada mais recente: usamos a data de hoje (UTC) só para rotular os
    # períodos; o dia real vem do valid_time do GRIB quando disponível.
    rodada_hoje = dt.date.today()
    sufixo = (_sem_acento(regiao["nome"]).replace(" ", "_").replace("/", "-")
              if regiao else "brasil")

    for dias in sorted(set(args.dias)):
        if dias < 1 or dias > 7:
            print(f"  (pulando dia {dias}: fora de 1..7)")
            continue
        step = dias * 24
        print(f"[{dias}d / {step}h] baixando ECMWF (tp)...")
        try:
            lons, lats, acum, diario, valido = obter_chuva(step, tmp_prefix)
        except Exception as e:
            print(f"  ERRO ao obter dados de {dias}d: {e}")
            continue

        # a data-base do período é o dia da rodada; se o GRIB trouxe valid_time,
        # ajusta a base para (valido - dias) para casar com a rodada real.
        base = rodada_hoje
        if valido is not None:
            base = (valido - dt.timedelta(days=dias)).date()

        tarefas = []
        if args.somente in (None, "acumulado"):
            tarefas.append(("acumulado", acum, "Acumulado (mm)",
                            periodo_acumulado(base, dias)))
        if args.somente in (None, "dia"):
            tarefas.append(("dia", diario, "Total (mm)",
                            periodo_diario(base, dias)))

        for tipo, campo, titulo, periodo_txt in tarefas:
            png = os.path.join(args.saida, f"ecmwf_chuva_{tipo}_{sufixo}_{dias}d.png")
            plotar(lons, lats, campo, titulo, periodo_txt, png,
                   extent=extent, regiao=regiao, todas_regioes=regioes,
                   recortar=args.recortar)
            print(f"  gerado: {png}")

    # limpeza de temporários
    for fn in os.listdir(args.saida):
        if fn.startswith("_tmp_"):
            _remover(os.path.join(args.saida, fn))
    print("Pronto.")


if __name__ == "__main__":
    main()
