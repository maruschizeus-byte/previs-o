#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gera figuras de previsão do ECMWF (Open Data) no estilo do painel "Total (mm)"
— colorbar à direita, título e período —, FOCANDO a imagem em um ESTADO ou
MESORREGIÃO lido de arquivo(s) KML.

Variáveis (escolha com --vars):
  chuva : precipitação — acumulado desde hoje 00 h e total de cada dia
  tmin  : temperatura mínima estimada do dia a 2 m (°C)
  tmax  : temperatura máxima estimada do dia a 2 m (°C)
  nuvem : média diária da cobertura total de nuvens (%)

Todos os dias usam 00 h a 00 h seguinte em America/Sao_Paulo (Brasília).
--dias 0 1 2 3 4 5 6 7 inclui hoje e os sete dias seguintes. Os estados são
sempre gerados. As mesorregiões mostram as cidades do CSV informado.

Estratégia: para cada horizonte (dia), o dado do ECMWF é baixado UMA vez e
depois recortado para TODAS as regiões pedidas. Mín e máx de temperatura saem
dos mesmos sub-passos de 2t (baixados uma vez).

USO
    pip install ecmwf-opendata xarray cfgrib numpy matplotlib
    # (sistema) eccodes p/ o cfgrib ler GRIB:  apt-get install libeccodes0

    python gerar_previsao_regiao.py --regioes "SP;MG;3105" \
        --kml estados.kml mesorregioes.kml --vars chuva tmin tmax nuvem
    python gerar_previsao_regiao.py --regiao SP --kml estados.kml --vars chuva
    python gerar_previsao_regiao.py --kml estados.kml --dias 1 2 3
    python gerar_previsao_regiao.py --todas-meso --todos-estados \
        --kml estados.kml mesorregioes.kml --fundo estados.kml \
        --kml-meso mesorregioes.kml --somente-estrutura

A busca da região casa (sem acento/caixa) contra o <name> do Placemark e
contra qualquer campo de ExtendedData (SimpleData/Data) — então nome, sigla
ou CÓDIGO IBGE funcionam se estiverem no KML.
"""

import os
import sys
import time
import json
import hashlib
import math
import argparse
import re
import unicodedata
import datetime as dt
import xml.etree.ElementTree as ET
from zoneinfo import ZoneInfo

import numpy as np

# =========================================================================
# CONFIGURAÇÕES
# =========================================================================
BRASILIA = ZoneInfo("America/Sao_Paulo")
UTC = dt.timezone.utc
DIAS_SEMANA = ["seg", "ter", "qua", "qui", "sex", "sáb", "dom"]
STEPS_ECMWF = tuple(range(0, 145, 3)) + tuple(range(150, 361, 6))
CIDADES_PADRAO = "Cidades_principais_137_mesorregioes_Brasil.csv"

# Domínio de reserva. O domínio efetivo cobre todos os enquadramentos + borda.
# Os campos globais são baixados uma vez por parâmetro/passo e recortados ao ler.
LON_MIN, LON_MAX = -82.0, -25.0
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


def _caminho_kml(caminho):
    return os.path.normcase(os.path.realpath(os.path.expanduser(caminho)))


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
            # Exportações do QGIS/IBGE podem ter um nome genérico no Placemark.
            # O nome oficial da mesorregião evita pastas repetidas/sem nome.
            for campo in ("nm_meso", "nm_mesorregiao", "nome_meso", "mesorregiao"):
                if campos.get(campo):
                    nome = campos[campo]
                    break
            if not nome:
                for campo in ("nm_uf", "nome", "name", "cd_meso", "cd_geocme"):
                    if campos.get(campo):
                        nome = campos[campo]
                        break
            if not nome:
                raise ValueError(f"Polígono sem nome ou código em {caminho}. "
                                 "Preencha o nome de cada região no KML.")
            chaves.add(_sem_acento(nome))
            regioes.append({"nome": nome, "chaves": chaves,
                            "campos": campos, "poligonos": poligonos,
                            "arquivo": os.path.basename(caminho),
                            "caminho": _caminho_kml(caminho)})
    return regioes


def ler_kmls(caminhos):
    todas = []
    for c in caminhos:
        lidas = ler_kml(c)
        print(f"KML: {len(lidas)} região(ões) em {c}")
        if not lidas:
            raise ValueError(f"Nenhum polígono de região encontrado em {c}. "
                             "Use um KML com Placemarks e polígonos; "
                             "pontos ou links para outros arquivos não bastam.")
        todas.extend(lidas)
    return todas


def carregar_geografia(caminhos, caminhos_fundo=None, caminhos_meso=None):
    """Valida os arquivos e identifica estados/mesorregiões independentemente."""
    fundo = {_caminho_kml(c) for c in (caminhos_fundo or [])}
    meso = {_caminho_kml(c) for c in (caminhos_meso or [])}
    if fundo & meso:
        raise ValueError("O mesmo KML foi indicado em --fundo e --kml-meso. "
                         "Informe arquivos separados para estados e mesorregiões.")
    arquivos = list(dict.fromkeys(_caminho_kml(c) for c in
                    list(caminhos) + list(caminhos_fundo or []) +
                    list(caminhos_meso or [])))
    ausentes = [c for c in arquivos if not os.path.isfile(c)]
    if ausentes:
        raise ValueError("KML não encontrado: " + "; ".join(ausentes) +
                         ". Confira o caminho e as letras maiúsculas/minúsculas "
                         "no repositório. Nenhuma região será ignorada.")
    regioes = ler_kmls(arquivos)
    estados, mesos = [], []
    campos_meso = ("cd_meso", "cd_geocme", "cd_mesorregiao", "nm_meso",
                   "nm_mesorregiao", "nome_meso", "mesorregiao")
    for r in regioes:
        origem = r["caminho"]
        nome = _sem_acento(r["nome"])
        arquivo = _sem_acento(r["arquivo"])
        if origem in meso:
            tipo = "mesorregiao"
        elif origem in fundo:
            tipo = "estado"
        elif any(r["campos"].get(c) for c in campos_meso) or "meso" in arquivo:
            tipo = "mesorregiao"
        elif ("estado" in arquivo or nome in UF_NOME2SIGLA or
              r["nome"].upper() in UF_COD2SIGLA.values()):
            tipo = "estado"
        else:
            tipo = "mesorregiao"
        r["tipo"] = tipo
        (estados if tipo == "estado" else mesos).append(r)
    print(f"Geografia: {len(estados)} estado(s), {len(mesos)} mesorregião(ões).")
    return regioes, estados, mesos


def selecionar_regiao(regioes, alvo, avisar=True):
    a = _sem_acento(alvo)
    exatos = [r for r in regioes if a in r["chaves"]]
    if exatos:
        if len(exatos) > 1 and avisar:
            fontes = ", ".join(f"{r['nome']} ({r['arquivo']})" for r in exatos)
            print(f"  AVISO: '{alvo}' casou com {len(exatos)} regiões: {fontes}. "
                  f"Usando a primeira. Para desambiguar, use o código IBGE.")
        return exatos[0]
    # Busca aproximada. Siglas e códigos curtos (SP, MA, 31...) só valem exatos:
    # antes, qualquer nome começando com "ma" virava Maranhão sem aviso.
    if len(a) < 3:
        return None
    achado = next((r for r in regioes if any(len(k) >= 3 and (k.startswith(a) or a.startswith(k))
                                              for k in r["chaves"] if k)), None)
    if achado is None:
        achado = next((r for r in regioes if any(len(k) >= 3 and a in k for k in r["chaves"] if k)), None)
    if achado is not None and avisar:
        print(f"  AVISO: '{alvo}' não é um nome exato; usando {achado['nome']} ({achado['arquivo']}).")
    return achado


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
UF_NOMES = {
    "RO": "Rondônia", "AC": "Acre", "AM": "Amazonas", "RR": "Roraima",
    "PA": "Pará", "AP": "Amapá", "TO": "Tocantins", "MA": "Maranhão",
    "PI": "Piauí", "CE": "Ceará", "RN": "Rio Grande do Norte", "PB": "Paraíba",
    "PE": "Pernambuco", "AL": "Alagoas", "SE": "Sergipe", "BA": "Bahia",
    "MG": "Minas Gerais", "ES": "Espírito Santo", "RJ": "Rio de Janeiro",
    "SP": "São Paulo", "PR": "Paraná", "SC": "Santa Catarina",
    "RS": "Rio Grande do Sul", "MS": "Mato Grosso do Sul", "MT": "Mato Grosso",
    "GO": "Goiás", "DF": "Distrito Federal",
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
        if v.upper() in UF_COD2SIGLA.values():
            return v.upper()
    for k in ("nm_uf", "nome_uf", "estado", "nm_estado", "name_uf"):
        s = UF_NOME2SIGLA.get(_sem_acento(c.get(k, "")))
        if s:
            return s
    for k in ("cd_uf", "uf", "cd_geocuf", "cd_meso", "geocodigo", "codigo",
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
    if regiao.get("nome", "").upper() in UF_COD2SIGLA.values():
        return regiao["nome"].upper()
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
    if (regiao.get("tipo") == "estado" or
            (not regiao.get("tipo") and id(regiao) in ids_estados)):
        return uf_sigla(regiao, estados) or _pasta_segura(regiao["nome"])
    uf = uf_sigla(regiao, estados) or "SEM_UF"  # é mesorregião
    return os.path.join(uf, _pasta_segura(regiao["nome"]))


def preparar_pastas(alvos, saida, estados):
    """Deduplica alvos, detecta colisões e registra as pastas antes do download."""
    unicos = {}
    for alvo in alvos:
        sub, regiao, _extent, _cidades = alvo
        chave = unicodedata.normalize("NFC", sub).casefold()
        anterior = unicos.get(chave)
        if anterior is not None:
            if anterior[1] is regiao:
                continue
            raise ValueError(f"Regiões diferentes gerariam a mesma pasta: {sub}. "
                             "Use nomes únicos para as mesorregiões no KML.")
        unicos[chave] = alvo
    registros = []
    for sub, regiao, _extent, _cidades in unicos.values():
        pasta = os.path.join(saida, sub)
        os.makedirs(pasta, exist_ok=True)
        registro = {
            "nome": regiao["nome"] if regiao else "Brasil",
            "tipo": regiao["tipo"] if regiao else "brasil",
            "uf": uf_sigla(regiao, estados) if regiao else None,
            "arquivo_kml": regiao["arquivo"] if regiao else None,
            "pasta": sub.replace(os.sep, "/"),
        }
        if regiao and regiao.get("pontos"):
            registro["pontos"] = regiao["pontos"]
        # O Git não versiona pastas vazias. Este arquivo identifica a região;
        # sua presença não significa que os mapas já foram atualizados.
        with open(os.path.join(pasta, "regiao.json"), "w", encoding="utf-8") as fp:
            json.dump(registro, fp, ensure_ascii=False, indent=2)
            fp.write("\n")
        registros.append(registro)
        print(f"Pasta: {registro['tipo']} | {registro['nome']} -> {pasta}")
    with open(os.path.join(saida, "regioes.json"), "w", encoding="utf-8") as fp:
        json.dump({"total": len(registros), "regioes": registros}, fp,
                  ensure_ascii=False, indent=2)
        fp.write("\n")
    return list(unicos.values())


# =========================================================================
# CIDADES (pontos de referência)
# =========================================================================
def ler_cidades(caminho):
    """Lê pontos de cidade de um CSV (nome,lat,lon) ou de um KML de pontos.
    Retorna dicionários com nome, lat, lon e identificação da mesorregião."""
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
                    cidades.append({"nome": nome or "(cidade)", "lat": lat, "lon": lon})
    return cidades


def _cabecalho_cidade(nome):
    return "".join(c for c in _sem_acento(nome) if c.isalnum())


def _ler_cidades_csv(caminho):
    import csv
    with open(caminho, encoding="utf-8-sig", newline="") as fp:
        amostra = fp.read(4096)
        fp.seek(0)
        delim = ";" if amostra.count(";") > amostra.count(",") else ","
        linhas = [ln for ln in csv.reader(fp, delimiter=delim) if any(v.strip() for v in ln)]
    if not linhas:
        raise ValueError(f"CSV de cidades vazio: {caminho}")
    cab = [_cabecalho_cidade(c) for c in linhas[0]]
    def indice(*nomes):
        return next((i for i, c in enumerate(cab) if c in nomes), None)
    i_lat = indice("lat", "latitude", "y")
    i_lon = indice("lon", "lng", "long", "longitude", "x")
    tem_cab = i_lat is not None and i_lon is not None
    i_nome = indice("nome", "name", "cidade", "municipio", "cidadeprincipal") if tem_cab else 0
    i_cod = indice("cdmeso", "codmesorregiao", "codigomesorregiao", "cdgeocme", "cdmesorregiao", "idmesorregiao") if tem_cab else None
    i_meso = indice("mesorregiao", "nmmeso", "nomemesorregiao") if tem_cab else None
    i_uf = indice("uf", "siglauf") if tem_cab else None
    if not tem_cab:
        i_lat, i_lon = 1, 2
    def celula(linha, i):
        return linha[i].strip() if i is not None and i < len(linha) else ""
    cidades = []
    for num, ln in enumerate(linhas[1:] if tem_cab else linhas, 2 if tem_cab else 1):
        try:
            lat = float(celula(ln, i_lat).replace(",", "."))
            lon = float(celula(ln, i_lon).replace(",", "."))
            nome = celula(ln, i_nome)
            if not nome or not -90 <= lat <= 90 or not -180 <= lon <= 180:
                raise ValueError("nome ou coordenadas inválidos")
        except (ValueError, IndexError) as e:
            raise ValueError(f"Cidade inválida na linha {num} de {caminho}: {e}") from e
        codigo = celula(ln, i_cod)
        if codigo.endswith(".0"):
            codigo = codigo[:-2]
        cidades.append({"nome": nome, "lat": lat, "lon": lon, "cd_meso": codigo,
                        "mesorregiao": celula(ln, i_meso), "uf": celula(ln, i_uf).upper()})
    return cidades


def codigo_mesorregiao(regiao):
    for chave in ("cd_meso", "cd_geocme", "cd_mesorregiao", "codigo_mesorregiao", "geocodigo", "codigo"):
        codigo = regiao.get("campos", {}).get(chave, "").strip()
        if codigo.endswith(".0"):
            codigo = codigo[:-2]
        if len(codigo) == 4 and codigo.isdigit():
            return codigo
    nome = regiao.get("nome", "").strip()
    return nome if len(nome) == 4 and nome.isdigit() else None


def cidades_da_mesorregiao(regiao, cidades, estados):
    if regiao.get("tipo") != "mesorregiao":
        return []
    codigo = codigo_mesorregiao(regiao)
    candidatas = [c for c in cidades if codigo and c.get("cd_meso") == codigo]
    if not candidatas:
        uf = uf_sigla(regiao, estados)
        candidatas = [c for c in cidades if c.get("mesorregiao") and
                      _sem_acento(c["mesorregiao"]) == _sem_acento(regiao["nome"]) and
                      (not uf or not c.get("uf") or c["uf"] == uf)]
    if not candidatas:
        candidatas = [c for c in cidades if ponto_na_regiao(c["lon"], c["lat"], regiao)]
    if not candidatas:
        raise ValueError(f"Nenhuma cidade de referência para {regiao['nome']} "
                         f"(código {codigo or 'não informado'}). Confira o CSV de cidades.")
    return list({(c["nome"], c["lat"], c["lon"]): c for c in candidatas}.values())


def _parse_cidade_cli(txt):
    """'Nome,lat,lon' -> dicionário de cidade. Nome pode ter vírgula: os DOIS
    últimos campos são lat e lon."""
    partes = [p.strip() for p in txt.split(",")]
    if len(partes) < 3:
        raise ValueError(f"--cidade inválido: '{txt}' (use Nome,lat,lon)")
    lon = float(partes[-1].replace(",", "."))
    lat = float(partes[-2].replace(",", "."))
    nome = ",".join(partes[:-2]).strip() or "(cidade)"
    return {"nome": nome, "lat": lat, "lon": lon}


def ponto_na_regiao(lon, lat, regiao):
    """True se (lon,lat) cai dentro de algum polígono da região."""
    from matplotlib.path import Path
    for poly in regiao["poligonos"]:
        if len(poly) >= 3 and Path(poly).contains_point((lon, lat)):
            return True
    return False


# =========================================================================
# GRUPOS DE PONTOS (ex.: AMAGGI) — definidos em grupos.json
# =========================================================================
GRUPOS_PADRAO = "grupos.json"
POSICOES_LOGO = ("inferior-esquerda", "inferior-direita", "superior-esquerda", "superior-direita")


def ler_grupos(caminho):
    """{"AMAGGI": {"estado": "MT", "logo": "logoamaggi.png", "pontos": [{nome, lat, lon}]}}"""
    try:
        with open(caminho, encoding="utf-8") as fp:
            dados = json.load(fp)
    except json.JSONDecodeError as e:
        raise ValueError(f"{caminho} não é um JSON válido (linha {e.lineno}): {e.msg}")
    if not isinstance(dados, dict) or not dados:
        raise ValueError(f"{caminho} deve ter o formato {{\"NOME\": {{\"estado\": \"MT\", \"pontos\": [...]}}}}")
    grupos = {}
    for nome, cfg in dados.items():
        nome = str(nome).strip()
        if not nome or not isinstance(cfg, dict):
            raise ValueError(f"{caminho}: grupo '{nome}' precisa ser um objeto com estado e pontos")
        uf = str(cfg.get("estado", "")).strip().upper()
        if uf not in UF_NOMES:
            raise ValueError(f"grupo {nome}: 'estado' deve ser a sigla da UF (ex.: MT), veio '{uf}'")
        pontos, vistos = [], set()
        for i, p in enumerate(cfg.get("pontos") or [], 1):
            try:
                rotulo, lat, lon = str(p["nome"]).strip(), float(p["lat"]), float(p["lon"])
            except (KeyError, TypeError, ValueError):
                raise ValueError(f"grupo {nome}: o ponto {i} precisa de nome, lat e lon numéricos")
            if not rotulo:
                raise ValueError(f"grupo {nome}: o ponto {i} está sem nome")
            if not (-35 <= lat <= 6 and -75 <= lon <= -28):
                raise ValueError(f"grupo {nome}: {rotulo} fora do Brasil (lat {lat}, lon {lon}); "
                                 "confira se lat e lon não estão trocadas")
            if rotulo.casefold() in vistos:
                raise ValueError(f"grupo {nome}: o ponto {rotulo} aparece duas vezes")
            vistos.add(rotulo.casefold())
            pontos.append({"nome": rotulo, "lat": lat, "lon": lon})
        if not pontos:
            raise ValueError(f"grupo {nome}: nenhum ponto informado")
        logo = str(cfg.get("logo") or "").strip() or None
        logo_pos = str(cfg.get("logo_posicao") or "inferior-esquerda").strip()
        if logo_pos not in POSICOES_LOGO:
            raise ValueError(f"grupo {nome}: logo_posicao deve ser uma de {', '.join(POSICOES_LOGO)}")
        grupos[nome] = {"nome": nome, "uf": uf, "pontos": pontos, "logo": logo, "logo_pos": logo_pos,
                        "mesorregioes": bool(cfg.get("mesorregioes", True)),
                        # nomes escritos dentro das mesorregiões: desligado por padrão
                        "nomes_mesorregioes": bool(cfg.get("nomes_mesorregioes", False))}
    return grupos


def remover_grupos_orfaos(saida, grupos):
    """Apaga pastas de grupo (regiao.json com tipo "grupo") que não estão mais no grupos.json."""
    import shutil
    validos = {k.casefold() for k in grupos}
    try:
        pastas = sorted(os.listdir(saida))
    except OSError:
        return
    for nome in pastas:
        pasta = os.path.join(saida, nome)
        try:
            with open(os.path.join(pasta, "regiao.json"), encoding="utf-8") as fp:
                info = json.load(fp)
        except (OSError, ValueError):
            continue
        if isinstance(info, dict) and info.get("tipo") == "grupo" and str(info.get("nome", "")).casefold() not in validos:
            shutil.rmtree(pasta)
            print(f"Removida a pasta {nome}/: o grupo {info.get('nome')} não está mais em grupos.json.")


def _area_anel(anel):
    xy = np.asarray(anel, dtype=float)
    return 0.5 * abs(np.dot(xy[:, 0], np.roll(xy[:, 1], 1)) - np.dot(xy[:, 1], np.roll(xy[:, 0], 1)))


def ponto_para_rotulo(regiao, evitar=()):
    """Ponto bem dentro do maior polígono, longe da borda e dos pontos do grupo."""
    from matplotlib.path import Path
    anel = np.asarray(max(regiao["poligonos"], key=_area_anel), dtype=float)
    (lo0, la0), (lo1, la1) = anel.min(0), anel.max(0)
    gx, gy = np.meshgrid(np.linspace(lo0, lo1, 48), np.linspace(la0, la1, 48))
    cand = np.c_[gx.ravel(), gy.ravel()]
    cand = cand[Path(anel).contains_points(cand)]
    if not len(cand):
        return float(anel[:, 0].mean()), float(anel[:, 1].mean())
    borda = anel[:: max(1, len(anel) // 400)]
    a, b = borda[:-1], borda[1:]
    ab = b - a
    t = np.clip(((cand[:, None, :] - a) * ab).sum(-1) / np.maximum((ab ** 2).sum(-1), 1e-12), 0, 1)
    d_borda = np.sqrt((((a + t[..., None] * ab) - cand[:, None, :]) ** 2).sum(-1)).min(1)
    nota = d_borda
    if len(evitar):
        ev = np.asarray(evitar, dtype=float)
        nota = np.minimum(d_borda, 0.8 * np.sqrt(((cand[:, None, :] - ev) ** 2).sum(-1)).min(1))
    lon, lat = cand[int(np.argmax(nota))]
    return float(lon), float(lat)


# =========================================================================
# ECMWF: download e processamento (genérico por parâmetro)
# =========================================================================
# Fontes do ECMWF Open Data, em ordem inicial de preferência. Os espelhos em
# nuvem (aws/azure/google) não têm o limite de 500 conexões do portal
# principal. 'ecmwf' (a origem) fica por último, como reserva. Ajustável por
# --fonte. Durante a execução, a fonte que entrega um arquivo passa para o
# início da lista e é a primeira tentada no arquivo seguinte.
FONTES = ["aws", "azure", "google", "ecmwf"]

# Política de tentativas. Por padrão, o Client do ecmwf-opendata faz até 500
# tentativas com 120 s de pausa em CADA requisição (o .index consultado antes
# de cada arquivo, o HEAD e o GET). Com "503 Slow Down" o script ficava muitos
# minutos preso na mesma fonte antes de trocar. Agora a biblioteca desiste
# cedo, o script troca de fonte e, se nenhuma entregar, espera e dá outra volta
# — sempre pedindo a MESMA rodada.
TENTATIVAS_POR_REQUISICAO = 3              # por requisição, dentro da biblioteca
PAUSA_REQUISICAO_S = 5                     # pausa entre essas tentativas
PAUSA_FONTE_S = 300                        # fonte com 503/429/erro de rede vai p/ o fim da fila
PAUSAS_ENTRE_VOLTAS_S = (30, 60, 120, 240)  # esperas entre voltas completas nas fontes
TIMEOUT_HTTP_S = (20, 120)                 # (conectar, ficar sem receber bytes)
ORIGEM = "ecmwf"                           # os espelhos só copiam o que a origem publica


class DadoAusente(RuntimeError):
    """O arquivo não existe (404 ou índice sem o campo): rodada ainda não publicada."""


class FontesIndisponiveis(RuntimeError):
    """O arquivo pode existir, mas nenhuma fonte o entregou (503/429/rede)."""


_CLIENTES = {}        # fonte -> Client reaproveitado (sessão HTTP e token SAS do Azure)
_PAUSADA_ATE = {}     # fonte -> time.monotonic() até quando fica no fim da fila
_ULTIMA_FONTE = []    # última fonte que entregou (só para o log)
_AVISOS = set()


def intervalo_brasilia(data):
    """Dia civil [00:00, 00:00 seguinte) sempre em America/Sao_Paulo."""
    inicio = dt.datetime.combine(data, dt.time.min, tzinfo=BRASILIA)
    fim = dt.datetime.combine(data + dt.timedelta(days=1), dt.time.min, tzinfo=BRASILIA)
    return inicio, fim


def _utc(valor):
    if valor.tzinfo is None:
        valor = valor.replace(tzinfo=UTC)
    return valor.astimezone(UTC)


def _horas(rodada, instante):
    return (_utc(instante) - _utc(rodada)).total_seconds() / 3600.0


def vizinhos_step(horas):
    """Limites nativos; após 144 h, a meia-noite de Brasília exige interpolação."""
    if not 0 <= horas <= STEPS_ECMWF[-1]:
        raise ValueError(f"Horário fora da rodada: {horas:g} h")
    pos = int(np.searchsorted(STEPS_ECMWF, horas))
    acima = STEPS_ECMWF[pos]
    if math.isclose(acima, horas, abs_tol=1e-8):
        return acima, acima, 0.0
    abaixo = STEPS_ECMWF[pos - 1]
    return abaixo, acima, (horas - abaixo) / (acima - abaixo)


def _novo_cliente(fonte):
    """Client com tentativas internas curtas e timeout em todas as requisições."""
    import inspect
    import requests
    from ecmwf.opendata import Client

    class _ComTimeout(requests.adapters.HTTPAdapter):
        # A biblioteca não passa timeout: uma conexão parada travaria o job.
        def send(self, request, **kw):
            if kw.get("timeout") is None:
                kw["timeout"] = TIMEOUT_HTTP_S
            return super().send(request, **kw)

    extras = {"maximum_retries": TENTATIVAS_POR_REQUISICAO,
              "retry_after": PAUSA_REQUISICAO_S,
              # Um Retry-After do servidor pode pedir minutos; é melhor trocar de fonte.
              "use_server_retry_after": False}
    aceitos = inspect.signature(Client.__init__).parameters
    faltando = [k for k in extras if k not in aceitos]
    if faltando and "versao" not in _AVISOS:
        _AVISOS.add("versao")
        print(f"  AVISO: ecmwf-opendata sem {', '.join(faltando)}; as tentativas internas "
              "não serão limitadas. Atualize: pip install -U ecmwf-opendata")
    cliente = Client(source=fonte, model="ifs", resol="0p25", infer_stream_keyword=False,
                     **{k: v for k, v in extras.items() if k in aceitos})
    adaptador = _ComTimeout()
    cliente.session.mount("https://", adaptador)
    cliente.session.mount("http://", adaptador)
    return cliente


def _status_http(e):
    return getattr(getattr(e, "response", None), "status_code", None)


def _arquivo_ausente(e):
    """404 ou índice sem o campo: o arquivo não existe (ainda) nesta fonte."""
    if _status_http(e) == 404:
        return True
    return isinstance(e, ValueError) and "Cannot find index entries" in str(e)


def _resumo_erro(e):
    """Mensagem curta para o log, sem URLs com token SAS."""
    status = _status_http(e)
    if status is not None:
        return f"HTTP {status} {getattr(e.response, 'reason', '') or ''}".strip()
    texto = re.sub(r"\?\S*", "?…", str(e))
    return f"{type(e).__name__}: {texto[:160]}"


def _ordem_fontes():
    """Fontes livres na ordem de preferência; as pausadas vão para o fim."""
    agora = time.monotonic()
    livres = [f for f in FONTES if _PAUSADA_ATE.get(f, 0.0) <= agora]
    pausadas = sorted((f for f in FONTES if f not in livres), key=lambda f: _PAUSADA_ATE[f])
    return livres + pausadas


def baixar(param, step, grib_file, data_rodada, hora_rodada):
    """Baixa um campo de UMA rodada fixa (data e hora explícitas, nunca 'latest').

    Todas as fontes recebem exatamente a mesma rodada. A fonte que entrega
    passa a ser a primeira da fila; a que responde 503/429 ou falha na rede vai
    para o fim da fila por PAUSA_FONTE_S. Levanta DadoAusente quando o arquivo
    não existe e FontesIndisponiveis quando ele existe mas nenhuma fonte entregou.
    """
    kw = dict(type="fc", stream="oper", param=param, step=step,
              target=grib_file, date=data_rodada.strftime("%Y%m%d"), time=hora_rodada)
    rotulo = f"{param}/{step} h"
    voltas = len(PAUSAS_ENTRE_VOLTAS_S) + 1
    ultimo = None
    for volta in range(voltas):
        tentadas = _ordem_fontes()
        ausentes = []
        for fonte in tentadas:
            try:
                if fonte not in _CLIENTES:
                    _CLIENTES[fonte] = _novo_cliente(fonte)
                if os.path.exists(grib_file):
                    os.remove(grib_file)  # não retomar download parcial de outra fonte
                _CLIENTES[fonte].retrieve(**kw)
                if not os.path.isfile(grib_file) or os.path.getsize(grib_file) == 0:
                    raise OSError("arquivo baixado vazio")
            except Exception as e:
                ultimo = e
                if _arquivo_ausente(e):
                    ausentes.append(fonte)
                    print(f"    ({rotulo} ainda não está em {fonte}; tentando próxima)")
                else:
                    _PAUSADA_ATE[fonte] = time.monotonic() + PAUSA_FONTE_S
                    _CLIENTES.pop(fonte, None)  # recria sessão (e token SAS) depois
                    print(f"    (fonte {fonte}: {_resumo_erro(e)}; fica no fim da fila "
                          f"por {PAUSA_FONTE_S // 60} min, tentando próxima)")
                continue
            FONTES.remove(fonte)
            FONTES.insert(0, fonte)
            _PAUSADA_ATE.pop(fonte, None)
            if _ULTIMA_FONTE[-1:] != [fonte]:
                _ULTIMA_FONTE[:] = [fonte]
                print(f"    usando fonte {fonte} a partir de {rotulo}")
            return
        # Os espelhos só têm o que a origem publicou: 404 na origem basta.
        if len(ausentes) == len(tentadas) or ORIGEM in ausentes:
            raise DadoAusente(f"{rotulo} não publicado (ausente em "
                              f"{', '.join(ausentes)})") from ultimo
        if volta < voltas - 1:
            espera = PAUSAS_ENTRE_VOLTAS_S[volta]
            print(f"    {rotulo}: nenhuma fonte entregou; nova volta em {espera} s "
                  f"({volta + 2}/{voltas}), mesma rodada")
            time.sleep(espera)
    raise FontesIndisponiveis(f"{rotulo}: nenhuma fonte entregou após {voltas} voltas "
                              f"(último erro: {_resumo_erro(ultimo)})") from ultimo


def ler_grib(grib_file, param, dominio, rodada, step):
    """Valida rodada/horário, normaliza coordenadas e carrega o domínio completo."""
    import xarray as xr
    aliases = {"tp": ("tp",), "2t": ("t2m", "2t"), "tcc": ("tcc",)}
    with xr.open_dataset(grib_file, engine="cfgrib",
                         backend_kwargs={"indexpath": ""}) as ds:
        nome = next((n for n in aliases[param] if n in ds.data_vars), None)
        if nome is None:
            raise ValueError(f"Parâmetro {param} ausente no GRIB: {list(ds.data_vars)}")
        da = ds[nome].squeeze(drop=False)
        for coord, esperado in (("time", rodada),
                                ("valid_time", rodada + dt.timedelta(hours=step))):
            if coord not in da.coords:
                raise ValueError(f"GRIB sem {coord}; não é possível validar o período")
            valor = np.asarray(da.coords[coord].values)
            if valor.size != 1:
                raise ValueError(f"Mais de um {coord} no GRIB")
            recebido = valor.reshape(-1)[0].astype("datetime64[s]").astype(dt.datetime)
            if _utc(recebido) != _utc(esperado):
                raise ValueError(f"GRIB incompatível: {coord}={recebido}, esperado={esperado}")
        da = da.assign_coords(longitude=(((da.longitude + 180) % 360) - 180))
        da = da.sortby("longitude").sortby("latitude", ascending=False)
        lo0, lo1, la0, la1 = dominio
        rec = da.sel(latitude=slice(la1, la0), longitude=slice(lo0, lo1))
        rec = rec.transpose("latitude", "longitude")
        lons = np.asarray(rec.longitude.values, dtype="float64")
        lats = np.asarray(rec.latitude.values, dtype="float64")
        arr = np.asarray(rec.values, dtype="float32").copy()
        unidades = str(da.attrs.get("units", "")).lower()
    if lons.size < 2 or lats.size < 2 or not np.isfinite(arr).all():
        raise ValueError("Grade vazia ou com dados ausentes no domínio solicitado")
    if param == "tp":
        if unidades not in ("mm", "kg m**-2", "kg m-2"):
            arr *= 1000.0
    elif param == "2t":
        if unidades not in ("c", "°c", "degc", "celsius"):
            arr -= 273.15
    elif param == "tcc":
        if unidades not in ("%", "percent", "percentage"):
            arr *= 100.0
        arr = np.clip(arr, 0.0, 100.0)
    return lons, lats, arr


def dominio_dos_alvos(alvos):
    """Uma célula extra em cada lado garante preenchimento até a moldura."""
    exts = [alvo[2] for alvo in alvos if alvo[2] is not None]
    if not exts:
        return (LON_MIN, LON_MAX, LAT_MIN, LAT_MAX)
    return (math.floor(min(e[0] for e in exts) * 4) / 4 - 0.5,
            math.ceil(max(e[1] for e in exts) * 4) / 4 + 0.5,
            math.floor(min(e[2] for e in exts) * 4) / 4 - 0.5,
            math.ceil(max(e[3] for e in exts) * 4) / 4 + 0.5)


class CamposECMWF:
    """Cache por rodada, resolução, domínio, parâmetro e passo; campos reutilizados."""
    def __init__(self, rodada, dominio, cache_dir=None):
        self.rodada = _utc(rodada)
        self.dominio = tuple(dominio)
        self.memoria = {}
        self.lons = self.lats = None
        assinatura = hashlib.sha256(json.dumps(self.dominio).encode()).hexdigest()[:12]
        self.pasta = (os.path.join(cache_dir, "brasilia_v1",
                      self.rodada.strftime("%Y%m%dT%H00Z"), assinatura) if cache_dir else None)
        if self.pasta:
            os.makedirs(self.pasta, exist_ok=True)

    def nativo(self, param, step):
        chave = (param, int(step))
        if chave in self.memoria:
            return self.memoria[chave]
        cp = os.path.join(self.pasta, f"{param}_{step:03d}.npz") if self.pasta else None
        campo = None
        if cp and os.path.isfile(cp):
            try:
                with np.load(cp, allow_pickle=False) as c:
                    if (str(c["rodada"]) != self.rodada.isoformat() or
                            int(c["step"]) != step or str(c["param"]) != param):
                        raise ValueError("metadados de cache incompatíveis")
                    campo = (c["lons"].copy(), c["lats"].copy(), c["dados"].copy())
            except (ValueError, OSError, KeyError):
                campo = None
        if campo is None:
            import tempfile
            with tempfile.TemporaryDirectory(prefix="ecmwf_") as tmp:
                grib = os.path.join(tmp, "campo.grib2")
                baixar(param, step, grib, self.rodada.date(), self.rodada.hour)
                campo = ler_grib(grib, param, self.dominio, self.rodada, step)
            if cp:
                temporario = cp + ".tmp.npz"
                np.savez_compressed(temporario, lons=campo[0], lats=campo[1], dados=campo[2],
                                    rodada=self.rodada.isoformat(), step=step, param=param)
                os.replace(temporario, cp)
        lo, la, arr = campo
        if (len(lo) < 2 or len(la) < 2 or arr.shape != (len(la), len(lo)) or
                not np.isfinite(arr).all()):
            raise ValueError(f"Campo inválido: {param}/{step} h")
        if self.lons is not None and (not np.array_equal(lo, self.lons) or
                                      not np.array_equal(la, self.lats)):
            raise ValueError("Grades diferentes na mesma rodada")
        self.lons, self.lats = lo, la
        self.memoria[chave] = arr
        return arr

    def em(self, param, instante):
        h = _horas(self.rodada, instante)
        a, b, peso = vizinhos_step(h)
        primeiro = self.nativo(param, a)
        return primeiro if a == b else primeiro * (1 - peso) + self.nativo(param, b) * peso


def escolher_rodada(hoje, dias, vars_sel, dominio, cache_dir=None, rodada_fixa=None):
    """Mais recente 00/12 UTC iniciada antes de 00 h BRT de hoje e disponível."""
    inicio, _ = intervalo_brasilia(hoje)
    _, fim = intervalo_brasilia(hoje + dt.timedelta(days=max(dias)))
    primeira = dt.datetime.combine(hoje, dt.time.min, tzinfo=UTC)
    candidatas = ([_utc(rodada_fixa)] if rodada_fixa else
                  [primeira - dt.timedelta(hours=12 * i) for i in range(4)])
    parametro = "tp" if "chuva" in vars_sel else ("2t" if set(vars_sel) & {"tmin", "tmax"} else "tcc")
    erros = []
    for rodada in candidatas:
        if rodada > _utc(inicio) or rodada.hour not in (0, 12) or rodada.minute or rodada.second:
            erros.append(f"{rodada.isoformat()}: não cobre o início do dia ou não é rodada 00/12 UTC")
            continue
        print(f"Conferindo rodada {rodada.isoformat()}...")
        try:
            campos = CamposECMWF(rodada, dominio, cache_dir)
            _, ultimo_step, _ = vizinhos_step(_horas(rodada, fim))
            campos.nativo(parametro, ultimo_step)
            return campos
        except FontesIndisponiveis as e:
            # Servidor ocupado não é rodada inexistente: não recua para uma
            # previsão mais antiga. O cache guarda o que já foi baixado.
            raise RuntimeError(f"rodada {rodada.isoformat()} não pôde ser conferida porque "
                               f"as fontes não responderam ({e}). Rode de novo em alguns "
                               "minutos.") from e
        except Exception as e:
            erros.append(f"{rodada.isoformat()}: {e}")
            print(f"  Rodada indisponível: {e}")
    raise RuntimeError("Nenhuma rodada disponível cobre os dias completos de Brasília. " + " | ".join(erros))


def _nos_periodo(rodada, inicio, fim):
    a, b = _horas(rodada, inicio), _horas(rodada, fim)
    return [a] + [s for s in STEPS_ECMWF if a < s < b] + [b]


def calcular_dia(campos, data, hoje, vars_sel):
    """Produtos do dia local completo; precipitação acumulada começa em hoje 00 h BRT."""
    inicio, fim = intervalo_brasilia(data)
    inicio_acum, _ = intervalo_brasilia(hoje)
    nos = _nos_periodo(campos.rodada, inicio, fim)
    chuva_interp = any(vizinhos_step(_horas(campos.rodada, t))[0] !=
                      vizinhos_step(_horas(campos.rodada, t))[1] for t in (inicio, fim))
    resultados = {}
    if "chuva" in vars_sel:
        fim_tp = campos.em("tp", fim)
        resultados["chuva_dia"] = np.maximum(fim_tp - campos.em("tp", inicio), 0)
        resultados["chuva_acumulado"] = np.maximum(fim_tp - campos.em("tp", inicio_acum), 0)
    if set(vars_sel) & {"tmin", "tmax"}:
        # Somente instantes dentro do dia: 00 h do dia seguinte é excluída.
        valores = [campos.em("2t", campos.rodada + dt.timedelta(hours=h)) for h in nos[:-1]]
        if "tmin" in vars_sel:
            resultados["tmin"] = np.minimum.reduce(valores)
        if "tmax" in vars_sel:
            resultados["tmax"] = np.maximum.reduce(valores)
    if "nuvem" in vars_sel:
        valores = [campos.em("tcc", campos.rodada + dt.timedelta(hours=h)) for h in nos]
        integral = sum((valores[i] + valores[i+1]) * (nos[i+1] - nos[i]) / 2
                       for i in range(len(nos) - 1))
        resultados["nuvem"] = np.clip(integral / (nos[-1] - nos[0]), 0, 100)
    metadados = {
        "data": data.isoformat(), "dia_semana": DIAS_SEMANA[data.weekday()],
        "inicio_brasilia": inicio.isoformat(), "fim_brasilia_exclusivo": fim.isoformat(),
        "inicio_utc": _utc(inicio).isoformat(), "fim_utc_exclusivo": _utc(fim).isoformat(),
        "inicio_acumulado_brasilia": inicio_acum.isoformat(),
        "chuva_limites_interpolados": chuva_interp,
        "temperatura": "mínima/máxima estimadas entre amostras dentro do dia",
        "nuvem": "média diária ponderada pelo tempo, integração trapezoidal",
    }
    return resultados, metadados


# =========================================================================
# FIGURA
# =========================================================================
def nome_no_mapa(regiao, estados=None):
    """Nome completo da área e UF, sem repetir uma sigla já presente no KML."""
    if regiao is None:
        return "Brasil"
    if regiao.get("nome_mapa"):  # grupo de pontos: "AMAGGI — Mato Grosso"
        return regiao["nome_mapa"]
    nome = " ".join(regiao["nome"].split())
    uf = uf_sigla(regiao, estados)
    if uf:
        if regiao.get("tipo") == "estado" or nome.upper() == uf:
            nome = UF_NOMES[uf]
        else:
            nome = re.sub(rf"\s*(?:[-–—/]\s*{uf}|\({uf}\))$", "", nome,
                          flags=re.IGNORECASE).strip()
        return f"{nome} — {uf}"
    return nome


def prefixo_png(regiao, estados, tipo, dias):
    """Prefixo estável por área/produto/horizonte, com caracteres portáveis."""
    if regiao is None:
        alvo = "brasil"
    else:
        uf = uf_sigla(regiao, estados)
        nome = UF_NOMES.get(uf, regiao["nome"]) if regiao.get("tipo") == "estado" else regiao["nome"]
        slug = re.sub(r"[^a-z0-9]+", "_", _sem_acento(nome)).strip("_") or "regiao"
        alvo = f"{uf}_{slug}" if uf else slug
    return f"ecmwf_{alvo}_{tipo}_{dias}d"


def nome_png(regiao, estados, tipo, dias, hoje):
    data = hoje + dt.timedelta(days=dias)
    periodo = (f"{hoje.isoformat()}_a_{data.isoformat()}" if tipo == "chuva_acumulado"
               else data.isoformat())
    return f"{prefixo_png(regiao, estados, tipo, dias)}_{periodo}.png"


def limpar_png_substituidos(pasta, prefixo, antigo, atual):
    """Retira só versões anteriores deste mapa após concluir toda a geração.

    Reconhece o nome legado exato e o prefixo gerado com datas ISO válidas.
    Outros produtos, horizontes, arquivos e subpastas são preservados.
    """
    removidos = 0
    for item in os.scandir(pasta):
        if item.name == atual or not item.is_file(follow_symlinks=False):
            continue
        substituir = item.name == antigo
        if item.name.startswith(prefixo + "_") and item.name.endswith(".png"):
            datas = item.name[len(prefixo) + 1:-4].split("_a_")
            if len(datas) in (1, 2):
                try:
                    substituir = all(dt.date.fromisoformat(s).isoformat() == s for s in datas)
                except ValueError:
                    pass
        if substituir:
            os.remove(item.path)
            removidos += 1
    return removidos


def _quebrar_texto_mapa(texto, largura_px, renderer, tamanho, peso="normal"):
    """Quebra por palavras conforme a largura renderizada, sem cortar nomes."""
    from matplotlib.font_manager import FontProperties
    fonte = FontProperties(size=tamanho, weight=peso)
    linhas, linha = [], ""
    for palavra in texto.split():
        candidata = f"{linha} {palavra}".strip()
        largura, _, _ = renderer.get_text_width_height_descent(candidata, fonte, False)
        if linha and largura > largura_px:
            linhas.append(linha)
            linha = palavra
        else:
            linha = candidata
    return "\n".join(linhas + [linha])


def _caminho_poligono(regiao):
    from matplotlib.path import Path
    verts, codes = [], []
    for poly in regiao["poligonos"]:
        for i, (lon, lat) in enumerate(poly):
            verts.append((lon, lat))
            codes.append(Path.MOVETO if i == 0 else Path.LINETO)
        codes[-1] = Path.CLOSEPOLY
    return Path(verts, codes)


def _add_logo(fig, ax, caminho, pos, escala, alpha, fundo=True):
    """Sobrepõe a logo (PNG) num canto do mapa, preservando a proporção.
    'escala' = largura da logo como fração da largura do eixo.
    'fundo'  = desenha uma caixa branca atrás (útil p/ logo com transparência)."""
    import matplotlib.image as mpimg
    from matplotlib.offsetbox import OffsetImage, AnnotationBbox
    try:
        img = mpimg.imread(caminho)
    except Exception as e:
        print(f"    (aviso: não consegui ler a logo {caminho}: {e})")
        return None
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
    bboxprops = dict(facecolor="white", edgecolor="#c8c8c8", linewidth=0.8) if fundo else None
    ab = AnnotationBbox(oi, (x, y), xycoords="axes fraction",
                        box_alignment=ba, frameon=fundo, pad=0.5,
                        bboxprops=bboxprops, zorder=10)
    ax.add_artist(ab)
    return ab


def _sobreposicao(a, b):
    w = min(a.x1, b.x1) - max(a.x0, b.x0)
    h = min(a.y1, b.y1) - max(a.y0, b.y0)
    return w * h if w > 0 and h > 0 else 0.0


def _rotular_grupo(ax, renderer, grupo, obstaculos=()):
    """Triângulos nos pontos do grupo e rótulos que desviam uns dos outros e das logos."""
    import matplotlib.patheffects as pe
    from matplotlib.transforms import Bbox
    eixo = ax.get_window_extent(renderer)
    ocupado, disp = list(obstaculos), []
    for p in grupo["pontos"]:
        ax.plot(p["lon"], p["lat"], linestyle="none", marker="^", markersize=9,
                markerfacecolor="#111111", markeredgecolor="white", markeredgewidth=1.2, zorder=8)
        x, y = ax.transData.transform((p["lon"], p["lat"]))
        disp.append((x, y))
        ocupado.append(Bbox.from_extents(x - 9, y - 8, x + 9, y + 10))
    for nome, lon, lat in grupo.get("rotulos_contornos") or []:
        t = ax.annotate(nome, (lon, lat), ha="center", va="center", fontsize=8.5, fontstyle="italic",
                        color="#3a454c", zorder=5, linespacing=1.1)
        t.set_path_effects([pe.withStroke(linewidth=2.2, foreground="white")])
        ocupado.append(t.get_window_extent(renderer))
    candidatos = [(7, 4, "left", "bottom"), (7, -4, "left", "top"), (-7, 4, "right", "bottom"),
                  (-7, -4, "right", "top"), (0, 10, "center", "bottom"), (0, -10, "center", "top"),
                  (12, 0, "left", "center"), (-12, 0, "right", "center")]
    halo = [pe.withStroke(linewidth=2.8, foreground="white")]
    # Os pontos com vizinhos mais próximos escolhem posição primeiro.
    xy = np.asarray(disp)
    vizinhos = ((np.hypot(*(xy[:, None, :] - xy[None, :, :]).transpose(2, 0, 1)) < 60).sum(1)
                if len(xy) else [])
    for i in sorted(range(len(disp)), key=lambda k: -vizinhos[k]):
        p, melhor = grupo["pontos"][i], None
        for dx, dy, ha, va in candidatos:
            t = ax.annotate(p["nome"], (p["lon"], p["lat"]), xytext=(dx, dy), textcoords="offset points",
                            ha=ha, va=va, fontsize=11, fontweight="bold", color="#111111", zorder=9)
            t.set_path_effects(halo)
            bb = t.get_window_extent(renderer).padded(2)
            fora = bb.width * bb.height - _sobreposicao(bb, eixo)
            custo = sum(_sobreposicao(bb, o) for o in ocupado) + 3 * fora
            if melhor is None or custo < melhor[0]:
                if melhor is not None:
                    melhor[1].remove()
                melhor = (custo, t, bb)
            else:
                t.remove()
            if custo == 0:
                break
        ocupado.append(melhor[2])


def plotar(lons, lats, dados, titulo, periodo_txt, png_path, faixas,
           cor_acima=None, cor_abaixo=None, extend="max",
           extent=None, regiao=None, fundo=None, recortar=False, cidades=None,
           logo=None, logo_pos="inferior-direita", logo_escala=0.16, logo_alpha=1.0,
           logo_fundo=True, rodape=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    # Desenha somente a grade necessária, incluindo as células que cobrem
    # a moldura. Não amplia o enquadramento para além dos dados disponíveis.
    if extent is not None:
        if (extent[0] < min(lons) or extent[1] > max(lons) or
                extent[2] < min(lats) or extent[3] > max(lats)):
            raise ValueError("O domínio dos dados não cobre a moldura do mapa")
        ix0 = max(0, int(np.searchsorted(lons, extent[0])) - 1)
        ix1 = min(len(lons), int(np.searchsorted(lons, extent[1], side="right")) + 1)
        dentro = np.flatnonzero((lats >= extent[2]) & (lats <= extent[3]))
        if dentro.size:
            iy0, iy1 = max(0, dentro[0] - 1), min(len(lats), dentro[-1] + 2)
        else:
            meio = int(np.argmin(np.abs(lats - (extent[2] + extent[3]) / 2)))
            iy0, iy1 = max(0, meio - 1), min(len(lats), meio + 2)
        lons, lats, dados = lons[ix0:ix1], lats[iy0:iy1], dados[iy0:iy1, ix0:ix1]
    levels, cmap, norm, ticks = construir_colormap(faixas, cor_acima, cor_abaixo)
    lon2d, lat2d = np.meshgrid(lons, lats)

    fig, ax = plt.subplots(figsize=(10, 9), dpi=130)
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
    # Grupo de pontos (ex.: AMAGGI): mesorregiões do estado em tracejado.
    grupo = regiao if regiao is not None and regiao.get("pontos") else None
    for r in (grupo or {}).get("contornos") or []:
        for poly in r["poligonos"]:
            xy = np.array(poly)
            ax.plot(xy[:, 0], xy[:, 1], color="#111111", linewidth=1.15, alpha=0.9,
                    linestyle=(0, (6, 3)))
    if regiao is not None:
        for poly in regiao["poligonos"]:
            xy = np.array(poly)
            ax.plot(xy[:, 0], xy[:, 1], color="black", linewidth=1.6)

    # pontos de cidade (referência p/ localizar) — sempre por cima do resto
    if cidades:
        import matplotlib.patheffects as pe
        halo = [pe.withStroke(linewidth=2.4, foreground="white")]
        for cidade in cidades:
            nome, lat, lon = cidade["nome"], cidade["lat"], cidade["lon"]
            ax.plot(lon, lat, marker="o", markersize=5.5, markerfacecolor="black",
                    markeredgecolor="white", markeredgewidth=0.9, zorder=6)
            perto_direita = extent is not None and lon > extent[0] + 0.7 * (extent[1] - extent[0])
            t = ax.annotate(nome, (lon, lat), xytext=(-5 if perto_direita else 5, 4),
                            ha="right" if perto_direita else "left",
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

    # A legenda acompanha a altura real do mapa, inclusive em enquadramentos
    # largos/baixos; evita faixas brancas externas impostas pela colorbar.
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    cax = make_axes_locatable(ax).append_axes("right", size="4.5%", pad=0.12)
    cb = fig.colorbar(cs, cax=cax, ticks=ticks, extend=extend)
    cb.ax.tick_params(labelsize=15)
    cb.set_ticklabels(["%g" % t for t in ticks])

    # O aspecto geográfico e a legenda definem a largura útil do cabeçalho.
    # Ajusta nomes compridos em linhas completas, sem cobrir o mapa.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    # Logos depois do desenho (a posição final do mapa já está definida) e
    # antes dos rótulos do grupo, para os rótulos desviarem delas.
    logos = []
    if logo:
        logos.append(_add_logo(fig, ax, logo, logo_pos, logo_escala, logo_alpha, fundo=logo_fundo))
    logo_grupo = (regiao or {}).get("logo_grupo")
    if logo_grupo:
        pos_grupo = logo_grupo[1]
        if logo and pos_grupo == logo_pos:  # mesmo canto da logo principal: vai para o lado oposto
            pos_grupo = pos_grupo.replace("direita", "X").replace("esquerda", "direita").replace("X", "esquerda")
        logos.append(_add_logo(fig, ax, logo_grupo[0], pos_grupo, logo_escala, logo_alpha, fundo=logo_fundo))
    if grupo:
        fig.canvas.draw()
        _rotular_grupo(ax, renderer, grupo, [a.get_window_extent(renderer).padded(4) for a in logos if a])
    largura_px = ax.get_window_extent(renderer).width
    nome_alvo = nome_no_mapa(regiao, fundo)
    cabecalho = _quebrar_texto_mapa(nome_alvo, largura_px, renderer, 14, "bold")
    subtitulo = _quebrar_texto_mapa(f"{titulo} · {periodo_txt}", largura_px,
                                   renderer, 11)
    if "\n" in subtitulo:
        subtitulo = (_quebrar_texto_mapa(titulo, largura_px, renderer, 11) + "\n" +
                     _quebrar_texto_mapa(periodo_txt, largura_px, renderer, 11))
    altura_subtitulo = (subtitulo.count("\n") + 1) * 11 * 1.3
    ax.set_title(cabecalho, fontsize=14, fontweight="bold",
                 pad=altura_subtitulo + 16, linespacing=1.15)
    ax.annotate(subtitulo, xy=(0.5, 1), xycoords="axes fraction",
                xytext=(0, 10), textcoords="offset points", ha="center", va="bottom",
                fontsize=11, color="#303844", linespacing=1.3)

    if rodape:
        ax.annotate(rodape, xy=(0, 0), xycoords="axes fraction",
                    xytext=(0, -13), textcoords="offset points",
                    fontsize=8, va="top", ha="left", linespacing=1.5)

    fig.savefig(png_path, bbox_inches="tight", facecolor="white",
                metadata={"Title": f"{nome_alvo} | {titulo} | {periodo_txt}",
                          "Source": "ECMWF", "Software": "gerar_previsao_regiao.py"})
    plt.close(fig)


# =========================================================================
# PERÍODO / RÓTULOS
# =========================================================================
def _fmt_dia(d):
    return f"{d.day}/{MESES_PT[d.month - 1]}/{str(d.year)[2:]}"


def periodo_acumulado(base, dias):
    fim = base + dt.timedelta(days=dias)
    return f"{_fmt_dia(base)} a {_fmt_dia(fim)}"


def periodo_diario(base, dias):
    data = base + dt.timedelta(days=dias)
    return f"{DIAS_SEMANA[data.weekday()]} — {_fmt_dia(data)}"


# =========================================================================
# CATÁLOGO DO PAINEL (painel_previsao.html lê <saida>/mapas.json)
# =========================================================================
VARIAVEIS_PAINEL = {  # id do produto -> (nome, unidade), na ordem do seletor
    "chuva_dia": ("Chuva do dia", "mm"),
    "chuva_acumulado": ("Chuva acumulada", "mm"),
    "tmin": ("Temperatura mínima", "°C"),
    "tmax": ("Temperatura máxima", "°C"),
    "nuvem": ("Nuvens — média diária", "%"),
}


def _caminho_painel(caminho):
    """Mesma regra do safePath() do painel: relativo, sem ':?#\\', termina em .png."""
    partes = caminho.split("/")
    return (len(caminho) < 1500 and caminho.lower().endswith(".png") and
            not re.search(r"[:\\?#\x00-\x1f]", caminho) and
            all(p and p not in (".", "..") for p in partes))


def escrever_catalogo(saida, alvos, estados, hoje, dias_dados, rodada, gerado_em):
    """Grava mapas.json só com PNGs que existem (escrita atômica).

    Numa execução parcial (ex.: só o AMAGGI), mantém as áreas que já estavam no
    catálogo do mesmo dia, para o painel não perder o restante.
    """
    regioes, ausentes, ignorados = [], 0, 0
    for subdir, regiao, _extent, _cids in alvos:
        pasta = subdir.replace(os.sep, "/")
        uf = uf_sigla(regiao, estados) if regiao else None
        mapas = {}
        for horizonte, d in dias_dados.items():
            for tipo, *_ in d["produtos"]:
                arquivo = nome_png(regiao, estados, tipo, horizonte, hoje)
                caminho = unicodedata.normalize("NFC", f"{pasta}/{arquivo}")
                if not _caminho_painel(caminho):
                    ignorados += 1
                elif not os.path.isfile(os.path.join(saida, subdir, arquivo)):
                    ausentes += 1
                else:
                    mapas.setdefault(tipo, {})[str(horizonte)] = caminho
        if not mapas:
            continue
        if regiao is None:
            item = {"id": "brasil", "nome": "Brasil", "tipo": "brasil"}
        else:
            nome = UF_NOMES.get(uf, regiao["nome"]) if regiao["tipo"] == "estado" else \
                " ".join(regiao["nome"].split())
            # O painel conhece brasil/estado/mesorregiao: o grupo entra como área da UF.
            item = {"id": pasta, "nome": nome,
                    "tipo": "mesorregiao" if regiao["tipo"] == "grupo" else regiao["tipo"],
                    "rotulo": nome_no_mapa(regiao, estados)}
            if regiao["tipo"] == "grupo":
                item["grupo"] = True
            if uf:
                item.update(uf=uf, estado=UF_NOMES.get(uf, uf))
        item["mapas"] = mapas
        regioes.append(item)
    if ignorados:
        print(f"  AVISO: {ignorados} mapa(s) fora do catálogo por caracteres inválidos no caminho.")
    destino = os.path.join(saida, "mapas.json")
    try:
        with open(destino, encoding="utf-8") as fp:
            anterior = json.load(fp)
    except (OSError, ValueError):
        anterior = None
    mantidas = 0
    if (isinstance(anterior, dict) and anterior.get("versao") == 1
            and anterior.get("data_base") == hoje.isoformat()):
        novos = {r["id"] for r in regioes}
        for r in anterior.get("regioes") or []:
            if not isinstance(r, dict) or r.get("id") in novos or not isinstance(r.get("mapas"), dict):
                continue
            mapas = {}
            for tipo, por_dia in r["mapas"].items():
                if tipo not in VARIAVEIS_PAINEL or not isinstance(por_dia, dict):
                    continue
                ok = {d: c for d, c in por_dia.items() if isinstance(c, str) and _caminho_painel(c)
                      and str(d).isdigit() and os.path.isfile(os.path.join(saida, *c.split("/")))}
                if ok:
                    mapas[tipo] = ok
            if mapas:
                regioes.append(dict(r, mapas=mapas))
                mantidas += 1
    tipos = {t for r in regioes for t in r["mapas"]}
    variaveis = [{"id": k, "nome": n, "unidade": u}
                 for k, (n, u) in VARIAVEIS_PAINEL.items() if k in tipos]
    horizontes = sorted({int(d) for r in regioes for por_dia in r["mapas"].values() for d in por_dia})
    dias = [{"id": str(d), "data": (hoje + dt.timedelta(days=d)).isoformat(),
             "rotulo": periodo_diario(hoje, d), "acumulado": periodo_acumulado(hoje, d)}
            for d in horizontes]
    catalogo = {"versao": 1, "data_base": hoje.isoformat(), "gerado_em": gerado_em,
                "rodada_utc": rodada.isoformat(), "fonte": "ECMWF Open Data (IFS 0,25°)",
                "fuso": "America/Sao_Paulo", "mapas_ausentes": ausentes + ignorados,
                "dias": dias, "variaveis": variaveis, "regioes": regioes}
    temporario = destino + ".tmp"
    with open(temporario, "w", encoding="utf-8") as fp:
        json.dump(catalogo, fp, ensure_ascii=False, separators=(",", ":"))
        fp.write("\n")
    os.replace(temporario, destino)
    extra = f"; {mantidas} já publicadas hoje foram mantidas" if mantidas else ""
    print(f"Catálogo do painel: {destino} ({len(regioes)} áreas, {len(dias)} dias{extra})")


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
    ap.add_argument("--kml-meso", nargs="+", default=None,
                    help="KML(s) que contêm as mesorregiões; identificação explícita")
    ap.add_argument("--fundo", nargs="*", default=None,
                    help="KML(s) usados como contorno de fundo. Padrão: os que "
                         "tiverem 'estado' no nome do arquivo.")
    ap.add_argument("--fonte", nargs="+", default=None,
                    choices=["aws", "azure", "google", "ecmwf"],
                    help="fonte(s) do ECMWF, em ordem inicial de preferência "
                         "(padrão: aws azure google ecmwf — espelhos primeiro). "
                         "A fonte que responder passa a ser a primeira.")
    ap.add_argument("--vars", nargs="+", default=VARS_VALIDAS, choices=VARS_VALIDAS,
                    help="variáveis a gerar (padrão: todas)")
    ap.add_argument("--dias", nargs="+", type=int, default=list(range(8)),
                    help="dias a partir de hoje em Brasília: 0=hoje, 1=amanhã, até 7")
    ap.add_argument("--data-base", type=dt.date.fromisoformat, default=None,
                    help="data local de referência AAAA-MM-DD; padrão: hoje em Brasília")
    ap.add_argument("--rodada", default=None,
                    help="rodada UTC opcional, ex.: 2026-09-23T00:00:00Z; padrão: automática")
    ap.add_argument("--saida", default="saida_previsao", help="pasta de saída")
    ap.add_argument("--margem", type=float, default=1.0,
                    help="folga em graus ao redor da região no recorte da imagem")
    ap.add_argument("--recortar", action="store_true",
                    help="limita o preenchimento ao polígono da região")
    ap.add_argument("--todas-meso", action="store_true",
                    help="gera TODAS as mesorregiões identificadas, "
                         "em pastas <saida>/<UF>/<mesorregião>/")
    ap.add_argument("--todos-estados", action="store_true", default=True,
                    help="compatibilidade: os estados são sempre gerados em <saida>/<UF>/")
    ap.add_argument("--sem-estados", action="store_true",
                    help="NÃO gerar os 27 estados (útil para rodar só um grupo, ex.: AMAGGI)")
    ap.add_argument("--grupo", nargs="+", default=None,
                    help="grupo(s) de pontos do grupos.json: nomes (ex.: AMAGGI), 'todos' ou 'nenhum'")
    ap.add_argument("--grupos", default=GRUPOS_PADRAO,
                    help=f"arquivo que define os grupos de pontos (padrão: {GRUPOS_PADRAO})")
    ap.add_argument("--somente-estrutura", action="store_true",
                    help="valida os KMLs e cria as pastas com sua identificação, "
                         "sem baixar previsão nem gerar mapas")
    ap.add_argument("--cidades", default=CIDADES_PADRAO,
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
    ap.add_argument("--logo-sem-fundo", action="store_true",
                    help="não desenhar a caixa branca atrás da logo")
    args = ap.parse_args()

    vars_sel = list(dict.fromkeys(args.vars))  # únicas, mantendo ordem
    dias_sel = sorted(set(args.dias))
    if any(d < 0 or d > 7 for d in dias_sel):
        ap.error("--dias aceita 0 (hoje) a 7, sempre pelo horário de Brasília")
    if not math.isfinite(args.margem) or args.margem < 0:
        ap.error("--margem deve ser um número maior ou igual a zero")
    hoje = args.data_base or dt.datetime.now(BRASILIA).date()
    rodada_fixa = None
    if args.rodada:
        try:
            rodada_fixa = _utc(dt.datetime.fromisoformat(args.rodada.replace("Z", "+00:00")))
        except ValueError:
            ap.error("--rodada inválida; use AAAA-MM-DDTHH:00:00Z")

    global FONTES
    if args.fonte:
        FONTES = list(dict.fromkeys(args.fonte))
    print(f"Fontes ECMWF (ordem de tentativa): {', '.join(FONTES)}")

    pedidos = list(args.regiao)
    if args.regioes:
        pedidos += [p.strip() for p in args.regioes.split(";") if p.strip()]
    pedidos_grupo = [g.strip() for item in (args.grupo or []) for g in re.split(r"[;,\s]+", item) if g.strip()]
    if all(g.casefold() == "nenhum" for g in pedidos_grupo):
        pedidos_grupo = []
    caminho_grupos = args.grupos
    if not os.path.isfile(caminho_grupos) and not os.path.isabs(caminho_grupos):
        caminho_grupos = os.path.join(os.path.dirname(__file__), caminho_grupos)
    grupos = None
    if pedidos_grupo or (pedidos and os.path.isfile(caminho_grupos)):
        if not os.path.isfile(caminho_grupos):
            sys.exit(f"ERRO: {args.grupos} não encontrado. Ele define os grupos de pontos (ex.: AMAGGI); "
                     "coloque-o na raiz do repositório ou ajuste --grupos.")
        try:
            grupos = ler_grupos(caminho_grupos)
        except (ValueError, OSError) as e:
            if pedidos_grupo:
                sys.exit(f"ERRO em {args.grupos}: {e}")
            print(f"  AVISO: {args.grupos} ignorado ({e}).")
    if grupos and pedidos:
        # Nome de grupo digitado em Regiões (ex.: AMAGGI) vale como grupo.
        nomes_grupo = {k.casefold() for k in grupos}
        em_regioes = [p for p in pedidos if p.strip().casefold() in nomes_grupo]
        if em_regioes:
            print(f"'{'; '.join(em_regioes)}' é grupo de pontos: gerando como grupo, não como região.")
            pedidos = [p for p in pedidos if p.strip().casefold() not in nomes_grupo]
            pedidos_grupo += [p.strip() for p in em_regioes]
    if not pedidos and args.sem_brasil and not args.todas_meso and args.sem_estados and not pedidos_grupo:
        sys.exit("Nada a gerar: sem regiões, grupos, Brasil, estados ou mesorregiões.")

    try:
        regioes, fundo_regioes, mesos = carregar_geografia(
            args.kml, args.fundo, args.kml_meso)
    except (ValueError, OSError, ET.ParseError) as e:
        sys.exit(f"ERRO nos KMLs: {e}")
    print(f"Total: {len(regioes)} região(ões) disponível(is) para busca")
    if args.todas_meso and not mesos:
        sys.exit("ERRO: nenhuma mesorregião identificada. Confira o conteúdo de "
                 "mesorregioes.kml e use --kml-meso mesorregioes.kml. "
                 "O arquivo deve conter um Placemark com polígono por região.")
    if args.todos_estados and not fundo_regioes:
        sys.exit("ERRO: nenhum estado identificado. Informe --fundo estados.kml.")
    if fundo_regioes:
        arqs = sorted({r["arquivo"] for r in fundo_regioes})
        print(f"Fundo (contorno de estados): {', '.join(arqs)} "
              f"({len(fundo_regioes)} polígono(s))")
    else:
        print("Fundo: sem estados. As mesorregiões mantêm sua classificação.")

    # cidades de referência
    cidades_arquivo = []
    if args.cidades:
        caminho_cidades = args.cidades
        if not os.path.isfile(caminho_cidades) and not os.path.isabs(caminho_cidades):
            caminho_cidades = os.path.join(os.path.dirname(__file__), caminho_cidades)
        if os.path.isfile(caminho_cidades):
            try:
                cidades_arquivo = ler_cidades(caminho_cidades)
            except (ValueError, OSError) as e:
                sys.exit(f"ERRO nas cidades: {e}")
            print(f"Cidades: {len(cidades_arquivo)} ponto(s) de {caminho_cidades}")
        elif args.todas_meso or any(r["tipo"] == "mesorregiao" for r in
                                  (selecionar_regiao(regioes, p, avisar=False) for p in pedidos) if r):
            sys.exit(f"ERRO: arquivo de cidades não encontrado: {args.cidades}. "
                     "Coloque o CSV de cidades no repositório ou ajuste --cidades.")
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

    grupos_sel = []
    if pedidos_grupo:
        if any(g.casefold() == "todos" for g in pedidos_grupo):
            grupos_sel = list(grupos.values())
        else:
            por_nome = {k.casefold(): v for k, v in grupos.items()}
            faltam = [g for g in pedidos_grupo if g.casefold() not in por_nome and g.casefold() != "nenhum"]
            if faltam:
                sys.exit(f"ERRO: grupo(s) {', '.join(faltam)} não existe(m) em {args.grupos}. "
                         f"Disponíveis: {', '.join(grupos)}.")
            grupos_sel = list({por_nome[g.casefold()]["nome"]: por_nome[g.casefold()]
                               for g in pedidos_grupo if g.casefold() in por_nome}.values())
        print(f"Grupos de pontos: {', '.join(g['nome'] for g in grupos_sel)} ({caminho_grupos})")

    alvos = []  # (subdir, regiao|None, extent|None, cidades)

    def _alvo_de_regiao(r):
        lo0, la0, lo1, la1 = bbox_regiao(r)
        m = args.margem
        ext = (lo0 - m, lo1 + m, la0 - m, la1 + m)
        cids = list(cidades_cli)
        try:
            cids += cidades_da_mesorregiao(r, cidades_arquivo, fundo_regioes)
        except ValueError as e:
            sys.exit(f"ERRO nas cidades: {e}")
        sub = subdir_do_alvo(r, fundo_regioes)
        return (sub, r, ext, cids)

    if not args.sem_brasil:
        # enquadra o Brasil no bbox dos estados (fundo) + margem, em vez do
        # domínio inteiro que é baixado (que vai muito além do Brasil).
        bb = bbox_uniao(fundo_regioes or regioes)
        ext_br = None
        if bb:
            lo0, la0, lo1, la1 = bb
            mb = min(max(args.margem, 0.15), 0.5)
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

    if args.todos_estados and not args.sem_estados:
        print(f"Todos os estados: {len(fundo_regioes)}")
        for r in fundo_regioes:
            sub, r, ext, cids = _alvo_de_regiao(r)
            alvos.append((sub, r, ext, cids))

    if args.todas_meso:
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

    for g in grupos_sel:
        uf = g["uf"]
        estado = next((r for r in fundo_regioes if uf_sigla(r, fundo_regioes) == uf), None)
        if estado is None:
            sys.exit(f"ERRO: grupo {g['nome']}: o estado {uf} não está no KML de estados.")
        contornos = [r for r in mesos if uf_sigla(r, fundo_regioes) == uf] if g["mesorregioes"] else []
        if g["mesorregioes"] and not contornos:
            print(f"  AVISO: grupo {g['nome']}: nenhuma mesorregião de {uf} no KML; sai só o contorno do estado.")
        evitar = [(p["lon"], p["lat"]) for p in g["pontos"]]
        rotulos = []
        for r in contornos if g["nomes_mesorregioes"] else []:
            nome = re.sub(rf"\s*(?:[-–—/]\s*{uf}|\({uf}\))$", "", " ".join(r["nome"].split()), flags=re.I)
            if len(nome) > 14 and " " in nome:  # duas linhas, quebrando no espaço mais central
                meio = min((i for i, c in enumerate(nome) if c == " "), key=lambda i: abs(i - len(nome) / 2))
                nome = nome[:meio] + "\n" + nome[meio + 1:]
            rotulos.append((nome, *ponto_para_rotulo(r, evitar)))
        logo_grupo = None
        if g["logo"]:
            caminho_logo = g["logo"]
            if not os.path.isfile(caminho_logo) and not os.path.isabs(caminho_logo):
                caminho_logo = os.path.join(os.path.dirname(caminho_grupos) or ".", g["logo"])
            if os.path.isfile(caminho_logo):
                logo_grupo = (caminho_logo, g["logo_pos"])
                print(f"Logo do grupo {g['nome']}: {caminho_logo} ({g['logo_pos']})")
            else:
                print(f"  AVISO: logo do grupo {g['nome']} não encontrada ({g['logo']}); seguindo sem ela. "
                      "Coloque o arquivo na raiz do repositório.")
        reg = dict(estado, nome=g["nome"], tipo="grupo", campos=dict(estado["campos"], sigla=uf),
                   logo_grupo=logo_grupo,
                   pontos=g["pontos"], contornos=contornos, rotulos_contornos=rotulos,
                   nome_mapa=f"{g['nome']} — {UF_NOMES[uf]}")
        xs = [lon for poly in estado["poligonos"] for lon, _ in poly] + [p["lon"] for p in g["pontos"]]
        ys = [lat for poly in estado["poligonos"] for _, lat in poly] + [p["lat"] for p in g["pontos"]]
        m = args.margem
        ext = (min(xs) - m, max(xs) + m, min(ys) - m, max(ys) + m)
        fora = [p["nome"] for p in g["pontos"] if not ponto_na_regiao(p["lon"], p["lat"], estado)]
        if fora:
            print(f"  AVISO: grupo {g['nome']}: {', '.join(fora)} fica(m) fora do contorno de {uf} "
                  "(entra(m) no mapa mesmo assim).")
        alvos.append((_pasta_segura(g["nome"]), reg, ext, []))
        print(f"Alvo: grupo {g['nome']} -> {_pasta_segura(g['nome'])}/  {UF_NOMES[uf]}, "
              f"{len(g['pontos'])} pontos, {len(contornos)} mesorregiões")

    if not alvos:
        sys.exit("Nenhum alvo válido — nada a gerar.")

    try:
        alvos = preparar_pastas(alvos, args.saida, fundo_regioes)
    except (ValueError, OSError) as e:
        sys.exit(f"ERRO ao preparar pastas: {e}")
    total_mesos = sum(1 for _, r, _, _ in alvos if r and r["tipo"] == "mesorregiao")
    print(f"Estrutura validada: {len(alvos)} pasta(s), "
          f"incluindo {total_mesos} mesorregião(ões).")
    if args.somente_estrutura:
        print("Somente estrutura: nenhum download ou mapa foi gerado.")
        return
    cache_dir = None if args.sem_cache else args.cache
    dominio = dominio_dos_alvos(alvos)
    print(f"Períodos: hoje={hoje.isoformat()}, dias={dias_sel}, fuso=America/Sao_Paulo")
    print("=== Fase 1: download e cálculo dos dias completos em Brasília ===")
    try:
        campos = escolher_rodada(hoje, dias_sel, vars_sel, dominio, cache_dir, rodada_fixa)
    except Exception as e:
        sys.exit(f"ERRO ao selecionar rodada: {e}")
    rodada_brt = campos.rodada.astimezone(BRASILIA)
    print(f"Rodada selecionada: {rodada_brt.isoformat()} (Brasília)")
    dias_dados = {}
    periodos = []
    definicoes = {
        "chuva_acumulado": ("Chuva acumulada (mm)", FAIXAS_CHUVA, CHUVA_ACIMA, None, "max"),
        "chuva_dia": ("Chuva do dia (mm)", FAIXAS_CHUVA, CHUVA_ACIMA, None, "max"),
        "tmin": ("Temperatura mínima (°C)", FAIXAS_TEMP, TEMP_ACIMA, TEMP_ABAIXO, "both"),
        "tmax": ("Temperatura máxima (°C)", FAIXAS_TEMP, TEMP_ACIMA, TEMP_ABAIXO, "both"),
        "nuvem": ("Nuvens — média diária (%)", FAIXAS_NUVEM, None, None, "neither"),
    }
    for dias in dias_sel:
        data = hoje + dt.timedelta(days=dias)
        print(f"[{dias}d] {periodo_diario(hoje, dias)}")
        try:
            valores, metadados = calcular_dia(campos, data, hoje, vars_sel)
        except Exception as e:
            # Falhar impede publicar somente parte dos dias solicitados.
            sys.exit(f"ERRO no dia {data}: {e}. Publicação interrompida.")
        produtos = []
        for tipo in definicoes:
            if tipo not in valores:
                continue
            titulo, faixas, c_a, c_b, ext_cb = definicoes[tipo]
            periodo = (periodo_acumulado(hoje, dias) if tipo == "chuva_acumulado"
                       else periodo_diario(hoje, dias))
            # O usuário pediu apenas o dia no mapa. Métodos e horários ficam
            # nos metadados de previsao.json, sem notas no rodapé das imagens.
            produtos.append((tipo, valores[tipo], titulo, periodo, faixas, c_a, c_b, ext_cb, ""))
        dias_dados[dias] = {"lons": campos.lons, "lats": campos.lats, "produtos": produtos}
        periodos.append(dict(metadados, horizonte_dias=dias, produtos=list(valores)))

    print("=== Fase 2: recortes e figuras (sem rede) ===")
    total_esperado = sum(len(alvos) * len(d["produtos"]) for d in dias_dados.values())
    por_tipo = {}
    for _, r, _, _ in alvos:
        t = "brasil" if r is None else r["tipo"]
        por_tipo[t] = por_tipo.get(t, 0) + 1
    rotulos_tipo = {"brasil": "Brasil", "estado": "estados", "mesorregiao": "mesorregiões", "grupo": "grupos"}
    partes = ", ".join(f"{rotulos_tipo.get(t, t)} {n}" for t, n in por_tipo.items())
    mapas_dia = len(next(iter(dias_dados.values()))["produtos"]) if dias_dados else 0
    print(f"A gerar {total_esperado} figura(s): {len(alvos)} áreas ({partes}) × {mapas_dia} mapas "
          f"× {len(dias_dados)} dias.")
    inicio = time.time()
    total = 0
    substituicoes = []
    for dias in sorted(dias_dados):
        d = dias_dados[dias]
        lons, lats, produtos = d["lons"], d["lats"], d["produtos"]
        for subdir, regiao, extent, cids in alvos:
            outdir = os.path.join(args.saida, subdir)
            for tipo, campo, titulo, per, faixas, c_a, c_b, ext_cb, rodape in produtos:
                arquivo_png = nome_png(regiao, fundo_regioes, tipo, dias, hoje)
                png = os.path.join(outdir, arquivo_png)
                plotar(lons, lats, campo, titulo, per, png, faixas,
                       cor_acima=c_a, cor_abaixo=c_b, extend=ext_cb,
                       extent=extent, regiao=regiao, fundo=fundo_regioes,
                       recortar=args.recortar, cidades=cids,
                       logo=logo, logo_pos=args.logo_pos,
                       logo_escala=args.logo_escala, logo_alpha=args.logo_alpha,
                       logo_fundo=not args.logo_sem_fundo, rodape=rodape)
                substituicoes.append((outdir, prefixo_png(regiao, fundo_regioes, tipo, dias),
                                       f"ecmwf_{tipo}_{dias}d.png", arquivo_png))
                total += 1
                if total % 50 == 0 or total == total_esperado:
                    seg = time.time() - inicio
                    taxa = total / seg if seg else 0
                    restam = (total_esperado - total) / taxa if taxa else 0
                    print(f"    {total}/{total_esperado} figuras | {seg:.0f}s | ~{restam:.0f}s restantes")
        print(f"  dia {dias}d concluído")
    if grupos:
        remover_grupos_orfaos(args.saida, grupos)
    # Catálogo primeiro: aponta só para os mapas novos, que já existem.
    gerado_em = dt.datetime.now(BRASILIA).isoformat(timespec="seconds")
    escrever_catalogo(args.saida, alvos, fundo_regioes, hoje, dias_dados,
                      campos.rodada, gerado_em)
    # Só substitui arquivos antigos depois de todos os novos mapas estarem prontos.
    removidos = sum(limpar_png_substituidos(*item) for item in substituicoes)
    if removidos:
        print(f"Substituídas {removidos} versões anteriores dos mapas gerados.")
    resumo = {"fuso": "America/Sao_Paulo", "data_base_brasilia": hoje.isoformat(),
              "rodada_utc": campos.rodada.isoformat(), "rodada_brasilia": rodada_brt.isoformat(),
              "gerado_em_brasilia": gerado_em,
              "total_mapas": total, "periodos": periodos,
              "identificacao_mapas": "Região/UF, variável/unidade, data e fonte ECMWF",
              "nota": "São previsões do modelo, inclusive para as horas de hoje já transcorridas."}
    with open(os.path.join(args.saida, "previsao.json"), "w", encoding="utf-8") as fp:
        json.dump(resumo, fp, ensure_ascii=False, indent=2)
        fp.write("\n")
    print(f"Fase 2 concluída: {total} figura(s) em {time.time() - inicio:.0f}s.")
    print("Pronto. Todos os dias calculados no horário de Brasília.")


if __name__ == "__main__":
    main()
