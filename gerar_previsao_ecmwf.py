#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gera os PNGs de previsão (ECMWF Open Data) para o WebGIS Chuva.

Variáveis geradas, para cada horizonte de 1 a 7 dias:
  - chuva       : precipitação total acumulada (mm)
  - nuvem       : cobertura total de nuvens (%)
  - temperatura : temperatura MÍNIMA do dia a 2 m (°C) — útil p/ alerta de geada
  - radiacao    : radiação solar incidente (kWh/m²/dia) — desacumulada

Cada variável vira uma série de PNGs com fundo transparente, coloridos com a
mesma escala usada no webgis, e uma entrada em previsao_meta.json com bounds,
step e label. O app.py serve /previsao/<arquivo>, e o webgis exibe como overlay.

RODAR FORA DO RENDER (na sua máquina ou num agendador), 1-2x ao dia:
    pip install ecmwf-opendata xarray cfgrib rioxarray numpy pillow
    python gerar_previsao_ecmwf.py

Requisitos de sistema: eccodes (para o cfgrib ler GRIB).
  - Ubuntu/Debian: sudo apt-get install libeccodes0
  - conda:         conda install -c conda-forge eccodes cfgrib

NOTAS SOBRE AS VARIÁVEIS DO ECMWF OPEN DATA:
  - tp   (chuva): acumulada desde o início da rodada, em metros. Como cada
    horizonte mostra "quanto choveu até esse dia", usamos o acumulado direto.
  - tcc  (nuvem): fração 0..1 instantânea. Vira %.
  - 2t   (temperatura): temperatura a 2 m em Kelvin, instantânea por step. Para
    a MÍNIMA do dia, baixamos todos os sub-steps do dia (3 em 3 h até 144 h, 6 em
    6 h depois) e tiramos o menor valor pixel a pixel. Vira °C.
  - ssrd (radiação): fluxo solar acumulado desde o início da rodada, em J/m².
    Para obter a radiação DE UM DIA, é preciso DESACUMULAR: subtrair o valor do
    step anterior (24 h antes) e converter para kWh/m²/dia. Por isso a radiação
    baixa dois steps por horizonte. No dia 1 (step 24 h) o "anterior" é o step 0,
    que vale zero (início da rodada) — então usamos o próprio acumulado.
"""

import os
import json
import datetime as dt

import numpy as np
from PIL import Image

# =========================
# CONFIGURAÇÕES
# =========================
# Horizontes em horas: 1 a 7 dias
STEPS = [24, 48, 72, 96, 120, 144, 168]

# Recorte aproximado da América do Sul
LON_MIN, LON_MAX = -82, -30
LAT_MIN, LAT_MAX = -60, 15

# Pasta de saída (servida pelo Flask em /previsao/).
# Sempre ao lado deste script, não importa de onde ele é executado.
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "previsao")

ALPHA = 200  # opacidade dos pixels coloridos (a opacidade fina é ajustada no webgis)

# MODO DE TESTE: desenha uma cruz em coordenadas conhecidas sobre cada PNG,
# para conferir visualmente o alinhamento. Ative com a variável de ambiente
# CRUZ_TESTE=1 ao rodar. Cada ponto é (lat, lon, rótulo).
CRUZ_TESTE = os.getenv("CRUZ_TESTE", "").strip() in ("1", "true", "True", "sim")
PONTOS_TESTE = [
    (-18.92946857632853, -48.276066905117396, "Uberlandia"),
]

# ---- Escalas de cor (limite_inferior, (R, G, B)). Abaixo do menor limite,
#      o pixel fica transparente. Devem casar com as legendas do webgis. ----

# Chuva (mm). Transparente abaixo de 0.2 mm.
ESCALA_CHUVA = [
    (0.2,  (168, 212, 255)),  # #a8d4ff
    (1,    (26, 75, 160)),    # #1a4ba0
    (5,    (45, 138, 62)),    # #2d8a3e
    (15,   (158, 196, 58)),   # #9ec43a
    (30,   (242, 194, 0)),    # #f2c200
    (50,   (232, 101, 27)),   # #e8651b
    (80,   (196, 30, 30)),    # #c41e1e
    (120,  (107, 30, 145)),   # #6b1e91
]

# Nuvem: cobertura total (%). Céu limpo (<10%) transparente.
ESCALA_NUVEM = [
    (10,  (210, 224, 235)),
    (30,  (170, 190, 205)),
    (50,  (130, 155, 175)),
    (70,  (95, 120, 142)),
    (90,  (60, 82, 105)),
]

# Temperatura MÍNIMA do dia (°C). Faixas focadas em geada (destaca < 0 e < 5°C).
# Deve casar com a legenda 'temperatura' do webgis.
ESCALA_TEMP = [
    (-100, (63, 0, 125)),     # #3f007d  (< 0°C, geada — cobre extremos frios)
    (0,    (94, 79, 162)),     # #5e4fa2  (0–3)
    (3,    (50, 136, 189)),    # #3288bd  (3–5)
    (5,    (102, 194, 165)),   # #66c2a5  (5–10)
    (10,   (171, 221, 164)),   # #abdda4  (10–15)
    (15,   (230, 245, 152)),   # #e6f598  (15–20)
    (20,   (254, 224, 139)),   # #fee08b  (20–25)
    (25,   (253, 174, 97)),    # #fdae61  (25–30)
    (30,   (244, 109, 67)),    # #f46d43  (> 30)
]

# Radiação solar (kWh/m²/dia). Rampa plasma, mesma da legenda do webgis.
# Transparente abaixo de 0.1 (basicamente sem sol / fora do domínio).
ESCALA_RAD = [
    (0.1, (13, 8, 135)),     # #0d0887  (< 1)
    (1,   (92, 1, 166)),     # #5c01a6
    (2,   (156, 23, 158)),   # #9c179e
    (3,   (204, 71, 120)),   # #cc4778
    (4,   (237, 121, 83)),   # #ed7953
    (5,   (253, 180, 47)),   # #fdb42f
    (6,   (247, 226, 37)),   # #f7e225
    (7,   (240, 249, 33)),   # #f0f921  (> 7)
]

# Cada variável: parâmetro ECMWF, como converter, escala, prefixo do arquivo e
# se é acumulada (precisa desacumular subtraindo o step anterior).
VARIAVEIS = {
    "chuva": {
        "param": "tp",       # total precipitation (acumulado, metros)
        "fator": 1000.0,     # m -> mm
        "minimo": 0.2,
        "escala": ESCALA_CHUVA,
        "prefixo": "ecmwf_chuva",
        "modo": "acumulado_total",   # usa o acumulado direto
    },
    "nuvem": {
        "param": "tcc",      # total cloud cover (0..1)
        "fator": 100.0,      # fração -> %
        "minimo": 10.0,
        "escala": ESCALA_NUVEM,
        "prefixo": "ecmwf_nuvem",
        "modo": "instantaneo",
    },
    "temperatura": {
        "param": "2t",       # 2 m temperature (Kelvin)
        "offset": -273.15,   # K -> °C (aplicado após o fator)
        "fator": 1.0,
        "minimo": -100.0,    # pinta tudo que tiver dado
        "escala": ESCALA_TEMP,
        "prefixo": "ecmwf_temperatura",
        "modo": "min_dia",   # mínima do dia (útil p/ geada): menor 2t entre os steps do dia
    },
    "temperatura_max": {
        "param": "2t",
        "offset": -273.15,
        "fator": 1.0,
        "minimo": -100.0,
        "escala": ESCALA_TEMP,   # mesma escala da mínima: permite comparar as duas
        "prefixo": "ecmwf_temperatura_max",
        "modo": "max_dia",   # máxima do dia: maior 2t entre os steps do dia
    },
    "radiacao": {
        "param": "ssrd",     # surface solar radiation downwards (J/m², acumulado)
        # J/m² por dia -> kWh/m²/dia: divide por 3_600_000 (1 kWh = 3.6e6 J)
        "fator": 1.0 / 3_600_000.0,
        "minimo": 0.1,
        "escala": ESCALA_RAD,
        "prefixo": "ecmwf_radiacao",
        "modo": "desacumular_dia",   # valor do dia = acum(step) - acum(step-24h)
    },
}


def baixar(param, step, grib_file, data_rodada=None, hora_rodada=0):
    """Baixa um passo do ECMWF Open Data.

    `data_rodada` (date) e `hora_rodada` (0 ou 12) pedem uma rodada ESPECÍFICA,
    em vez da mais recente. É o que permite reconstruir o histórico: para saber
    o que estava previsto para ontem, pega-se a rodada de anteontem/ontem.
    O Open Data mantém só os últimos dias — rodadas antigas não estão mais lá.
    """
    from ecmwf.opendata import Client
    client = Client(source="ecmwf")
    kw = dict(type="fc", stream="oper", param=param, step=step, target=grib_file)
    if data_rodada is not None:
        kw["date"] = data_rodada.strftime("%Y%m%d")
        kw["time"] = hora_rodada
    client.retrieve(**kw)


def ler_grib(grib_file, param):
    """Lê o GRIB (sem conversão) e ajusta longitude para -180..180 + recorta SA."""
    import xarray as xr
    ds = xr.open_dataset(grib_file, engine="cfgrib")
    # o nome da variável no dataset pode diferir do 'param' de download;
    # pega a primeira variável de dados se o nome exato não existir.
    if param in ds:
        da = ds[param]
    else:
        nome = list(ds.data_vars)[0]
        da = ds[nome]
    if float(da.longitude.max()) > 180:
        da = da.assign_coords(
            longitude=(((da.longitude + 180) % 360) - 180)
        ).sortby("longitude")
    rec = da.sel(latitude=slice(LAT_MAX, LAT_MIN), longitude=slice(LON_MIN, LON_MAX))
    lt = np.asarray(rec.latitude.values, dtype="float64")
    ln = np.asarray(rec.longitude.values, dtype="float64")
    if len(lt) > 1 and len(ln) > 1:
        print(f"      [grade] lat {lt[0]:.3f}..{lt[-1]:.3f} (n={len(lt)}, passo={abs(lt[1]-lt[0]):.3f}) | "
              f"lon {ln[0]:.3f}..{ln[-1]:.3f} (n={len(ln)}, passo={abs(ln[1]-ln[0]):.3f})")
    return rec


def _iso_de(rec, coord):
    """Lê uma coordenada de tempo do GRIB e devolve texto ISO (ou None).

    Serve para o meta guardar o horário REAL da rodada e da validade, em vez
    de o webgis deduzir a data somando dias — dedução que erra quando a rodada
    usada não é a 00z.
    """
    try:
        import numpy as _np
        v = rec.coords.get(coord)
        if v is None:
            return None
        val = v.values
        if getattr(val, "ndim", 0) > 0:
            val = val.reshape(-1)[0]
        return str(_np.datetime_as_string(_np.datetime64(val), unit="m"))
    except Exception:
        return None


# =========================
# ARQUIVO HISTÓRICO (para comparar previsto x medido depois)
# =========================
# Quantos dias de histórico manter. Sem limite, a pasta cresceria para sempre
# dentro do repositório (o histórico do Git nunca encolhe).
HIST_DIAS = int(os.getenv("HIST_DIAS", "45"))
HIST_DIR = os.path.join(OUT_DIR, "historico")
# O dia anterior é reconstruído automaticamente. Só a precipitação entra nesse
# preenchimento; temperatura, nuvem e radiação não são baixadas. O período é
# 00:00–23:59 de Brasília (03:00–03:00 UTC).
HIST_CHUVA_DIAS = max(0, int(os.getenv("HIST_CHUVA_DIAS", "1") or 0))


def arquivar_imagem(variavel, valido_ate_iso, rodada, png_path, bounds):
    """Guarda uma cópia do PNG com a DATA DE VALIDADE no nome.

    A comparação "previsto x medido" é feita olhando: a plataforma mostra esta
    imagem e desenha as estações por cima, na mesma escala de cor. Onde a
    estação destoa do fundo, a previsão errou. Simples e sem depender de
    coordenadas nem de amostrar a grade.
    """
    import shutil
    if not valido_ate_iso or not os.path.exists(png_path):
        return None
    os.makedirs(HIST_DIR, exist_ok=True)
    dia = str(valido_ate_iso)[:10]
    destino = os.path.join(HIST_DIR, f"{variavel}_{dia}.png")
    shutil.copyfile(png_path, destino)
    # meta mínima ao lado: bounds e rodada, para o mapa posicionar a imagem
    with open(os.path.join(HIST_DIR, f"{variavel}_{dia}.json"), "w", encoding="utf-8") as fp:
        json.dump({"variavel": variavel, "valido_para": dia, "rodada": rodada,
                   "arquivo": os.path.basename(destino), "bounds": bounds},
                  fp, ensure_ascii=False)
    print(f"   [historico] imagem de {dia} arquivada")
    return destino


def limpar_historico_antigo():
    """Remove arquivos de histórico mais velhos que HIST_DIAS."""
    if not os.path.isdir(HIST_DIR):
        return
    limite = (dt.date.today() - dt.timedelta(days=HIST_DIAS)).isoformat()
    removidos = 0
    for fn in os.listdir(HIST_DIR):
        if not (fn.endswith(".json") or fn.endswith(".png") or fn.endswith(".f32")):
            continue
        # nome no formato <variavel>_AAAA-MM-DD.json
        try:
            dia = fn.rsplit("_", 1)[1].rsplit(".", 1)[0]
        except Exception:
            continue
        if len(dia) == 10 and dia < limite:
            try:
                os.remove(os.path.join(HIST_DIR, fn))
                removidos += 1
            except OSError:
                pass
    if removidos:
        print(f"   [historico] {removidos} arquivo(s) antigo(s) removido(s)")


def preencher_historico(dias_atras):
    """Reconstrói o histórico dos últimos dias buscando rodadas passadas.

    Para o dia D, usa a rodada de D-1 (00z) no passo de 24h — exatamente o que
    estava previsto para aquele dia. Gera o PNG e o arquiva com a data de D.

    Limite: o ECMWF Open Data guarda apenas as rodadas recentes (poucos dias).
    Dias antigos demais falham; o script avisa e segue.
    """
    hoje = dt.date.today()
    for n in range(1, dias_atras + 1):
        alvo = hoje - dt.timedelta(days=n)
        rodada_dia = alvo - dt.timedelta(days=1)
        print(f"\n[preencher] dia {alvo} (rodada {rodada_dia} 00z)")
        for nome, cfg in VARIAVEIS.items():
            if nome not in ("temperatura", "temperatura_max"):
                continue
            destino = os.path.join(HIST_DIR, f"{nome}_{alvo.isoformat()}.png")
            if os.path.exists(destino):
                print(f"   {nome}: já existe, pulando")
                continue
            try:
                arr, recorte = valores_do_horizonte(
                    cfg, 24, os.path.join(OUT_DIR, f"_tmp_{cfg['param']}"),
                    data_rodada=rodada_dia)
                tmp_png = os.path.join(OUT_DIR, f"_tmp_hist_{nome}.png")
                bounds = gerar_png(arr, recorte, cfg["escala"], cfg["minimo"], tmp_png)
                arquivar_imagem(nome, alvo.isoformat(), f"{rodada_dia} 00z", tmp_png, bounds)
                _remover(tmp_png)
            except Exception as e:
                print(f"   {nome}: não foi possível ({e})")
        for fn in os.listdir(OUT_DIR):
            if fn.startswith("_cache_") or fn.startswith("_tmp_"):
                _remover(os.path.join(OUT_DIR, fn))


def arquivar_chuva_diaria(alvo, rodada_dia, arr, recorte):
    """Arquiva chuva diária prevista em PNG + grade numérica Float32.

    Os caminhos guardados no JSON são relativos a ``/previsao/`` para que o
    mesmo metadado sirva ao mapa e ao download em Excel.
    """
    os.makedirs(HIST_DIR, exist_ok=True)
    dia = alvo.isoformat()
    base = f"chuva_ecmwf_{dia}"
    png_path = os.path.join(HIST_DIR, base + ".png")
    grade_path = os.path.join(HIST_DIR, base + ".f32")
    meta_path = os.path.join(HIST_DIR, base + ".json")
    bounds = gerar_png(arr, recorte, ESCALA_CHUVA, 0.2, png_path)
    grade = salvar_grade_f32(arr, recorte, grade_path)
    grade["arquivo"] = "historico/" + os.path.basename(grade_path)
    fim = alvo + dt.timedelta(days=1)
    meta = {
        "modelo": "ECMWF",
        "variavel": "chuva",
        "unidade": "mm",
        "valido_para": dia,
        "periodo_brasilia": {
            "inicio": f"{dia}T00:00:00-03:00",
            "fim_exclusivo": f"{fim.isoformat()}T00:00:00-03:00",
        },
        "rodada": f"{rodada_dia.isoformat()} 12z",
        "arquivo": "historico/" + os.path.basename(png_path),
        "bounds": bounds,
        "grade": grade,
    }
    with open(meta_path, "w", encoding="utf-8") as fp:
        json.dump(meta, fp, ensure_ascii=False, indent=2)
    print(f"   [histórico chuva] ECMWF {dia} arquivado")


def preencher_historico_chuva(dias_atras):
    """Reconstrói somente chuva para os últimos dias completos de Brasília.

    Para o dia local D usa a rodada 12z de D-1 (09:00 em Brasília). O acumulado
    entre os passos 15h e 39h corresponde exatamente a D 00:00 até D+1 00:00.
    """
    cfg = VARIAVEIS["chuva"]
    hoje_brasilia = (dt.datetime.utcnow() - dt.timedelta(hours=3)).date()
    os.makedirs(HIST_DIR, exist_ok=True)
    for n in range(1, dias_atras + 1):
        alvo = hoje_brasilia - dt.timedelta(days=n)
        rodada_dia = alvo - dt.timedelta(days=1)
        base = os.path.join(HIST_DIR, f"chuva_ecmwf_{alvo.isoformat()}")
        if os.path.exists(base + ".json") and os.path.exists(base + ".f32"):
            print(f"   [histórico chuva] ECMWF {alvo}: já existe, pulando")
            continue
        print(f"\n[histórico chuva] ECMWF {alvo} (rodada {rodada_dia} 12z)")
        tmp = os.path.join(OUT_DIR, "_tmp_hist_chuva_ecmwf")
        try:
            arr_fim, recorte = valores_do_horizonte(
                cfg, 39, tmp, data_rodada=rodada_dia, hora_rodada=12
            )
            arr_ini, _ = valores_do_horizonte(
                cfg, 15, tmp, data_rodada=rodada_dia, hora_rodada=12
            )
            arr = np.maximum(0.0, np.asarray(arr_fim) - np.asarray(arr_ini))
            arquivar_chuva_diaria(alvo, rodada_dia, arr, recorte)
        except Exception as e:
            print(f"   [histórico chuva] ECMWF {alvo}: não foi possível ({e})")
        finally:
            for fn in os.listdir(OUT_DIR):
                if fn.startswith("_tmp_hist_chuva_ecmwf"):
                    _remover(os.path.join(OUT_DIR, fn))


def valores_do_horizonte(cfg, step, tmp_prefix, data_rodada=None, hora_rodada=0):
    """Baixa e devolve o array de valores já convertidos para a unidade final.

    Trata os três modos: instantâneo, acumulado total e desacumulação diária.
    Retorna (arr, recorte) — recorte é mantido para extrair lat/lon/bounds.
    """
    fator = cfg.get("fator", 1.0)
    offset = cfg.get("offset", 0.0)
    modo = cfg.get("modo", "instantaneo")

    grib_atual = f"{tmp_prefix}_{step}h.grib2"
    baixar(cfg["param"], step, grib_atual, data_rodada, hora_rodada)
    recorte = ler_grib(grib_atual, cfg["param"])
    arr = np.asarray(recorte.values, dtype="float32")

    if modo in ("min_dia", "max_dia"):
        # extremo do dia entre todos os sub-steps daquele dia.
        # min_dia -> menor valor (geada); max_dia -> maior valor (calor).
        # O dia N (step N*24) cobre as horas (N-1)*24+delta .. N*24.
        # ECMWF Open Data grátis: passo de 3h até 144h, de 6h depois.
        passo = 3 if step <= 144 else 6
        ini = step - 24 + passo          # primeiro sub-step dentro do dia
        sub_steps = list(range(ini, step + 1, passo))
        if step not in sub_steps:
            sub_steps.append(step)
        acumulado = arr.copy()           # já temos o step final; começa por ele
        for s in sub_steps:
            if s == step or s <= 0:
                continue
            # Nome por (parâmetro, sub-step) e NÃO por variável: a mínima e a
            # máxima do dia usam exatamente os mesmos arquivos (mesmo `2t`,
            # mesmos sub-passos). Reaproveitando, a máxima sai praticamente de
            # graça em vez de dobrar os downloads. A limpeza é feita no fim.
            # a chave inclui a rodada: no preenchimento retroativo, o mesmo
            # sub-step de rodadas diferentes tem conteúdo diferente
            _r = (data_rodada.strftime("%Y%m%d") + f"_{hora_rodada:02d}z"
                  if data_rodada else "atual")
            grib_s = os.path.join(OUT_DIR, f"_cache_{cfg['param']}_{_r}_{s}h.grib2")
            try:
                if not os.path.exists(grib_s):
                    baixar(cfg["param"], s, grib_s, data_rodada, hora_rodada)
                rec_s = ler_grib(grib_s, cfg["param"])
                arr_s = np.asarray(rec_s.values, dtype="float32")
                # fmin/fmax ignoram NaN — um sub-step faltando não zera o resultado
                acumulado = (np.fmin(acumulado, arr_s) if modo == "min_dia"
                             else np.fmax(acumulado, arr_s))
            except Exception as e:
                print(f"      (sub-step {s}h indisponível: {e})")
            # NÃO apaga aqui: o arquivo é reaproveitado pela outra variável
            # (mínima/máxima). A limpeza acontece no fim da execução.
        arr = acumulado
        _remover(grib_atual)
        arr = arr * fator + offset
        return arr, recorte

    if modo == "desacumular_dia" and step > 24:
        # subtrai o acumulado de 24 h antes para obter o total do dia
        step_ant = step - 24
        grib_ant = f"{tmp_prefix}_{step_ant}h.grib2"
        baixar(cfg["param"], step_ant, grib_ant, data_rodada, hora_rodada)
        rec_ant = ler_grib(grib_ant, cfg["param"])
        arr_ant = np.asarray(rec_ant.values, dtype="float32")
        arr = arr - arr_ant
        arr = np.clip(arr, 0, None)  # ruído numérico pode dar negativos pequenos
        _remover(grib_ant)

    _remover(grib_atual)

    arr = arr * fator + offset
    return arr, recorte


def _para_mercator(rgba, lat_n, lat_s):
    """Reprojeta a imagem de lat/lon linear (Plate Carrée) para Web Mercator.

    O PNG do ECMWF vem em grade geográfica linear, mas o Leaflet (imageOverlay)
    assume Web Mercator. Sem reprojetar, a imagem estica e escorrega para o sul
    (quanto mais longe do equador, maior o erro). Aqui reamostramos as LINHAS
    (eixo Y = latitude) para o espaçamento de Mercator. As colunas (longitude)
    não mudam, pois em Mercator a longitude continua linear.

    Retorna (rgba_merc, lat_n, lat_s) — os bounds de latitude não mudam; o que
    muda é a distribuição vertical dos pixels dentro deles.
    """
    import math
    h, w, _ = rgba.shape

    def merc_y(lat):
        lat = max(min(lat, 85.05), -85.05)
        return math.log(math.tan(math.pi / 4 + math.radians(lat) / 2))

    mN, mS = merc_y(lat_n), merc_y(lat_s)
    saida = np.zeros_like(rgba)
    for row in range(h):
        # y de mercator uniformemente distribuído na saída (topo=norte)
        ym = mN - (row / (h - 1)) * (mN - mS) if h > 1 else mN
        # latitude real correspondente a esse y de mercator
        lat = math.degrees(2 * math.atan(math.exp(ym)) - math.pi / 2)
        # de qual linha da imagem ORIGINAL (linear) essa latitude vem
        frac = (lat_n - lat) / (lat_n - lat_s) if lat_n != lat_s else 0.0
        src = int(round(frac * (h - 1)))
        src = max(0, min(h - 1, src))
        saida[row, :, :] = rgba[src, :, :]
    return saida


def gerar_png(arr, recorte, escala, minimo, png_path):
    """Colore o array pela escala e salva PNG RGBA. Retorna bounds [[s,w],[n,e]].

    Importante: as coordenadas do ECMWF são os CENTROS das células da grade.
    O overlay de imagem no Leaflet encaixa as BORDAS do PNG nos bounds, então
    é preciso expandir meia célula em cada direção — senão a imagem fica
    deslocada ~meia célula (efeito "puxado para baixo/lado").
    """
    lats = np.asarray(recorte.latitude.values, dtype="float64")
    lons = np.asarray(recorte.longitude.values, dtype="float64")

    # passo da grade (espaçamento entre células) em lat e lon
    dlat = abs(float(lats[1] - lats[0])) if len(lats) > 1 else 0.0
    dlon = abs(float(lons[1] - lons[0])) if len(lons) > 1 else 0.0

    if lats[0] < lats[-1]:
        arr = arr[::-1, :]
        lat_n, lat_s = float(lats[-1]), float(lats[0])
    else:
        lat_n, lat_s = float(lats[0]), float(lats[-1])
    lon_w, lon_e = float(lons.min()), float(lons.max())

    # expande meia célula: dos centros extremos para as bordas externas
    lat_n += dlat / 2.0
    lat_s -= dlat / 2.0
    lon_w -= dlon / 2.0
    lon_e += dlon / 2.0

    h, w = arr.shape
    rgba = np.zeros((h, w, 4), dtype="uint8")
    base = (~np.isnan(arr)) & (arr >= minimo)
    for lim, c in escala:
        m = base & (arr >= lim)
        rgba[m, 0] = c[0]; rgba[m, 1] = c[1]; rgba[m, 2] = c[2]; rgba[m, 3] = ALPHA

    # reprojeta de lat/lon linear para Web Mercator (o que o Leaflet espera)
    rgba = _para_mercator(rgba, lat_n, lat_s)

    Image.fromarray(rgba, mode="RGBA").save(png_path)
    return [[lat_s, lon_w], [lat_n, lon_e]]


def salvar_grade_f32(arr, recorte, arquivo):
    """Salva a grade numérica original em Float32 little-endian.

    O PNG é ótimo para desenhar a previsão, mas perde o valor exato ao trocar
    milímetros por uma faixa de cor. O comparador do WebGIS usa este arquivo
    compacto para consultar o valor da célula clicada no mapa.
    """
    valores = np.asarray(arr, dtype="<f4")
    lats = np.asarray(recorte.latitude.values, dtype="float64")
    lons = np.asarray(recorte.longitude.values, dtype="float64")
    if valores.ndim != 2 or valores.shape != (len(lats), len(lons)):
        raise ValueError(
            f"grade incompatível: valores={valores.shape}, "
            f"lat/lon=({len(lats)}, {len(lons)})"
        )
    valores.tofile(arquivo)
    return {
        "arquivo": os.path.basename(arquivo),
        "dtype": "float32-le",
        "shape": [int(valores.shape[0]), int(valores.shape[1])],
        "lat_inicio": float(lats[0]),
        "lat_passo": float(lats[1] - lats[0]) if len(lats) > 1 else 0.0,
        "lon_inicio": float(lons[0]),
        "lon_passo": float(lons[1] - lons[0]) if len(lons) > 1 else 0.0,
        "unidade": "mm",
    }


def desenhar_cruz_teste(png_path, bounds):
    """Desenha uma cruz vermelha nos PONTOS_TESTE, mapeando lat/lon -> pixel
    pelos MESMOS bounds da imagem. Se a cruz cair sobre a cidade no mapa, o
    alinhamento está correto. Sobrescreve o PNG com as cruzes por cima.
    """
    img = Image.open(png_path).convert("RGBA")
    w, h = img.size
    px = img.load()
    (lat_s, lon_w), (lat_n, lon_e) = bounds
    span_lat = (lat_n - lat_s) or 1e-9
    span_lon = (lon_e - lon_w) or 1e-9

    print(f"      [cruz] imagem {w}x{h}px, bounds S={lat_s:.3f} N={lat_n:.3f} "
          f"W={lon_w:.3f} E={lon_e:.3f}")
    for (lat, lon, rot) in PONTOS_TESTE:
        fx = (lon - lon_w) / span_lon
        # a imagem está em Mercator: a posição vertical usa a projeção de Mercator
        import math
        def merc_y(la):
            la = max(min(la, 85.05), -85.05)
            return math.log(math.tan(math.pi / 4 + math.radians(la) / 2))
        mN, mS = merc_y(lat_n), merc_y(lat_s)
        fy = (mN - merc_y(lat)) / (mN - mS) if mN != mS else 0.0
        cx = int(round(fx * (w - 1)))
        cy = int(round(fy * (h - 1)))
        print(f"      [cruz] {rot} ({lat:.4f},{lon:.4f}) -> pixel col={cx} lin={cy} "
              f"(fx={fx:.3f} fy={fy:.3f}) [Mercator]")
        if not (0 <= fx <= 1 and 0 <= fy <= 1):
            print(f"      [cruz] {rot} FORA do recorte!")
            continue
        R = 6
        for d in range(-R, R + 1):
            for (x, y) in ((cx + d, cy), (cx, cy + d)):
                if 0 <= x < w and 0 <= y < h:
                    px[x, y] = (255, 0, 0, 255)

    img.save(png_path)


def _remover(*caminhos):
    for base in caminhos:
        for f in (base, base + ".idx"):
            if os.path.exists(f):
                try:
                    os.remove(f)
                except OSError:
                    pass


def verificar_alinhamento(recorte, bounds, nome):
    """Sanidade dos bounds: confere que o span dos bounds = n_células * passo.

    Se a imagem tem N colunas de largura, os bounds devem cobrir exatamente N
    células (N * dlon). Divergência indica erro de meia célula ou pior.
    """
    lats = np.asarray(recorte.latitude.values, dtype="float64")
    lons = np.asarray(recorte.longitude.values, dtype="float64")
    (lat_s, lon_w), (lat_n, lon_e) = bounds
    if len(lats) > 1 and len(lons) > 1:
        dlat = abs(float(lats[1] - lats[0]))
        dlon = abs(float(lons[1] - lons[0]))
        span_lat_esperado = len(lats) * dlat
        span_lon_esperado = len(lons) * dlon
        span_lat_real = abs(lat_n - lat_s)
        span_lon_real = abs(lon_e - lon_w)
        erro_lat = abs(span_lat_real - span_lat_esperado)
        erro_lon = abs(span_lon_real - span_lon_esperado)
        tol = max(dlat, dlon) * 0.01  # 1% de uma célula de tolerância
        ok = erro_lat < tol and erro_lon < tol
        marca = "OK" if ok else "!! DESLOCADO !!"
        print(f"   [alinhamento {nome}] {marca} "
              f"(erro lat={erro_lat:.4f}° lon={erro_lon:.4f}°, célula={dlat:.2f}°)")
        return ok
    return True


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rodada = dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    # Recupera automaticamente somente a chuva do dia anterior.
    if HIST_CHUVA_DIAS > 0:
        preencher_historico_chuva(HIST_CHUVA_DIAS)

    # PREENCHER=N reconstrói os últimos N dias de histórico usando rodadas
    # passadas de temperatura. É mantido por compatibilidade com a camada de
    # erro de temperatura; o preenchimento automático de chuva é separado.
    preencher = int(os.getenv("PREENCHER", "0") or 0)
    if preencher > 0:
        preencher_historico(preencher)
        if os.getenv("SO_PREENCHER", "").strip() in ("1", "true", "sim"):
            print("\nSó preenchimento — encerrando sem gerar os PNGs do dia.")
            return
    meta = {"rodada": rodada, "variaveis": {}}

    for nome, cfg in VARIAVEIS.items():
        meta["variaveis"][nome] = {"horizontes": {}}
        # Para chuva, cada passo do ECMWF e acumulado desde a rodada. Guardar
        # o passo anterior permite publicar tambem a chuva isolada de cada dia.
        chuva_acumulada_anterior = None
        chuva_dia_anterior = 0
        for step in STEPS:
            dias = step // 24
            tmp_prefix = os.path.join(OUT_DIR, f"_tmp_{cfg['param']}")
            png_name = f"{cfg['prefixo']}_{dias}d.png"
            png_path = os.path.join(OUT_DIR, png_name)
            print(f"[{nome} {dias}d / {step}h] baixando ECMWF...")
            try:
                arr, recorte = valores_do_horizonte(cfg, step, tmp_prefix)
                bounds = gerar_png(arr, recorte, cfg["escala"], cfg["minimo"], png_path)
                verificar_alinhamento(recorte, bounds, f"{nome} {dias}d")
                if CRUZ_TESTE:
                    desenhar_cruz_teste(png_path, bounds)
                horizonte_meta = {
                    "arquivo": png_name,
                    "bounds": bounds,
                    "step_h": step,
                    "label": f"+{dias} dia{'s' if dias > 1 else ''}",
                    # Horários reais lidos do GRIB, para o webgis rotular sem
                    # adivinhar. `valido_ate` é o fim do período; a CHUVA é
                    # ACUMULADA desde a rodada, então o rótulo correto é
                    # "até <data>", e não "chuva do dia <data>".
                    "ref_utc": _iso_de(recorte, "time"),
                    "valido_ate_utc": _iso_de(recorte, "valid_time"),
                    "acumulado": (cfg.get("modo") == "acumulado_total"),
                }
                # Só a chuva precisa da grade numérica neste momento. Os
                # demais overlays continuam leves e exclusivamente visuais.
                if nome == "chuva":
                    grade_name = f"ecmwf_chuva_{dias}d.f32"
                    horizonte_meta["grade"] = salvar_grade_f32(
                        arr, recorte, os.path.join(OUT_DIR, grade_name)
                    )
                    if dias == 1:
                        arr_diario = np.maximum(arr, 0.0)
                    elif chuva_acumulada_anterior is not None and chuva_dia_anterior == dias - 1:
                        arr_diario = np.maximum(arr - chuva_acumulada_anterior, 0.0)
                    else:
                        arr_diario = None
                    if arr_diario is not None:
                        png_dia_nome = f"ecmwf_chuva_dia_{dias}d.png"
                        bounds_dia = gerar_png(
                            arr_diario, recorte, ESCALA_CHUVA, 0.2,
                            os.path.join(OUT_DIR, png_dia_nome)
                        )
                        grade_dia_nome = f"ecmwf_chuva_dia_{dias}d.f32"
                        grade_dia = salvar_grade_f32(
                            arr_diario, recorte, os.path.join(OUT_DIR, grade_dia_nome)
                        )
                        horizonte_meta["diario"] = {
                            "arquivo": png_dia_nome,
                            "bounds": bounds_dia,
                            "grade": grade_dia,
                            "label": f"chuva do dia +{dias}",
                            "acumulado": False,
                        }
                    chuva_acumulada_anterior = np.asarray(arr, dtype="float64").copy()
                    chuva_dia_anterior = dias
                meta["variaveis"][nome]["horizontes"][str(dias)] = horizonte_meta
                # Arquiva o previsto POR ESTAÇÃO no horizonte de 1 dia — é o
                # que será comparado com o medido quando o dia chegar.
                # guarda a imagem do horizonte de 1 dia para comparação futura
                if dias == 1 and nome in ("temperatura", "temperatura_max"):
                    arquivar_imagem(nome, _iso_de(recorte, "valid_time"),
                                    rodada, png_path, bounds)
                print(f"   gerado: {png_path}")
            except Exception as e:
                print(f"   ERRO ({nome} {step}h): {e}")

        # limpa .idx / tmp residuais desta variável
        for fn in os.listdir(OUT_DIR):
            if fn.startswith("_tmp_"):
                try:
                    os.remove(os.path.join(OUT_DIR, fn))
                except OSError:
                    pass

    # compatibilidade: mantém também o formato antigo (só chuva) em "horizontes"
    meta["horizontes"] = meta["variaveis"].get("chuva", {}).get("horizontes", {})

    # limpa o cache de sub-steps compartilhado entre mínima e máxima
    for fn in os.listdir(OUT_DIR):
        if fn.startswith("_cache_") or fn.startswith("_tmp_"):
            try:
                os.remove(os.path.join(OUT_DIR, fn))
            except OSError:
                pass
    limpar_historico_antigo()
    meta_path = os.path.join(OUT_DIR, "previsao_meta.json")
    with open(meta_path, "w", encoding="utf-8") as fp:
        json.dump(meta, fp, ensure_ascii=False, indent=2)
    print(f"\nMeta: {meta_path}")
    print("Pronto. Comite a pasta ./previsao/ no repositório.")


if __name__ == "__main__":
    main()
