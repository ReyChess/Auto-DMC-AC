# Auto-DMC-AC v5.3

Esta versión amplía v5.2 para seleccionar conjuntamente mediante validación cruzada interna:

1. la variante estructural (`global_p`, `global_pn`, `cns_p`, `cns_pn`);
2. el peso del prior `lambda_prior`.

Se mantienen fijos `K=12` y `beta=2`.

## Compilación en Cygwin

Desde la carpeta del proyecto:

```bash
bash build_cygwin.sh
```

También puede usarse:

```bat
build_with_cygwin.bat
```

## Ejecución

Ejemplo:

```bash
./build/auto_dmc_ac \
  --partition-dir /ruta/Partition \
  --miner ./build/dmc_miner_unified \
  --output-dir resultados_v5_3 \
  --sigma 0.03 \
  --inner-folds 3 \
  --outer-folds 10 \
  --lambda-grid 0.05,0.10,0.15,0.20,0.30 \
  --pn-policy auto \
  --cns-policy auto
```

## Reutilización eficiente

Para cada variante estructural y cada inner fold, las reglas se minan una sola vez. Las mismas reglas se reclasifican con todos los valores de lambda. Por ello, cinco valores de lambda no multiplican por cinco el coste de minería.

## Salidas principales

- `dataset_summary.csv`: variante y lambda seleccionados en cada outer fold;
- `selection_log.csv`: ranking completo de cada par estructura–lambda;
- `fold_variant_summary.csv`: métricas de cada ejecución interna;
- `run_configuration.txt`: configuración reproducible;
- `auto_dmc_report.txt`: resumen final.

## Comparación con v5.2

Para reproducir el comportamiento de v5.2, use:

```bash
--lambda-grid 0.15
```

La selección estructural seguirá funcionando como antes, manteniendo lambda fijo en 0.15.
