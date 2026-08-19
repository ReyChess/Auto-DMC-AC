# Auto-DMC-AC v5.3.1 optimizado

Esta versión conserva la lógica de v5.3, pero evita recorrer nuevamente todas las reglas para cada valor de `lambda_prior` durante la validación interna.

## Uso

La línea de ejecución es la misma que en v5.3:

```bash
./build/auto_dmc_ac \
  --partition-dir /ruta/Particiones \
  --miner ./build/dmc_miner_unified \
  --output-dir resultados_v5_3_1 \
  --sigma 0.001 \
  --lambda-grid 0.05,0.10,0.15,0.20,0.30 \
  --inner-folds 3 \
  --outer-folds 10 \
  --seed 20260715 \
  --pn-policy auto \
  --cns-policy auto
```

## Qué cambia internamente

Para cada variante estructural e inner fold:

1. se mina una sola vez;
2. se recorren una sola vez las reglas para cada instancia de validación;
3. se guardan los scores positivos, el near-tie y la evidencia negativa;
4. se prueban todos los lambdas sobre esa evidencia ya calculada.

La salida mantiene los archivos `predictions_lambda_*.csv`, `selection_log.csv`, `fold_variant_summary.csv`, `instance_log.csv` y los resúmenes externos.

## Recomendación experimental

La ejecución de v5.3 que ya esté en curso debe terminarse sin sustituir el ejecutable. v5.3.1 debe utilizarse como una rama nueva y validarse primero en varios datasets comparando las predicciones por lambda con v5.3.
