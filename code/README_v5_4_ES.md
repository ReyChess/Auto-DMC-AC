# Auto-DMC-AC v5.4

> **Actualización v5.4.2.** La decisión positiva ajustada por el prior se
> calcula ahora globalmente sobre todas las clases con evidencia positiva,
> también cuando existe un *near-tie* no ajustado. El *near-tie* conserva su
> función diagnóstica y de delimitación de la etapa negativa, pero no restringe
> el `argmax` positivo. La misma corrección se aplica a la evaluación ordinaria
> y a la caché de la rejilla de `lambda_prior`. Véase `CHANGELOG_v5_4_2.md`.

> **Actualización v5.4.1.** La admisión de evidencia positiva es ahora
> estricta: `Netconf(A=>c) > min_netconf_pos`. Con el valor experimental
> predeterminado `min_netconf_pos=0`, `Netconf=0` se interpreta correctamente
> como independencia y no se almacena como evidencia positiva. Véase
> `CHANGELOG_v5_4_1.md`. El resto del protocolo v5.4 descrito abajo no cambia.

Esta versión elimina la selección manual de `sigma` por dataset. El selector
anidado considera conjuntamente soporte, variante estructural y peso del prior,
sin consultar nunca el fold externo de prueba.

Configuración predeterminada relevante:

```text
sigma_grid = 0.30,0.10,0.03,0.01,0.003,0.001
lambda_grid = 0,0.05,0.10,0.15,0.20,0.30
inner_folds = 3
sigma_mining_time_limit_sec = 30
```

Ejemplo:

```bash
./auto_dmc_ac \
  --partition-dir /ruta/DatasetPartition \
  --miner ./dmc_miner_unified \
  --output-dir resultados_v5_4 \
  --sigma-grid 0.30,0.10,0.03,0.01,0.003,0.001 \
  --lambda-grid 0,0.05,0.10,0.15,0.20,0.30 \
  --inner-folds 3 \
  --outer-folds 10 \
  --seed 20260715
```

El grid de soporte se recorre desde el umbral más estricto al más permisivo.
Si `global_p` alcanza el límite computacional en un valor de `sigma`, los
valores menores se descartan automáticamente porque su espacio de itemsets
frecuentes contiene al del candidato que ya excedió el presupuesto.

Para reproducir un experimento antiguo con soporte fijo se conserva
`--sigma VALUE`, que equivale a un grid de un solo candidato.
