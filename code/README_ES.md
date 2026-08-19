# Auto-DMC-AC v5 — selector automático con control de expansión PN

Este proyecto evalúa las variantes:

- `global_p`
- `global_pn`
- `cns_p`
- `cns_pn`

La selección se realiza mediante validación cruzada interna y macro-F1. La evaluación externa permanece separada.

## Cambio principal de v5

La generación de reglas negativas puede crecer aproximadamente como:

`R_neg_est = R_pos_max * (C - 1) * safety_factor`

Por ello se añadió una política explícita y reproducible:

- `--pn-policy auto`: evalúa PN solamente cuando pasa los límites configurados.
- `--pn-policy always`: fuerza PN, incluso si la estimación es grande.
- `--pn-policy never`: evalúa únicamente las variantes positivas.

El modo `auto` usa exclusivamente información del entrenamiento interno:

- máximo número de reglas positivas observado para el mismo tipo de soporte;
- número de clases del entrenamiento;
- límite de clases configurable;
- presupuesto de reglas derivado de memoria:

`rule_budget = memory_budget_bytes / bytes_per_rule`

Valores predeterminados:

- `--pn-max-classes 20`
- `--pn-memory-budget-mb 128`
- `--pn-bytes-per-rule 256`
- `--pn-estimation-safety 1.25`

Todos los cálculos y decisiones quedan en `pn_guard.csv`. Los parámetros completos quedan en `run_configuration.txt`.

## Compilar con Cygwin

Desde PowerShell, dentro de `auto_fixed`:

```powershell
.\build_with_cygwin.bat
```

Los ejecutables se crean en `build`.

## Ejecución recomendada para LetRecog

Suponiendo que `LetRecogPartition` está dentro de `auto_fixed`:

```powershell
& "C:\cygwin64\bin\bash.exe" -lc '
cd "/cygdrive/e/AutoDMCACFinal/Auto_DMC_AC_CPP_final_optimized_cygwin_v5/auto_fixed/build" &&
./auto_dmc_ac.exe \
  --partition-dir "../LetRecogPartition" \
  --miner "./dmc_miner_unified.exe" \
  --output-dir "../letRecog_auto_results_v5" \
  --sigma 0.01 \
  --min-netconf 0.0 \
  --inner-folds 3 \
  --seed 20260715 \
  --outer-folds 10 \
  --analysis-scope final \
  --instance-log-scope final \
  --pn-policy auto \
  --pn-max-classes 20 \
  --pn-memory-budget-mb 128 \
  --pn-bytes-per-rule 256 \
  --pn-estimation-safety 1.25
'
```

Con 26 clases, LetRecog omite PN en modo `auto` por el límite predeterminado de clases. Para ejecutar las cuatro variantes deliberadamente, use `--pn-policy always`.

## Archivos de salida relevantes

- `run_configuration.txt`: configuración completa.
- `pn_guard.csv`: decisión de evaluar u omitir PN y su justificación.
- `fold_variant_summary.csv`: métricas de cada ejecución realizada.
- `selection_log.csv`: ranking de las variantes realmente evaluadas.
- `dataset_summary.csv`: resultados externos por fold.
- `auto_dmc_report.txt`: resumen general.

## Otras mejoras

- La evidencia detallada por regla e instancia solo se conserva cuando `analysis-scope` la necesita. Esto reduce mucho la memoria en datasets grandes.
- La compilación Cygwin verifica individualmente cada ejecutable y ya no depende del glob problemático de `ls *.exe`.
- La configuración efectiva se imprime al inicio para evitar ejecutar accidentalmente con otro valor de `sigma`.

## Guardia computacional CNS (v5.2)

La opción `--cns-policy auto` evita que el soporte normalizado por clase bloquee un fold cuando la reducción efectiva del soporte abre un espacio de búsqueda excesivo. La decisión usa exclusivamente el conjunto de entrenamiento externo.

Parámetros predeterminados:

```
--cns-policy auto
--cns-max-support-relaxation 12
--cns-min-absolute-support 5
--cns-pilot-fraction 0.10
--cns-pilot-time-limit-sec 30
--cns-max-estimated-time-sec 300
--cns-memory-budget-mb 1024
--cns-bytes-per-rule 256
--cns-estimation-safety 2.0
```

Lógica:

1. Si el soporte absoluto mínimo por clase es inferior al suelo estructural, se omite CNS.
2. Si la relajación `ceil(sigma*N)/min_c ceil(sigma*N_c)` no supera el límite, CNS se evalúa normalmente.
3. En caso contrario se ejecuta un piloto estratificado y limitado por tiempo.
4. CNS se omite si el piloto agota el tiempo, falla, o proyecta un coste superior a los presupuestos de tiempo o memoria.

Las decisiones quedan registradas en `cns_guard.csv`. `--cns-policy always` fuerza CNS y `--cns-policy never` lo desactiva.
