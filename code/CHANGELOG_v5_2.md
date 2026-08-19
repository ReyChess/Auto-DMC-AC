# Auto-DMC-AC v5.2

- Añadida política `--cns-policy auto|always|never`.
- Añadido pre-check estructural por soporte absoluto mínimo y relajación efectiva del soporte.
- Añadido piloto CNS estratificado, limitado mediante `/usr/bin/timeout` en Cygwin/Linux.
- Añadidos presupuestos reproducibles de tiempo, memoria y huella estimada por regla.
- Las variantes `cns_p` y `cns_pn` se omiten juntas cuando la guardia determina que CNS no es viable.
- Añadido `cns_guard.csv` y persistencia completa de parámetros en `run_configuration.txt` y `auto_dmc_report.txt`.
- La decisión de viabilidad usa únicamente el entrenamiento externo y nunca el fold de prueba.
