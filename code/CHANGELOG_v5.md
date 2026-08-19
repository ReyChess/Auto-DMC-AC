# Cambios de v5

1. Política PN configurable: `auto`, `always`, `never`.
2. Guardia automática por número de clases y presupuesto derivado de memoria.
3. Estimación conservadora basada en las reglas positivas de los folds internos.
4. Registro reproducible en `pn_guard.csv` y `run_configuration.txt`.
5. Las variantes omitidas no se tratan como perdedoras; simplemente no entran en el ranking.
6. Menor consumo de memoria: no se copian antecedentes y evidencias completas en ejecuciones que no generan `AnalisisInstancias.dat`.
7. Verificación Cygwin robusta, ejecutable por ejecutable, sin `ls "$BUILD_DIR"/*.exe`.
8. Mensaje inicial con los parámetros efectivos de la ejecución.
