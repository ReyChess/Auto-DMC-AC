# Auto-DMC-AC v5.3.2

Esta versión parte directamente de v5.3.1 y mantiene sin cambios el clasificador
y la optimización de caché de la rejilla de `lambda_prior`.

## Cambio respecto a v5.3.1

Se modifica únicamente el desempate de la selección automática. El tiempo de
ejecución deja de ser un criterio de selección y se conserva solo como medida
de coste computacional.

Ante empate exacto en macro-F1 medio y número medio de reglas:

1. Para valores de lambda de una misma variante estructural se prefiere el más
   cercano a `lambda_prior=0.15`.
2. Si dos valores son equidistantes (por ejemplo, 0.10 y 0.20), se selecciona
   el menor.
3. Los empates estructurales restantes prefieren P sobre PN, global sobre CNS
   y, finalmente, un orden léxico fijo.

No se modifican la minería, Netconf, K=12, el shrinkage, la cobertura exact-first,
el fallback parcial, las reglas negativas ni el veto conservador.

Los argumentos de ejecución y el formato de los resultados se mantienen
compatibles con v5.3.1.
