/*	@ este programa construye un clasificador basado en CAR's para lo cual realiza los distintos pasos:
 * 	@ a.- Lectura del dataset
 * 	@ b.- Formacion de 10 sub-conjuntos disjuntos del dataset de forma tal que en cada sub-conjunto
 * 				esten presentes ejemplos de todas las clases.
 * 	@ c.- A partir de los sub-conjuntos realizar las fases de entrenamiento o construccion del clasificador
 * 				utilizando 9 de los 10 sub-conjuntos y a continuacion evaluacion del clasificador construido utilizando
 * 				el sub-conjunto restante. Este paso se descompone en:
 * 					i.- Formacion del file del dataset a partir de los sub-conjuntos a utilizar en el entrenamiento.
 * 					ii.- Ejecucion del algoritmo de extraccion de CAR's a partir del file del dataset creado en (i).
 * 					iii.- Construccion del clasificador con las reglas obtenidas en el paso anterior.
 * 					iv.- Evaluacion del clasificador con el conjunto faltante del entrenamiento.
 * 					v.- Reporte de la eficiencia alcanzada. 
 * 	@ d.- Ejecutar el paso c con cada una de las combinaciones de 9 que se pueden extraer de 10 sub-conjuntos.	
 * 	@ e. Calculo de la eficiencia promedio del proceso.
 *  NOTA :- Aqui la clase por defecto se selecciona en dependencia de la clase cuyo conjunto de reglas representativo
 * 					tenga un promedio de netconf mayor
 * */

// directivas incluidas

#include <string.h>
#include <math.h>
#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <stdint.h>
#include <inttypes.h>


// ------------------ Partition / fold file signature helpers ------------------
// When doing repeated runs with externally-generated folds (DivideEn10), we want
// to record *which* partition was used, so later stability analysis can be done
// correctly across runs.
// We compute a small signature per fold file based on (size, mtime).
static inline void file_signature(const char* path, long long* out_size, long long* out_mtime)
{
  struct stat st;
  if (stat(path, &st) == 0) {
    *out_size = (long long)st.st_size;
    *out_mtime = (long long)st.st_mtime;
  } else {
    *out_size = -1;
    *out_mtime = -1;
  }
}

// ================== Tipos base (DEBEN estar completos antes de usarlos) ==================
// En C/C++ NO se puede acceder a miembros de un struct declarado solo con "forward declaration".
// El error que viste ("invalid use of incomplete type") ocurre exactamente por eso.
// Por tanto, definimos aquí los structs tal y como están en tu main.cpp original.

typedef struct
{
  int * elementos, longitud, tamanno;
} CConjunto;

typedef struct
{
  CConjunto * antecedente;
  int clase; // etiqueta de clase (siempre positiva)
  unsigned char negated; // 0: A->c (positiva), 1: A->neg(c) (negativa)
  double netconfREGLA, interesDONG, soporteREGLA, soporteANTECEDENTE, soporteCLASE;
} CCAR;

typedef struct
{
  CConjunto * items;
  int clase;
} CTransaccion;

// ================== Stable rule selection (deterministic & smoothed) ==================
// NOTA:
//  - El esquema anterior ("K dinámico" con K = N_min) puede ser muy inestable cuando
//    N_min es pequeño (pocas reglas por clase cubriendo la instancia). En esos casos,
//    el promedio de netconf se vuelve muy ruidoso y el ganador puede cambiar drásticamente
//    de un fold a otro.
//
// Este bloque implementa una selección más estable:
//  1) Usa un K fijo acotado (K_MAX) por clase.
//  2) Promedio ponderado por rango (reglas mejores pesan más).
//  3) Suavizado Bayesiano/shrinkage hacia un prior global (MU0) con fuerza BETA.
//  4) Desempates deterministas: score, luego prior, luego id de clase.
//
// Ajusta estos parámetros si lo deseas (valores conservadores):
#ifndef STABLE_K_MAX
#define STABLE_K_MAX 12
#endif

#ifndef STABLE_BETA
#define STABLE_BETA 2.0
#endif

// Negatives should be more conservative than positives.
// We regularize negative evidence harder to avoid one spurious negative dominating.
#ifndef STABLE_BETA_NEG
#define STABLE_BETA_NEG 15.0
#endif

#ifndef STABLE_LAMBDA_NEG
#define STABLE_LAMBDA_NEG 0.25
#endif

// Adaptive negative weight: negatives matter most when positives are extremely close.
// tau controls how fast the negative influence decays as the positive gap grows.
#ifndef STABLE_TAU
#define STABLE_TAU 0.01
#endif

// Usar CARs negativas SOLO para romper empates en la evidencia positiva.
// Un "empate" se define como clases con ScorePos a distancia <= STABLE_TIE_EPS del máximo.
#ifndef STABLE_TIE_EPS
// Near-tie threshold (POS evidence units). For FLARE, 0.002 is a safe starting point.
#define STABLE_TIE_EPS 0.005
#endif

// Numerical epsilon (do NOT tune this for behavior).
#ifndef STABLE_NUM_EPS
#define STABLE_NUM_EPS 1e-12
#endif

// Para evitar que la evidencia negativa domine (especialmente en datasets ruidosos),
// limitamos el número de negativas consideradas por clase.
#ifndef STABLE_K_NEG_MAX
#define STABLE_K_NEG_MAX 2
#endif

#ifndef STABLE_LAMBDA_PRIOR
// Small prior term to avoid over-predicting minority classes in highly imbalanced datasets.
// This term is used in the BASE score (always) and in near-tie resolution.
#define STABLE_LAMBDA_PRIOR 0.15
#endif

#ifndef STABLE_POS_MIN
// If the best positive evidence is below this threshold, treat the instance as weakly covered
// and fall back to the default class.
#define STABLE_POS_MIN -1e300
#endif


// ================== M4: M2 + negative rules only for tie-breaking ==================
// M4 keeps M2 positive exact/partial coverage. Negative rules are not used as an
// independent classifier and are not applied when there is a clear positive winner.
// They are evaluated lazily only inside the near-tie region, for tied classes only.
#ifndef M2_PARTIAL_COVERAGE
#define M2_PARTIAL_COVERAGE 1
#endif

#ifndef M2_PARTIAL_MAX_MISSING
#define M2_PARTIAL_MAX_MISSING 1
#endif

#ifndef M4_USE_NEGATIVES_FOR_TIEBREAK
#define M4_USE_NEGATIVES_FOR_TIEBREAK 1
#endif

// M4d policy: negative rules are evaluated only inside the positive near-tie
// region and ONLY with exact coverage. They are used as a veto against the
// current positive+prior winner, not as a general re-ranking mechanism. The
// winner is replaced only when it is strongly negated and another tied class
// has clearly lower negative evidence.
#ifndef M4_NEGATIVE_PARTIAL_COVERAGE
#define M4_NEGATIVE_PARTIAL_COVERAGE 0
#endif

#ifndef M4_NEG_PARTIAL_MAX_MISSING
#define M4_NEG_PARTIAL_MAX_MISSING 1
#endif

#ifndef M4_NEG_VETO_MIN
#define M4_NEG_VETO_MIN 0.10
#endif

#ifndef M4_NEG_VETO_DIFF
#define M4_NEG_VETO_DIFF 0.05
#endif

static inline double stable_rank_weight(int rank0)
{
  // rank0 = 0 para la mejor regla, 1 para la segunda, ...
  // pesos decrecientes: 1, 1/2, 1/3, ...
  return 1.0 / (double)(rank0 + 1);
}

// Evidencia POSITIVA (A->c): promedio ponderado top-K + shrinkage hacia mu0_pos
static inline double stable_score_pos_class(
    int c,
    CConjunto **reglasCUBREN_POS,
    CCAR ***listaCCAR,
    double mu0_pos)
{
  int Nc = reglasCUBREN_POS[c]->longitud;
  if (Nc <= 0) return -1e300;

  int K = (Nc < STABLE_K_MAX ? Nc : STABLE_K_MAX);

  double num = 0.0, den = 0.0;
  for (int j = 0; j < K; ++j) {
    int idx = reglasCUBREN_POS[c]->elementos[j];
    double w = stable_rank_weight(j);
    num += w * listaCCAR[c][idx]->netconfREGLA; // netconf positivo (A->c)
    den += w;
  }

  double avg = (den > 0.0 ? (num / den) : 0.0);

  // shrinkage hacia mu0_pos (baseline de netconf de reglas positivas que cubren la instancia)
  double shrunk = (avg * (double)K + STABLE_BETA * mu0_pos) / ((double)K + STABLE_BETA);

  return shrunk;
}

// Evidencia NEGATIVA (A->neg(c)): penalización contra la clase c.
// En el archivo, una regla negativa se almacena como A -> -c, pero su ultimo
// valor es Netconf(A -> c), que debe ser negativo. Por tanto, la fuerza negativa
// se toma como -Netconf(A -> c). Esta función se evalúa solo en región de empate.
static inline double stable_score_neg_class(
    int c,
    CConjunto **reglasCUBREN_NEG,
    CCAR ***listaCCAR)
{
  int Nc = reglasCUBREN_NEG[c]->longitud;
  if (Nc <= 0) return 0.0;

  // Para no depender del orden del archivo, buscamos las negativas más fuertes.
  // Esto es más seguro que asumir que las reglas negativas quedaron ordenadas.
  double top[STABLE_K_NEG_MAX];
  int used = 0;
  for (int i = 0; i < STABLE_K_NEG_MAX; ++i) top[i] = 0.0;

  for (int j = 0; j < Nc; ++j) {
    int idx = reglasCUBREN_NEG[c]->elementos[j];
    double strength = -listaCCAR[c][idx]->netconfREGLA;
    if (strength <= 0.0) continue;

    if (used < STABLE_K_NEG_MAX) {
      top[used++] = strength;
    } else {
      int worst = 0;
      for (int k = 1; k < STABLE_K_NEG_MAX; ++k)
        if (top[k] < top[worst]) worst = k;
      if (strength > top[worst]) top[worst] = strength;
    }
  }

  if (used <= 0) return 0.0;

  // Usamos la mayor evidencia negativa, con shrinkage hacia 0. El factor used
  // hace que dos negativas fuertes sean un poco más confiables que una sola,
  // pero STABLE_BETA_NEG evita sobrepenalización.
  double maxNeg = 0.0;
  for (int k = 0; k < used; ++k) if (top[k] > maxNeg) maxNeg = top[k];

  double shrunk = (maxNeg * (double)used) / ((double)used + STABLE_BETA_NEG);
  return shrunk;
}

static inline void append_rule_index(CConjunto *dst, int idx)
{
  if (dst->longitud == dst->tamanno) {
    dst->tamanno *= 2;
    dst->elementos = (int*)realloc(dst->elementos, sizeof(int) * dst->tamanno);
  }
  dst->elementos[dst->longitud++] = idx;
}






// Count how many antecedent items are missing from the current instance.
// itemsEJEMPLO is a 0/1 array indexed by item_id-1 and has length 'limite'.
// An item whose id is outside the current instance range is treated as absent.
static inline int missing_items_in_antecedent(const CConjunto *antecedente,
                                             const int *itemsEJEMPLO_local,
                                             int limite,
                                             int stop_after)
{
  int missing = 0;
  for (int l = 0; l < antecedente->longitud; ++l) {
    int item = antecedente->elementos[l];
    int pos = item - 1;
    if (pos < 0 || pos >= limite || itemsEJEMPLO_local[pos] == 0) {
      missing++;
      if (missing > stop_after) return missing;
    }
  }
  return missing;
}

static inline int antecedent_covers_exactly(const CConjunto *antecedente,
                                            const int *itemsEJEMPLO_local,
                                            int limite)
{
  return missing_items_in_antecedent(antecedente, itemsEJEMPLO_local, limite, 0) == 0;
}

static inline int antecedent_covers_partially_m2(const CConjunto *antecedente,
                                                 const int *itemsEJEMPLO_local,
                                                 int limite)
{
  int missing = missing_items_in_antecedent(antecedente, itemsEJEMPLO_local, limite, M2_PARTIAL_MAX_MISSING);
  return (missing > 0 && missing <= M2_PARTIAL_MAX_MISSING);
}

// definicion de variables
int * clasesCOLECCION, cCLASES = 0 , tCLASES = 100, tBuffer = 2000, defaultCLASS = 0;
const int tamanno_arregloCjt = 6000;
char ** nombres_files;
char ** nombres_REGLAS;

double eficienciaPROMEDIO = 0, eficienciaEXPERIMENTO = 0;

CCAR *** listaCCAR;
int * cantidadCCAR, * totalCCAR;
double ** aveREGLA, ** desvREGLA;
int * itemsEJEMPLO;
CConjunto ** reglasCUBREN_POS, ** reglasCUBREN_NEG, ** reglasCUBREN_POS_CANTIDAD;
int claseASIGNADA;
double * averageWEIGHT;

// ================== Stability logging (CSV) ==================
// This file is intentionally lightweight: it enables direct measurement of
// decision stability (agreement/entropy) and tie-region behavior across
// repeated CV runs. It does NOT require access to baseline models.
static int STABILITY_RUN_ID = 0; // optional: passed as argv[2]

static inline long file_size_bytes(const char* path)
{
  struct stat st;
  if (stat(path, &st) != 0) return 0;
  return (long)st.st_size;
}

static inline void csv_write_header_if_empty(FILE* f, const char* path, const char* header)
{
  if (!f) return;
  if (file_size_bytes(path) == 0) {
    fprintf(f, "%s\n", header);
    fflush(f);
  }
}
double * priorClase;   // prior probability for each class (same index as clasesCOLECCION)

// definicion de variables

// prototipos de los metodos
int process_arguments(int argc, char *argv[]);
void Leer_File_CLASSES();
void Leer_Clasificador(char * file_clasificador);
void Construye_ClasificadorEXACTO(char * file_clasificador,char * file_prueba, int fold_idx);
void Construye_ClasificadorFUZZY(char * file_clasificador,char * file_prueba, double b0);
void Clasifica_Dataset(int argc, char *argv[]);

// prototipos de los metodos

// implementacion de los metodos
int process_arguments(int argc, char *argv[])
{
  if (argc != 2 && argc != 3)
  {
    printf("\nError! Usage: %s <dataset_name> [run_id]\n", argv[0]);
    return 1;  }

  if (argc == 3) {
    STABILITY_RUN_ID = atoi(argv[2]);
  }
  
  return 0;
}

void Leer_File_CLASSES()
{
  clasesCOLECCION = 0;
  clasesCOLECCION = (int*)malloc(sizeof(int)*tCLASES);
  priorClase = 0;
  priorClase = (double*)malloc(sizeof(double)*tCLASES);
  cCLASES = 0;

  FILE * ficheroREAD = 0;
  ficheroREAD = fopen("Classes.dat","r");

  // buffer que almacenara lo leido
  char * buffer = 0;
  buffer = (char *)malloc(sizeof(char)*tBuffer);
  char * eof = 0;
  char * temp = 0;
  temp = (char *)malloc(sizeof(char)*20);
  int inicio = 0, j = 0;

  double totalCount = 0.0;   // suma de los conteos de todas las clases

  eof = fgets(buffer,tBuffer,ficheroREAD);
  while (eof != 0)
  {
    // ----- leer el label de la clase (primer numero) -----
    inicio = 0; j = 0;
    while (buffer[inicio]!= ' ' && buffer[inicio] != '\t' && buffer[inicio] != '\n' && buffer[inicio] != '\0')
    {
      temp[j] = buffer[inicio];
      inicio++; j++;
    }
    temp[j] = '\0';

    // si es necesario, agrandar arreglos
    if (cCLASES == tCLASES)
    {
      tCLASES *= 2;
      clasesCOLECCION = (int*)realloc(clasesCOLECCION, sizeof(int)*tCLASES);
      priorClase      = (double*)realloc(priorClase,      sizeof(double)*tCLASES);
    }

    clasesCOLECCION[cCLASES] = atoi(temp);

    // ----- leer el conteo de la clase (segundo numero) -----
    // saltar espacios
    while (buffer[inicio] == ' ' || buffer[inicio] == '\t')
      inicio++;

    j = 0;
    while (buffer[inicio]!= ' ' && buffer[inicio] != '\t' && buffer[inicio] != '\n' && buffer[inicio] != '\0')
    {
      temp[j] = buffer[inicio];
      inicio++; j++;
    }
    temp[j] = '\0';

    int count = atoi(temp);
    priorClase[cCLASES] = (double)count;   // por ahora guardamos el conteo

    totalCount += count;
    cCLASES++;

    eof = fgets(buffer,tBuffer,ficheroREAD);
  }

  fclose(ficheroREAD);
  ficheroREAD = 0;

  // ---- normalizar para obtener priors reales ----
  if (totalCount > 0.0)
  {
    for (int i = 0; i < cCLASES; ++i)
      priorClase[i] /= totalCount;
  }

  // Seleccionar la clase por defecto (GLOBAL) como la de mayor prior.
  // Esto es crítico en datasets muy desbalanceados (p.ej. FLARE): evita que el
  // clasificador "se escape" hacia clases minoritarias cuando la evidencia de reglas es débil.
  defaultCLASS = 0;
  double bestPrior = -1.0;
  for (int i = 0; i < cCLASES; ++i) {
    if (priorClase[i] > bestPrior) { bestPrior = priorClase[i]; defaultCLASS = i; }
  }
}


void Leer_Clasificador(char * file_clasificador)
{
  cantidadCCAR = 0;
  cantidadCCAR = (int*)malloc(sizeof(int)*cCLASES);
  totalCCAR = 0;
  totalCCAR = (int*)malloc(sizeof(int)*cCLASES);
	aveREGLA = (double**)malloc(sizeof(double*)*cCLASES);
	desvREGLA = (double**)malloc(sizeof(double*)*cCLASES);
	
	listaCCAR = (CCAR ***)malloc(sizeof(CCAR **)*cCLASES);
	
	for (int i = 0 ; i < cCLASES ; i++)
	{
		cantidadCCAR[i] = 0; totalCCAR[i] = tamanno_arregloCjt;
		listaCCAR[i] = (CCAR **)malloc(sizeof(CCAR *)*totalCCAR[i]);
		aveREGLA[i] = (double*)malloc(sizeof(double)*totalCCAR[i]);
		desvREGLA[i] = (double*)malloc(sizeof(double)*totalCCAR[i]);
	}	
	
  FILE * fichero = fopen(file_clasificador,"r");
  //buffer que almacenara lo leido
  char * buffer = 0;
  buffer = (char *)malloc(sizeof(char)*tBuffer);  
  
  char * temp = 0;
  temp = (char *)malloc(sizeof(char)*30);
  
  int inicio = 0, j = 0, elem = 0, iPOS = -1, clase = 0;
  char *eof = 0;
  int nITEMS = 0;
  CConjunto * iConjunto = 0;
  iConjunto = (CConjunto*)malloc(sizeof(CConjunto));
  iConjunto->longitud = 0;
  iConjunto->tamanno = tamanno_arregloCjt;
  iConjunto->elementos = 0;
  iConjunto->elementos = (int*)malloc(sizeof(int)*iConjunto->tamanno);
  
  eof = fgets(buffer,tBuffer,fichero);
  //int linea = 0;
  while (eof != 0)
  {
  	inicio = 0;
  	iConjunto->longitud = 0;

  	///leer la cantidad de items en la regla
  	j = 0;
  	while (buffer[inicio]!=' ')
  	{
  		temp[j] = buffer[inicio];
  		inicio++; j++;
  	}
  	inicio++;
  	temp[j] = '\0';
  	nITEMS = atoi(temp);
  	
  	for (int i = 0 ; i < nITEMS ; i++)
  	{
  		// leyendo cada item de la regla
	  	j = 0;
	  	while (buffer[inicio]!=' ')
	  	{
	  		temp[j] = buffer[inicio];
	  		inicio++; j++;
	  	}
	  	temp[j] = '\0';
  		elem = atoi(temp);
  		inicio++;
  		if (iConjunto->longitud == iConjunto->tamanno)
  		{
  			iConjunto->tamanno *= 2;
  			iConjunto->elementos = (int*)realloc(iConjunto->elementos, sizeof(int)*iConjunto->tamanno);
  		}
  		iConjunto->elementos[iConjunto->longitud] = elem;
  		iConjunto->longitud++;
  		
  	}
  	
  	// localizar la clase para insertar su CCAR
	// Soportamos reglas negativas: en el minado salen como A -> -c (equivalente a A -> neg(c)).
	int claseRaw = iConjunto->elementos[iConjunto->longitud - 1];
	unsigned char esNeg = (claseRaw < 0 ? 1 : 0);
	int claseAbs = (claseRaw < 0 ? -claseRaw : claseRaw);

	clase = claseAbs;
	iPOS = -1;
	for (int i = 0 ; i < cCLASES ; i++)
	{
		if (clasesCOLECCION[i] == claseAbs)
		{
			iPOS = i; break;
		}
	}// insertar la CCAR leida en la clase correspondiente
  	if (cantidadCCAR[iPOS] == totalCCAR[iPOS])
  	{
  		totalCCAR[iPOS] *= 2;
  		listaCCAR[iPOS] = (CCAR **)realloc(listaCCAR[iPOS], sizeof(CCAR *)*totalCCAR[iPOS]);
			aveREGLA[iPOS] = (double*)realloc(aveREGLA[iPOS], sizeof(double)*totalCCAR[iPOS]);
			desvREGLA[iPOS] = (double*)realloc(desvREGLA[iPOS], sizeof(double)*totalCCAR[iPOS]);  		
  	}
  	
  	listaCCAR[iPOS][cantidadCCAR[iPOS]] = 0;
  	listaCCAR[iPOS][cantidadCCAR[iPOS]] = (CCAR*)malloc(sizeof(CCAR));
  	listaCCAR[iPOS][cantidadCCAR[iPOS]]->clase = clase;
	listaCCAR[iPOS][cantidadCCAR[iPOS]]->negated = esNeg;
  	listaCCAR[iPOS][cantidadCCAR[iPOS]]->antecedente = (CConjunto*)malloc(sizeof(CConjunto));
  	listaCCAR[iPOS][cantidadCCAR[iPOS]]->antecedente->longitud = 0;
  	listaCCAR[iPOS][cantidadCCAR[iPOS]]->antecedente->tamanno = nITEMS - 1;
  	listaCCAR[iPOS][cantidadCCAR[iPOS]]->antecedente->elementos = 0;
  	listaCCAR[iPOS][cantidadCCAR[iPOS]]->antecedente->elementos = (int*)malloc(sizeof(int)*listaCCAR[iPOS][cantidadCCAR[iPOS]]->antecedente->tamanno);
  	
  	for (int i = 0; i < nITEMS - 1 ; i++)
  	{
  		listaCCAR[iPOS][cantidadCCAR[iPOS]]->antecedente->elementos[listaCCAR[iPOS][cantidadCCAR[iPOS]]->antecedente->longitud] = iConjunto->elementos[i];
  		listaCCAR[iPOS][cantidadCCAR[iPOS]]->antecedente->longitud++;
  	}
  	
  	// leer los valores de soporte, y netCONF correspondientes
  	// soporte del antecedente
  	j = 0;
  	while (buffer[inicio]!=' ')
  	{
  		temp[j] = buffer[inicio];
  		inicio++; j++;
  	}
  	inicio++;
  	temp[j] = '\0';
		listaCCAR[iPOS][cantidadCCAR[iPOS]]->soporteANTECEDENTE = atof(temp); 
  	
  	// soporte del consecuente (clase)
  	j = 0;
  	while (buffer[inicio]!=' ')
  	{
  		temp[j] = buffer[inicio];
  		inicio++; j++;
  	}
  	inicio++;
  	temp[j] = '\0';
		listaCCAR[iPOS][cantidadCCAR[iPOS]]->soporteCLASE = atof(temp); 

  	// soporte de la regla
  	j = 0;
  	while (buffer[inicio]!=' ')
  	{
  		temp[j] = buffer[inicio];
  		inicio++; j++;
  	}
  	inicio++;
  	temp[j] = '\0';
		listaCCAR[iPOS][cantidadCCAR[iPOS]]->soporteREGLA = atof(temp);
		 
  	// netconf de la regla
  	j = 0;
  	while (buffer[inicio]!=' ' && buffer[inicio]!='\n')
  	{
  		temp[j] = buffer[inicio];
  		inicio++; j++;
  	}
  	inicio++;
  	temp[j] = '\0';
		listaCCAR[iPOS][cantidadCCAR[iPOS]]->netconfREGLA = atof(temp); 
		
		//actualizar el valor de aveREGLA y desvREGLA de la regla leida y de las anteriormente procesadas.
		aveREGLA[iPOS][cantidadCCAR[iPOS]] = 0;
		desvREGLA[iPOS][cantidadCCAR[iPOS]] = 0;
		
		for(int i = 0 ; i < cantidadCCAR[iPOS] ; i++)
		{
			aveREGLA[iPOS][cantidadCCAR[iPOS]] += listaCCAR[iPOS][i]->netconfREGLA;
			aveREGLA[iPOS][i] += listaCCAR[iPOS][cantidadCCAR[iPOS]]->netconfREGLA;
		}
		
		cantidadCCAR[iPOS]++;
		
	  eof = fgets(buffer,tBuffer,fichero);
  }  
  
  double mayor = 0;
  // calculando el average de cada regla de cada clase Y LA Clase mas popular
	reglasCUBREN_POS = (CConjunto**)malloc(sizeof(CConjunto*)*cCLASES);
	reglasCUBREN_NEG = (CConjunto**)malloc(sizeof(CConjunto*)*cCLASES);
	reglasCUBREN_POS_CANTIDAD = (CConjunto**)malloc(sizeof(CConjunto*)*cCLASES);

	averageWEIGHT = (double*)malloc(sizeof(double)*cCLASES);
	  
  for (int i = 0 ; i < cCLASES ; i++)
  {
  	averageWEIGHT[i] = 0;
	reglasCUBREN_POS[i] = (CConjunto*)malloc(sizeof(CConjunto));
	reglasCUBREN_POS[i]->longitud = 0;
	reglasCUBREN_POS[i]->tamanno = tamanno_arregloCjt;
	reglasCUBREN_POS[i]->elementos = (int*)malloc(sizeof(int)*reglasCUBREN_POS[i]->tamanno);

	reglasCUBREN_NEG[i] = (CConjunto*)malloc(sizeof(CConjunto));
	reglasCUBREN_NEG[i]->longitud = 0;
	reglasCUBREN_NEG[i]->tamanno = tamanno_arregloCjt;
	reglasCUBREN_NEG[i]->elementos = (int*)malloc(sizeof(int)*reglasCUBREN_NEG[i]->tamanno);

  	reglasCUBREN_POS_CANTIDAD[i] = 0;
  	reglasCUBREN_POS_CANTIDAD[i] = (CConjunto*)malloc(sizeof(CConjunto));
  	reglasCUBREN_POS_CANTIDAD[i]->longitud = 0;
  	reglasCUBREN_POS_CANTIDAD[i]->tamanno = tamanno_arregloCjt;
  	reglasCUBREN_POS_CANTIDAD[i]->elementos = 0;
  	reglasCUBREN_POS_CANTIDAD[i]->elementos = (int*)malloc(sizeof(int)*reglasCUBREN_POS_CANTIDAD[i]->tamanno);
  	
  	for (int j = 0 ; j < cantidadCCAR[i] ; j++)
  	{
  		aveREGLA[i][j] /= (cantidadCCAR[i] - 1);
  		averageWEIGHT[i] += listaCCAR[i][j]->netconfREGLA;
  	}
  	
  	averageWEIGHT[i] /= (1.0*cantidadCCAR[i]);
  	if (mayor < averageWEIGHT[i])
  	{
  		mayor = averageWEIGHT[i];
  		defaultCLASS = i;
  	}
  	  	
  }

  // IMPORTANTE: no usar average netconf para elegir la clase por defecto en FLARE.
  // Reforzar defaultCLASS como la clase mayoritaria (prior máximo) ya leída en Classes.dat.
  {
    int d = 0;
    double bp = -1.0;
    for (int i = 0; i < cCLASES; ++i) {
      if (priorClase[i] > bp) { bp = priorClase[i]; d = i; }
    }
    defaultCLASS = d;
  }
  
  // calculando el principio de la desviacion standard
  for (int i = 0 ; i < cCLASES ; i++)
  {
  	for (int j = 0 ; j < cantidadCCAR[i] ; j++)
  	{
  		for (int k = 0 ; k < j ; k++)
  		{
  			desvREGLA[i][j] += (listaCCAR[i][k]->netconfREGLA - aveREGLA[i][j])*(listaCCAR[i][k]->netconfREGLA - aveREGLA[i][j]);
				desvREGLA[i][k] += (listaCCAR[i][j]->netconfREGLA - aveREGLA[i][k])*(listaCCAR[i][j]->netconfREGLA - aveREGLA[i][k]);
  		}
  	}
  }

  // calculando la desviacion standard e interes de dong li
  for (int i = 0 ; i < cCLASES ; i++)
  {
  	for (int j = 0 ; j < cantidadCCAR[i] ; j++)
  	{
  		desvREGLA[i][j] = sqrt(desvREGLA[i][j]/(cantidadCCAR[i] - 1));
  		listaCCAR[i][j]->interesDONG = listaCCAR[i][j]->netconfREGLA - aveREGLA[i][j];
  		listaCCAR[i][j]->interesDONG = listaCCAR[i][j]->interesDONG - desvREGLA[i][j];
  	}
  }

	//ordenar el conjunto de reglas para cada clase
  for (int i = 0 ; i < cCLASES ; i++)
  {
  	for (int j = 0 ; j < cantidadCCAR[i] - 1 ; j++)
  	{
  		for (int k = cantidadCCAR[i] - 1 ; k > 0 ; k--)
  		{
//  			if(listaCCAR[i][k]->interesDONG > listaCCAR[i][k-1]->interesDONG)
//  			{
//  				CCAR * tmp = listaCCAR[i][k];
//  				listaCCAR[i][k] = listaCCAR[i][k-1];
//  				listaCCAR[i][k - 1] = tmp;
//  			}

  			int lenK  = listaCCAR[i][k]->antecedente->longitud;
  			int lenK1 = listaCCAR[i][k-1]->antecedente->longitud;

  			if ( (lenK > lenK1) ||
  			     (lenK == lenK1 && listaCCAR[i][k]->netconfREGLA > listaCCAR[i][k-1]->netconfREGLA) )
  			{
  			    CCAR * tmp = listaCCAR[i][k];
  			    listaCCAR[i][k] = listaCCAR[i][k-1];
  			    listaCCAR[i][k - 1] = tmp;
  			}

  		}
  	}
  }// fin de la ordenacion por clases
  
  fclose(fichero);
  fichero = 0;
  
  free(buffer);
  buffer = 0;
  free(temp);
  temp = 0;
  
  free(iConjunto->elementos);
  iConjunto->elementos = 0;

  free(iConjunto);
  iConjunto = 0;

	for (int i = 0 ; i < cCLASES ; i++)
	{
		free(aveREGLA[i]);
		aveREGLA[i] = 0;
		free(desvREGLA[i]);
		desvREGLA[i] = 0;  
	}
	
	free(aveREGLA);
	aveREGLA = 0;
	free(desvREGLA); 
	desvREGLA = 0; 
}

void Construye_ClasificadorEXACTO(char * file_clasificador,char * file_prueba, int fold_idx)
{
	// Record a small signature of the fold file so repeated runs can be matched
	// to the correct external partition generated by DivideEn10.
	long long fold_file_size = 0, fold_file_mtime = 0;
	file_signature(file_prueba, &fold_file_size, &fold_file_mtime);

	Leer_Clasificador(file_clasificador);
	printf("Clasificador cargado!!!\n");
	eficienciaEXPERIMENTO = 0;
	int aciertosCASOS = 0, totalCASOS = 0;
	int nearTieCASOS = 0;        // #instances with tieSize>1
	int negEvaluatedCASOS = 0;   // #instances where negative stage was executed (tie region)
	int negCoveredCASOS = 0;     // #instances where at least one tied class had a covering negative rule
	int negPartialUsedCASOS = 0; // #instances where partial negative coverage was used in tie region
	int partialNegCoverCASOS = 0; // total number of partial negative rules selected in tie region
	int partialUsedCASOS = 0;  // #instances where M2 partial coverage fallback was used
	int partialPosCoverCASOS = 0; // total number of positive partial-covering rules selected by fallback

    // --- ALPHA / EPS commented out for experiments ---
    // const double ALPHA = 1.0;   // fuerza del prior (probar 0.5, 1.0, 2.0)
    // const double EPS   = 0.0;   // margen opcional (prueba 0.01 si quieres ser más conservador)

    FILE * ficheroWRITE = fopen("AnalisisInstancias.dat","a");

    // --- NEW: lightweight CSV logs for stability analysis ---
    // One row per test instance. This enables agreement/entropy computation across repeated runs.
    const char* inst_csv_path = "m4d_veto_instance_log.csv";
    FILE* fInstCSV = fopen(inst_csv_path, "a");
    csv_write_header_if_empty(
        fInstCSV,
        inst_csv_path,
        "run_id,fold,fold_file_size,fold_file_mtime,inst_idx,true_class,pred_class,correct,default_used,partial_used,partial_pos_cover,near_tie,tie_size,neg_evaluated,neg_covered,neg_partial_used,partial_neg_cover,best_pos,second_pos,margin,mu0_pos,pos_cover_total,neg_cover_total,chosen_pos,chosen_neg,lambda_eff");

    // One row per fold.
    const char* fold_csv_path = "m4d_veto_fold_summary.csv";
    FILE* fFoldCSV = fopen(fold_csv_path, "a");
    csv_write_header_if_empty(
        fFoldCSV,
        fold_csv_path,
        "run_id,fold,fold_file_size,fold_file_mtime,total,correct,accuracy,default_count,partial_used_count,partial_pos_cover_count,near_tie_count,neg_evaluated_count,neg_covered_count,neg_partial_used_count,partial_neg_cover_count");

    // --- Delimitador por fold (para apendizar resultados de los 10 folds) ---
        	fprintf(ficheroWRITE, "\n==================== FOLD %d | test=%s | reglas=%s ====================\n", fold_idx, file_prueba, file_clasificador);

	// cargar cada ejemplo y clasificarlo
  FILE * fichero = fopen(file_prueba,"r");
  //buffer que almacenara lo leido
  char * buffer = 0;
  buffer = (char *)malloc(sizeof(char)*tBuffer);

  char * temp = 0;
  temp = (char *)malloc(sizeof(char)*30);

  int inicio = 0, j = 0, elem = 0, clase = 0, limite = 0;
  char *eof = 0;
  CConjunto * iConjunto = 0;
  iConjunto = (CConjunto*)malloc(sizeof(CConjunto));
  iConjunto->longitud = 0;
  iConjunto->tamanno = tamanno_arregloCjt;
  iConjunto->elementos = 0;
  iConjunto->elementos = (int*)malloc(sizeof(int)*iConjunto->tamanno);

  //double mayorAVERAGE = 0;

  int claseDEFECTO = 0;
  eof = fgets(buffer,tBuffer,fichero);
  while (eof != 0)
  {
  	inicio = 0;
  	limite = 0;
  	iConjunto->longitud = 0;
  	while (buffer[inicio] != '\n')
  	{
  		// leer el elemento de la transaccion
  		j = 0;
  		while (buffer[inicio]!= ' ' && buffer[inicio]!= '\n')
  		{
  			temp[j] = buffer[inicio];
  			inicio++; j++;
  		}
  		temp[j] = '\0';
  		elem = atoi(temp);
  		if (elem > limite) limite = elem;

  		if (iConjunto->longitud == iConjunto->tamanno)
  		{
  			iConjunto->tamanno *=2;
  			iConjunto->elementos = (int*)realloc(iConjunto->elementos, sizeof(int)*iConjunto->tamanno);
  		}

  		iConjunto->elementos[iConjunto->longitud] = elem;
  		iConjunto->longitud++;

  		if(buffer[inicio]!= '\n')
           inicio++;
  	}

  	totalCASOS++;
  	itemsEJEMPLO = (int*)malloc(sizeof(int)*limite);

  	for (int i = 0 ; i < limite ; i++)
  		itemsEJEMPLO[i] = 0;

  	for (int i = 0 ; i < iConjunto->longitud  - 1; i++)
  		itemsEJEMPLO[iConjunto->elementos[i] - 1] = 1;

  	// === PRINT INSTANCE ===
  	fprintf(ficheroWRITE, "INSTANCIA: ");
  	for (int ii = 0 ; ii < iConjunto->longitud - 1 ; ii++)
  	    fprintf(ficheroWRITE, "%d ", iConjunto->elementos[ii]);
  	fprintf(ficheroWRITE, " | Clase real = %d\n", iConjunto->elementos[iConjunto->longitud - 1]);

  	// clasificar el caso leido segun las reglas cargadas
		clase = iConjunto->elementos[iConjunto->longitud - 1];
		bool partial_used = false;
		int partial_pos_cover = 0;

		// ---------- M4 coverage policy ----------
		// First pass: collect POSITIVE exact coverage only. Negative rules are
		// intentionally NOT evaluated here. For efficiency and conceptual clarity,
		// M4 evaluates negative rules lazily only if the positive stage produces a
		// near-tie between classes.
		for (int i = 0 ; i < cCLASES ; i++)
		{
			reglasCUBREN_POS[i]->longitud = 0;
			reglasCUBREN_NEG[i]->longitud = 0;
			for (int j = 0 ; j < cantidadCCAR[i] ; j++)
			{
				if (listaCCAR[i][j]->negated) continue;
				if (antecedent_covers_exactly(listaCCAR[i][j]->antecedente, itemsEJEMPLO, limite))
				{
					append_rule_index(reglasCUBREN_POS[i], j);
				}
			}
		}

#if M2_PARTIAL_COVERAGE
		int exact_pos_cover = 0;
		for (int i = 0 ; i < cCLASES ; i++) exact_pos_cover += reglasCUBREN_POS[i]->longitud;

		if (exact_pos_cover == 0)
		{
			partial_used = true;
			for (int i = 0 ; i < cCLASES ; i++)
			{
				for (int j = 0 ; j < cantidadCCAR[i] ; j++)
				{
					if (listaCCAR[i][j]->negated) continue; // M4 partial fallback is positive-only; negatives are exact and tie-only.
					if (antecedent_covers_partially_m2(listaCCAR[i][j]->antecedente, itemsEJEMPLO, limite))
					{
						append_rule_index(reglasCUBREN_POS[i], j);
						partial_pos_cover++;
					}
				}
			}
			if (partial_pos_cover == 0) partial_used = false;
		}
#endif

		// Rule coverage details are printed after the tie-breaking stage,
		// so that lazily evaluated negative rules are also visible in AnalisisInstancias.dat.
		// ==================== SELECCIÓN ESTABLE (STABLE OTF) ====================

		claseASIGNADA = -1;
		bool default_used = false;
		bool near_tie = false;
		int tie_size = 0;
		bool neg_evaluated = false;
		bool neg_covered = false;
		bool neg_partial_used = false;
		int partial_neg_cover = 0;
		double best_pos = -1e300;
		double second_pos = -1e300;
		double margin = 0.0;
		double mu0_pos_logged = 0.0;
		int pos_cover_total = 0;
		int neg_cover_total = 0;
		double chosen_pos = -1e300;
		double chosen_neg = 0.0;
		double lambda_eff_chosen = 0.0;

		for (int c = 0; c < cCLASES; ++c) {
			pos_cover_total += reglasCUBREN_POS[c]->longitud;
			neg_cover_total += reglasCUBREN_NEG[c]->longitud;
		}

		// 1) Si ninguna clase cubre esta instancia, usar DEFAULT
		int clasesConCobertura = 0;
		for (int c = 0; c < cCLASES; ++c)
		    if (reglasCUBREN_POS[c]->longitud > 0) clasesConCobertura++;

		if (clasesConCobertura == 0) {
		    claseASIGNADA = clasesCOLECCION[defaultCLASS];
		    default_used = true;
		    claseDEFECTO++;
		} else {
		    // 2) mu0_pos global: promedio de netconf de TODAS las reglas POSITIVAS que cubren esta instancia
		    //    (si el ejemplo está poco cubierto, esto evita que el score dependa de 1 sola regla)
		    double mu0_pos = 0.0;
		    int cnt0 = 0;
		    for (int c = 0; c < cCLASES; ++c) {
		        for (int r = 0; r < reglasCUBREN_POS[c]->longitud; ++r) {
		            int idx = reglasCUBREN_POS[c]->elementos[r];
		            mu0_pos += listaCCAR[c][idx]->netconfREGLA;
		            cnt0++;
		        }
		    }
		    if (cnt0 > 0) mu0_pos /= (double)cnt0;
		    mu0_pos_logged = mu0_pos;

		    // 3) Selección en 2 etapas:
		    //    - Primero decidimos SOLO con evidencia positiva (score estable POS).
		    //    - SOLO si hay empate por evidencia positiva (dentro de STABLE_TIE_EPS),
		    //      usamos CARs negativas para deshacer el empate.
		    //    Esto evita que la evidencia negativa "mate" la clase correcta cuando
		    //    ya hay un ganador claro por soporte/positivas.
		    double *posScore = (double*)malloc(sizeof(double) * cCLASES);
		    for (int c = 0; c < cCLASES; ++c) posScore[c] = -1e300;

		    double bestPos = -1e300;
		    int bestPosC = -1;
		    // Base score includes a small prior term to avoid over-predicting minority classes in highly imbalanced datasets (e.g., FLARE).
		    double bestBase = -1e300;
		    int bestBaseC = -1;
		    for (int c = 0; c < cCLASES; ++c) {
		        double p = stable_score_pos_class(c, reglasCUBREN_POS, listaCCAR, mu0_pos);
		        posScore[c] = p;
		        if (p <= -1e200) continue;
		        // Track pure-positive winner (for near-tie detection)

		        // Find the true maximum by positives (do NOT use the behavioral tie epsilon here).
		        if (bestPosC == -1 || p > bestPos + STABLE_NUM_EPS) {
		            bestPos = p;
		            bestPosC = c;
		        } else if (fabs(p - bestPos) <= STABLE_NUM_EPS) {
		            // desempate determinista por id de clase (solo para fijar bestPosC)
		            if (clasesCOLECCION[c] < clasesCOLECCION[bestPosC]) bestPosC = c;
		        }

		        // Track base-score winner (positives + small prior term) for the final decision when there is a clear winner by positives.
		        double base = p + STABLE_LAMBDA_PRIOR * log(priorClase[c]);
		        if (bestBaseC == -1 || base > bestBase + STABLE_NUM_EPS) {
		            bestBase = base;
		            bestBaseC = c;
		        } else if (fabs(base - bestBase) <= STABLE_NUM_EPS) {
		            if (clasesCOLECCION[c] < clasesCOLECCION[bestBaseC]) bestBaseC = c;
		        }
		    }

		    // Track the second-best positive score (for margin / ambiguity diagnostics)
		    second_pos = -1e300;
		    for (int c = 0; c < cCLASES; ++c) {
		        if (posScore[c] <= -1e200) continue;
		        if (c == bestPosC) continue;
		        if (second_pos <= -1e200 || posScore[c] > second_pos + STABLE_NUM_EPS) {
		            second_pos = posScore[c];
		        }
		    }
		    best_pos = bestPos;
		    if (second_pos > -1e200) margin = bestPos - second_pos;

		    if (bestPosC == -1) {
		        free(posScore);
		        claseASIGNADA = clasesCOLECCION[defaultCLASS];
		        default_used = true;
		        claseDEFECTO++;
		    } else {

		        // Construir conjunto de clases empatadas por evidencia positiva
		        int tieCount = 0;
		        for (int c = 0; c < cCLASES; ++c) {
		            if (posScore[c] <= -1e200) continue;
		            // Near-tie set: classes whose positive score is close to the best.
		            if ((bestPos - posScore[c]) <= STABLE_TIE_EPS) tieCount++;
		        }
		        tie_size = tieCount;
		        near_tie = (tieCount > 1);

		        if (tieCount <= 1 || !M4_USE_NEGATIVES_FOR_TIEBREAK) {
		            // M2: decide with positive evidence plus the class prior. If there is a
		            // near-tie but negatives are disabled, restrict the prior tie-breaker to
		            // the tied classes.
		            int chosenC = bestBaseC;
		            if (tieCount > 1 && !M4_USE_NEGATIVES_FOR_TIEBREAK) {
		                double bestScoreTied = -1e300;
		                chosenC = -1;
		                for (int c = 0; c < cCLASES; ++c) {
		                    if (posScore[c] <= -1e200) continue;
		                    if ((bestPos - posScore[c]) > STABLE_TIE_EPS) continue;
		                    double sc = posScore[c] + STABLE_LAMBDA_PRIOR * log(priorClase[c]);
		                    if (chosenC == -1 || sc > bestScoreTied + STABLE_NUM_EPS) {
		                        bestScoreTied = sc;
		                        chosenC = c;
		                    } else if (fabs(sc - bestScoreTied) <= STABLE_NUM_EPS) {
		                        if (clasesCOLECCION[c] < clasesCOLECCION[chosenC]) chosenC = c;
		                    }
		                }
		            }
		            claseASIGNADA = clasesCOLECCION[chosenC];
		            chosen_pos = posScore[chosenC];
		            free(posScore);
		        } else {
		            neg_evaluated = true;
		            // M4d: Empate por positivas => evaluar negativas SOLO para clases empatadas.
		            // Las negativas se usan como VETO conservador contra la clase ganadora
		            // por positivas+prior. No se reordenan todas las clases por menor evidencia
		            // negativa; solo se cambia la decisión si la ganadora está fuertemente
		            // negada y otra clase empatada tiene evidencia negativa claramente menor.
		            // Para eficiencia y estabilidad, solo se acepta cobertura EXACTA.
		            for (int c = 0; c < cCLASES; ++c) {
		                reglasCUBREN_NEG[c]->longitud = 0;
		                if (posScore[c] <= -1e200) continue;
		                if ((bestPos - posScore[c]) > STABLE_TIE_EPS) continue;

		                for (int j = 0; j < cantidadCCAR[c]; ++j) {
		                    if (!listaCCAR[c][j]->negated) continue;
		                    if (antecedent_covers_exactly(listaCCAR[c][j]->antecedente, itemsEJEMPLO, limite)) {
		                        append_rule_index(reglasCUBREN_NEG[c], j);
		                    }
		                }
		            }

		            // 1) Decisión base dentro del conjunto empatado: positiva + prior.
		            int baseC = -1;
		            double bestBaseTie = -1e300;
		            for (int c = 0; c < cCLASES; ++c) {
		                if (posScore[c] <= -1e200) continue;
		                if ((bestPos - posScore[c]) > STABLE_TIE_EPS) continue;
		                double baseTie = posScore[c] + STABLE_LAMBDA_PRIOR * log(priorClase[c]);
		                if (baseC == -1 || baseTie > bestBaseTie + STABLE_NUM_EPS) {
		                    bestBaseTie = baseTie;
		                    baseC = c;
		                } else if (fabs(baseTie - bestBaseTie) <= STABLE_NUM_EPS) {
		                    if (clasesCOLECCION[c] < clasesCOLECCION[baseC]) baseC = c;
		                }
		            }

		            // 2) Buscar una alternativa empatada con menor evidencia negativa exacta.
		            //    M4d NO escoge globalmente la clase con menor negativa. Primero conserva
		            //    la ganadora base (positiva+prior) y solo la veta si ella está
		            //    fuertemente negada.
		            int minNegC = -1;
		            double minNeg = 1e300;
		            for (int c = 0; c < cCLASES; ++c) {
		                if (posScore[c] <= -1e200) continue;
		                if ((bestPos - posScore[c]) > STABLE_TIE_EPS) continue;
		                if (reglasCUBREN_NEG[c]->longitud > 0) neg_covered = true;
		                double negSc = stable_score_neg_class(c, reglasCUBREN_NEG, listaCCAR);
		                if (minNegC == -1 || negSc < minNeg - STABLE_NUM_EPS) {
		                    minNeg = negSc;
		                    minNegC = c;
		                } else if (fabs(negSc - minNeg) <= STABLE_NUM_EPS) {
		                    // empate de evidencia negativa: preferir la clase base; si no,
		                    // la de mayor score base positivo+prior dentro del empate.
		                    if (c == baseC) {
		                        minNegC = c;
		                    } else if (minNegC != baseC) {
		                        double baseTieC = posScore[c] + STABLE_LAMBDA_PRIOR * log(priorClase[c]);
		                        double baseTieMin = posScore[minNegC] + STABLE_LAMBDA_PRIOR * log(priorClase[minNegC]);
		                        if (baseTieC > baseTieMin + STABLE_NUM_EPS) minNegC = c;
		                        else if (fabs(baseTieC - baseTieMin) <= STABLE_NUM_EPS && clasesCOLECCION[c] < clasesCOLECCION[minNegC]) minNegC = c;
		                    }
		                }
		            }

		            // 3) M4d: veto negativo contra la clase base, no re-ranking general.
		            //    Se cambia la decisión SOLO si:
		            //      a) la clase base tiene evidencia negativa exacta suficiente;
		            //      b) existe una alternativa empatada con evidencia negativa claramente menor;
		            //      c) la alternativa no es la propia clase base.
		            int bestC = baseC;
		            if (baseC != -1 && minNegC != -1 && minNegC != baseC) {
		                double negBase = stable_score_neg_class(baseC, reglasCUBREN_NEG, listaCCAR);
		                if (negBase >= M4_NEG_VETO_MIN && (negBase - minNeg) >= M4_NEG_VETO_DIFF) {
		                    bestC = minNegC;
		                }
		            }

		            if (bestC == -1) {
		                claseASIGNADA = clasesCOLECCION[defaultCLASS];
		                default_used = true;
		                claseDEFECTO++;
		            } else {
		                claseASIGNADA = clasesCOLECCION[bestC];
		                chosen_pos = posScore[bestC];
		                chosen_neg = stable_score_neg_class(bestC, reglasCUBREN_NEG, listaCCAR);
		                lambda_eff_chosen = 1.0; // M4d: veto discreto, sin penalización continua.
		            }
		            free(posScore);
		        }
		    }

		}


		// === PRINT RULES THAT COVER THE INSTANCE (POS and NEG with netconf) ===
		// Printed after the decision stage because negative rules are evaluated lazily
		// only in the positive near-tie region.
		neg_cover_total = 0;
		for (int c = 0; c < cCLASES; ++c) neg_cover_total += reglasCUBREN_NEG[c]->longitud;
		fprintf(ficheroWRITE, "M4d partial POS coverage used = %s | partial POS rules = %d\n", partial_used ? "YES" : "NO", partial_pos_cover);
		fprintf(ficheroWRITE, "M4d negative veto evaluated = %s | negative covered = %s | partial NEG used = %s | partial NEG rules = %d\n",
		        neg_evaluated ? "YES" : "NO", neg_covered ? "YES" : "NO", neg_partial_used ? "YES" : "NO", partial_neg_cover);
		fprintf(ficheroWRITE, "Reglas que cubren la instancia (POS/NEG):\n");
		for (int c = 0; c < cCLASES; c++)
		{
		    fprintf(ficheroWRITE, "  Clase %d:\n", clasesCOLECCION[c]);

		    fprintf(ficheroWRITE, "    [+] POS: ");
		    if (reglasCUBREN_POS[c]->longitud == 0) {
		        fprintf(ficheroWRITE, "NINGUNA\n");
		    } else {
		        fprintf(ficheroWRITE, "{ ");
		        for (int r = 0; r < reglasCUBREN_POS[c]->longitud; r++) {
		            int idx = reglasCUBREN_POS[c]->elementos[r];
		            fprintf(ficheroWRITE, "#%d(", idx);
		            for (int a = 0; a < listaCCAR[c][idx]->antecedente->longitud; a++)
		                fprintf(ficheroWRITE, "%d ", listaCCAR[c][idx]->antecedente->elementos[a]);
		            fprintf(ficheroWRITE, ") [netconf=%.4f] ", listaCCAR[c][idx]->netconfREGLA);
		        }
		        fprintf(ficheroWRITE, "}\n");
		    }

		    fprintf(ficheroWRITE, "    [-] NEG: ");
		    if (reglasCUBREN_NEG[c]->longitud == 0) {
		        fprintf(ficheroWRITE, "NINGUNA\n");
		    } else {
		        fprintf(ficheroWRITE, "{ ");
		        for (int r = 0; r < reglasCUBREN_NEG[c]->longitud; r++) {
		            int idx = reglasCUBREN_NEG[c]->elementos[r];
		            fprintf(ficheroWRITE, "#%d(", idx);
		            for (int a = 0; a < listaCCAR[c][idx]->antecedente->longitud; a++)
		                fprintf(ficheroWRITE, "%d ", listaCCAR[c][idx]->antecedente->elementos[a]);
		            fprintf(ficheroWRITE, ") -> neg(%d) [netconf=%.4f] ", clasesCOLECCION[c], listaCCAR[c][idx]->netconfREGLA);
		        }
		        fprintf(ficheroWRITE, "}\n");
		    }
		}

  	// chequear clasificacion realizada con la clase que traia el ejemplo
	int correct = (clase == claseASIGNADA) ? 1 : 0;
	if (correct) aciertosCASOS++;

	// Update per-fold stability counters
	if (partial_used) { partialUsedCASOS++; partialPosCoverCASOS += partial_pos_cover; }
	if (near_tie) nearTieCASOS++;
	if (neg_evaluated) negEvaluatedCASOS++;
	if (neg_covered) negCoveredCASOS++;
	if (neg_partial_used) { negPartialUsedCASOS++; partialNegCoverCASOS += partial_neg_cover; }

	// Write lightweight instance row (CSV)
	if (fInstCSV) {
	    fprintf(fInstCSV,
	        "%d,%d,%lld,%lld,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%.10g,%.10g,%.10g,%.10g,%d,%d,%.10g,%.10g,%.10g\n",
	        STABILITY_RUN_ID,
	        fold_idx,
	        fold_file_size,
	        fold_file_mtime,
	        totalCASOS,
	        clase,
	        claseASIGNADA,
	        correct,
	        default_used ? 1 : 0,
	        partial_used ? 1 : 0,
	        partial_pos_cover,
	        near_tie ? 1 : 0,
	        tie_size,
	        neg_evaluated ? 1 : 0,
	        neg_covered ? 1 : 0,
	        neg_partial_used ? 1 : 0,
	        partial_neg_cover,
	        best_pos,
	        second_pos,
	        margin,
	        mu0_pos_logged,
	        pos_cover_total,
	        neg_cover_total,
	        chosen_pos,
	        chosen_neg,
	        lambda_eff_chosen);
	}
  	fprintf(ficheroWRITE, "Clase = %i, ASIGNADA = %i\n", clase,claseASIGNADA);
  	fprintf(ficheroWRITE, "Aciertos = %i de Total = %i\n",aciertosCASOS, totalCASOS);
  	fprintf(ficheroWRITE, "Casos clasificador con clase por defecto = %i\n",claseDEFECTO);

  	eof = fgets(buffer,tBuffer,fichero);
  	free(itemsEJEMPLO);
  	itemsEJEMPLO = 0;
  }
	// cargar cada ejemplo y clasificarlo

	// calcular eficacia de la clasificacion del grupo de ejemplos
	eficienciaEXPERIMENTO = (aciertosCASOS*100.0)/totalCASOS;

	// Write fold summary row (CSV)
	if (fFoldCSV) {
	    fprintf(fFoldCSV, "%d,%d,%lld,%lld,%d,%d,%.6f,%d,%d,%d,%d,%d,%d,%d,%d\n",
	        STABILITY_RUN_ID,
	        fold_idx,
	        fold_file_size,
	        fold_file_mtime,
	        totalCASOS,
	        aciertosCASOS,
	        eficienciaEXPERIMENTO,
	        claseDEFECTO,
	        partialUsedCASOS,
	        partialPosCoverCASOS,
	        nearTieCASOS,
	        negEvaluatedCASOS,
	        negCoveredCASOS,
	        negPartialUsedCASOS,
	        partialNegCoverCASOS);
	    fflush(fFoldCSV);
	}
	if (fInstCSV) { fflush(fInstCSV); fclose(fInstCSV); fInstCSV = NULL; }
	if (fFoldCSV) { fclose(fFoldCSV); fFoldCSV = NULL; }

	for (int i = 0 ; i < cCLASES ; i++)
	{
		free(reglasCUBREN_POS[i]->elementos);
		reglasCUBREN_POS[i]->elementos = 0;
		free(reglasCUBREN_POS[i]);
		reglasCUBREN_POS[i] = 0;

		free(reglasCUBREN_NEG[i]->elementos);
		reglasCUBREN_NEG[i]->elementos = 0;
		free(reglasCUBREN_NEG[i]);
		reglasCUBREN_NEG[i] = 0;

		free(reglasCUBREN_POS_CANTIDAD[i]->elementos);
		reglasCUBREN_POS_CANTIDAD[i]->elementos = 0;
		free(reglasCUBREN_POS_CANTIDAD[i]);
		reglasCUBREN_POS_CANTIDAD[i] = 0;
		for (int j = 0 ; j < cantidadCCAR[i] ; j++)
		{
			free(listaCCAR[i][j]->antecedente->elementos);
			listaCCAR[i][j]->antecedente->elementos = 0;
			free(listaCCAR[i][j]->antecedente);
			listaCCAR[i][j]->antecedente = 0;
			free(listaCCAR[i][j]);
			listaCCAR[i][j] = 0;

		}

		free(listaCCAR[i]);
		listaCCAR[i] = 0;

	}

	free(reglasCUBREN_POS);
	reglasCUBREN_POS = 0;
	free(reglasCUBREN_POS_CANTIDAD);
	reglasCUBREN_POS_CANTIDAD = 0;
	free(reglasCUBREN_NEG);
	reglasCUBREN_NEG = 0;
	free(listaCCAR);
	listaCCAR = 0;

	free(cantidadCCAR);
	cantidadCCAR = 0;
	free(totalCCAR);
	totalCCAR = 0;

	free(averageWEIGHT);
	averageWEIGHT = 0;

	fclose(fichero);
	fichero = 0;
	fclose(ficheroWRITE);
	ficheroWRITE = 0;
	free(buffer);
	buffer = 0;
	free(temp);
	temp = 0;

	free(iConjunto->elementos);
	iConjunto->elementos = 0;
	free(iConjunto);
	iConjunto = 0;
}

void Clasifica_Dataset(int argc, char *argv[])
{
  printf("Iniciando...\n");	
  Leer_File_CLASSES();
	eficienciaPROMEDIO = 0;
  printf("readed...\n");

	nombres_files = (char**)malloc(sizeof(char*)*10);
	nombres_REGLAS = (char**)malloc(sizeof(char*)*10);

  // Start each full 10-fold run with a clean instance-analysis file.
  // Each fold is then appended inside Construye_ClasificadorEXACTO(...).
  FILE * fAnalisisReset = fopen("AnalisisInstancias.dat", "w");
  if (fAnalisisReset) {
    fprintf(fAnalisisReset, "ANALISIS DE INSTANCIAS - M4d POSITIVE PARTIAL COVERAGE + NEGATIVE VETO TIE-BREAK\n");
    fclose(fAnalisisReset);
  }

  FILE * ficheroWRITE = 0;
  ficheroWRITE = fopen(argv[1],"w");
	
	for (int i = 0 ; i < 10 ; i++)
	{
  	nombres_files[i] = (char*)malloc(sizeof(char)*10);
  	sprintf(nombres_files[i],"%i.dat",i+1);
		nombres_REGLAS[i] = (char*)malloc(sizeof(char)*20);
  	sprintf(nombres_REGLAS[i],"ReglasDataset%i.dat",i+1);		

		Construye_ClasificadorEXACTO(nombres_REGLAS[i],nombres_files[i],i+1);
		eficienciaPROMEDIO += eficienciaEXPERIMENTO;
		
		fprintf(ficheroWRITE, "%s\n",nombres_files[i]);
		fprintf(ficheroWRITE, "Eficiencia obtenida = %.4f\n\n",eficienciaEXPERIMENTO);
		fflush(ficheroWRITE);
		printf("Eficiencia obtenida = %.4f\n\n",eficienciaEXPERIMENTO);
		
		
	}
	printf("Clasificado dataset!!\n");
	
	eficienciaPROMEDIO /= 10.0;
	fprintf(ficheroWRITE, "Eficiencia promedio del proceso = %.4f\n", eficienciaPROMEDIO);
	printf("Eficiencia promedio del proceso = %.4f\n", eficienciaPROMEDIO);
	fflush(ficheroWRITE);
	fclose(ficheroWRITE);
	ficheroWRITE = 0;
	for (int i = 0 ; i < 10 ; i++)
	{
		free(nombres_files[i]);
		nombres_files[i] = 0;
		free(nombres_REGLAS[i]);
		nombres_REGLAS[i] = 0;
	}
	free(nombres_files);
	nombres_files = 0;
	free(nombres_REGLAS);
	nombres_REGLAS = 0;

	free(clasesCOLECCION);
	clasesCOLECCION = 0;

    free(priorClase);
    priorClase = 0;
}

// implementacion de los metodos
int main(int argc, char *argv[])
{
  if (process_arguments(argc,argv))
    return 1;
  
	Clasifica_Dataset(argc,argv);
  return EXIT_SUCCESS;
}
