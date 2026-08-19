/*
 ============================================================================
 Name        : MaximalesDFS.c
 Author      : rhl
 Version     : 1.0
 Copyright   : Your copyright notice
 Description : Hello World in C, Ansi-style
 ============================================================================
 */

#pragma GCC target("sse4.2")
#include <stdio.h>
#include <stdlib.h>

#include <string.h>
#include <math.h>
#include <sys/timeb.h>
#include <nmmintrin.h>

#define MAX_ITEMS 15						// Máxima longitud permitida por defecto, para un itemset.
#define POW_DOS_MAX_ITEMS (1 << MAX_ITEMS)  // Dos elevado a la 15, para la cantidad de combinaciones.
#define MAX_UNIQUE_ITEMS 100000	    		// Máxima cantidad de ítems, por defecto.
#define MAX_TRANSACTIONS 1000000			// Máxima cantidad de transacciones, por defecto.
#define MAX_MAXIMAL 200000					// Máxima cantidad de maximales, por defecto.

#ifndef TOPK_SUBRULES
#define TOPK_SUBRULES 10   /* Top-K subreglas por maximal */
#endif

#ifndef MIN_NETCONF
#define MIN_NETCONF 0.0    /* umbral de Netconf positivas (ej. 0.01) */
#endif

/*
 * M4 = M2 + reglas negativas formales para desempate en el clasificador.
 * El minador produce:
 *   - una regla positiva directa por maximal etiquetado: M={c+} U X ==> X -> c+
 *   - reglas negativas derivadas del MISMO antecedente maximal X: X -> neg(c)
 *     si Netconf(X -> c) < 0 para clases alternativas c.
 * No se generan subreglas. No se usa ningun umbral adicional para negativas: el corte teorico
 * es la independencia Netconf=0.
 */
#ifndef M4_DIRECT_MAXIMAL_POS_NEG
#define M4_DIRECT_MAXIMAL_POS_NEG 1
#endif

#ifndef M4_GENERATE_NEGATIVES
#define M4_GENERATE_NEGATIVES 1
#endif

#define BLOOM_FILTER_SIZE 1000000			// Tamaño del filtro BLOOM.

/*
 * Código para chequear fuga de memoria
 */
/*
#define DEBUG_MEMORY
#ifdef DEBUG_MEMORY
static unsigned int memory_allocated = 0;

typedef struct {
    unsigned int size;
    unsigned char data[]; // Datos reales
} mem_block;

void* track_malloc(unsigned int size) {
	mem_block* block = (mem_block*) malloc(sizeof(mem_block) + size);
	if (block)
	{
	    block->size = size;
	    memory_allocated += size;
	    printf("[MALLOC] Allocated: %u bytes at %p, Total: %u\n",
	                 size, (void*)block->data, memory_allocated);
	    return block->data;
	}
	return NULL;
}

void* track_calloc(unsigned int nmemb, unsigned int size) {
	unsigned int total_size = nmemb * size;
    mem_block* block = (mem_block*)calloc(1, sizeof(mem_block) + total_size);
    if (block)
    {
        block->size = total_size;
        memory_allocated += total_size;
        printf("[CALLOC] Allocated: %u bytes (%u x %u) at %p, Total: %u\n",
               total_size, nmemb, size, (void*)block->data, memory_allocated);
        return block->data;
    }
    return NULL;
}

void* track_realloc(void* ptr, unsigned int new_size) {
	unsigned int old_size = 0;

    if (ptr == NULL)
	    return track_malloc(new_size);

	// Guardamos el tamaño antiguo antes del realloc
	mem_block* old_block = (mem_block*)((unsigned char*)ptr - sizeof(mem_block));
	old_size = old_block->size;

	// Realizamos el realloc
	mem_block* new_block = (mem_block*)realloc(old_block, sizeof(mem_block) + new_size);

	if (new_block)
	{
	    memory_allocated -= old_size;
	    memory_allocated += new_size;
	    new_block->size = new_size;
	    printf("[REALLOC] Resized: %u -> %u bytes at %p, Total: %u\n",
	           old_size, new_size, (void*)new_block->data, memory_allocated);
	    return new_block->data;
	}
	return NULL;
}

void track_free(void* ptr) {
	if (ptr)
	{
	    mem_block* block = (mem_block*)((unsigned char*)ptr - sizeof(mem_block));
	    memory_allocated -= block->size;
	    printf("[FREE] Freed: %u bytes at %p, Total: %u\n",
	               block->size, ptr, memory_allocated);
	        free(block);
	}
}

#define malloc(size) track_malloc(size)
#define calloc(nmemb, size) track_calloc(nmemb, size)
#define realloc(ptr, size) track_realloc(ptr, size)
#define free(ptr) track_free(ptr)
#endif
*/
/*
 * Código para chequear fuga de memoria
 */

// Definición de la estructura Itemset
typedef struct {
	unsigned int items[MAX_ITEMS];
    unsigned int sup;
    unsigned int supAntec;
    char size;
} Itemset;

// Definición de la estructura Transaction
typedef struct
{
   unsigned int count;
   unsigned int* items;
} Transaction;

// Definición de la estructura Clase
typedef struct
{
   unsigned int item;
   unsigned int sup;
} Clase;

//Definición de la estructura MaximalList
typedef struct {
    Itemset sets[MAX_MAXIMAL];
    unsigned int count;
} MaximalList;


//Definición de la estructura Bloom
typedef struct {
    Itemset** sets;
    unsigned int sets_count;
    char flat;
} Bloom;

//Definición de la estructura CAR
typedef struct {
    Itemset maximal;
    Clase consecuente;
    double soporte;
    double soporteAntec;
    double soporteConsec;
    double confianza;
    double netconf;
    double score;   /* score estable para ranking */
    double lift;
    double conviction;
    unsigned char negated; /* 0: A->c, 1: A->neg(c) */
} CAR;

typedef struct
{
   unsigned int count;
   unsigned int* items;
   unsigned int clase;
   //CAR* reglasQueLaCubren;
} Instance;

// Definición de la estructura Matriz Binaria
typedef struct
{
   unsigned int* bti_data;
   unsigned int** bti;
   unsigned int btiCountRow;
   unsigned int btiCountColumns;
   char btiItemsCountLastRow;
} Bti;

// Definición de la estructura UnoFrecuentes
typedef struct
{
	unsigned int* items;
	unsigned int items_count;
	unsigned int* items_support;
	unsigned int* map1_N;
} UnoFrecuentes;

// Definición de la estructura Block
typedef struct
{
   unsigned int value;
   unsigned int number;
} Block;


struct timeb T1, T2;
UnoFrecuentes* uno_frecuentes;
MaximalList* maximal_list[MAX_ITEMS + 1];
Bloom* bloom;
unsigned int* tempHash;
Transaction* transactions;
unsigned int transactions_count;
Instance* instances;
unsigned int instances_count;
unsigned int maximalTotal;
Clase* clases;
CAR* CARS;
unsigned int clases_count;
unsigned int total_CARs;

/* ========================
 *  Deduplicación de reglas
 * ========================
 *
 * Motivo:
 *  Una misma regla X -> Y puede generarse desde varios maximales distintos.
 *  Eso provoca que aparezca repetida varias veces en el archivo de salida.
 *
 * Estrategia:
 *  - Se deduplica por clave (tamaño, consecuente, items del antecedente).
 *  - Si hay duplicado positivo, se conserva la versión con mayor Netconf.
 *  - Si hay duplicado negativo, se conserva la versión con menor Netconf
 *    porque una negativa más fuerte tiene Netconf más negativo.
 */

static unsigned int next_pow2_u32(unsigned int v)
{
    if (v == 0) return 1;
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v++;
    return v;
}

static unsigned long long fnv1a64_step(unsigned long long h, unsigned int x)
{
    // FNV-1a 64-bit sobre palabras de 32 bits
    h ^= (unsigned long long)x;
    h *= 1099511628211ULL;
    return h;
}


/* Score estable: penaliza reglas con poco soporte para reducir varianza entre folds.
   Mantiene netconf como medida de calidad (dirección) pero suaviza por robustez del conteo. */
static inline double stable_score_from_counts(double netconf, unsigned int supXY, unsigned int min_support_abs)
{
    const double beta = (min_support_abs > 0 ? (double)min_support_abs : 1.0);
    /* shrinkage: sup/(sup+beta) ∈ (0,1) */
    double shrink = (double)supXY / ((double)supXY + beta);
    return netconf * shrink;
}


static void fill_metrics_from_counts(CAR* c, unsigned int supXY, unsigned int supX, unsigned int supY, unsigned int N);
static void topk_push(CAR* heap, int* heapSize, int K, const CAR* cand);

/* Hash and deduplication helpers for the active direct-maximal rule generator. */
static unsigned long long hash_rule_key(const CAR* r)
{
    // Clave: size (incluye clase), antecedente (items[1..size-1]), clase (items[0])
    unsigned long long h = 1469598103934665603ULL; // offset basis
    h = fnv1a64_step(h, (unsigned int)(unsigned char)r->maximal.size);
    h = fnv1a64_step(h, r->maximal.items[0]);
    h = fnv1a64_step(h, (unsigned int)r->negated);
    for (int i = 1; i < r->maximal.size; i++) {
        h = fnv1a64_step(h, r->maximal.items[i]);
    }
    return h;
}

static int same_rule_key(const CAR* a, const CAR* b)
{
    if (a->maximal.size != b->maximal.size) return 0;
    if (a->maximal.items[0] != b->maximal.items[0]) return 0; // clase
    if (a->negated != b->negated) return 0;
    for (int i = 1; i < a->maximal.size; i++) {
        if (a->maximal.items[i] != b->maximal.items[i]) return 0;
    }
    return 1;
}

static void dedup_CARS_keep_best_netconf(void)
{
    if (total_CARs == 0 || CARS == NULL) return;

    // Tamaño de tabla hash ~ 2x elementos, potencia de 2
    unsigned int tableSize = next_pow2_u32(total_CARs * 2u);
    unsigned long long* keys = (unsigned long long*) calloc(tableSize, sizeof(unsigned long long));
    int* values = (int*) malloc(tableSize * sizeof(int));
    unsigned char* used = (unsigned char*) calloc(tableSize, sizeof(unsigned char));
    if (!keys || !values || !used) {
        fprintf(stderr, "No hay memoria para deduplicar reglas.\n");
        exit(EXIT_FAILURE);
    }

    CAR* unique = (CAR*) malloc(total_CARs * sizeof(CAR));
    if (!unique) {
        fprintf(stderr, "No hay memoria para buffer de deduplicación.\n");
        exit(EXIT_FAILURE);
    }
    unsigned int uniqueCount = 0;

    for (unsigned int i = 0; i < total_CARs; i++) {
        CAR* r = &CARS[i];
        unsigned long long k = hash_rule_key(r);
        unsigned int mask = tableSize - 1u;
        unsigned int pos = (unsigned int)k & mask;

        while (used[pos]) {
            if (keys[pos] == k && same_rule_key(&unique[values[pos]], r)) {
                // Duplicado: conservar el mas fuerte segun el tipo de regla.
                int idx = values[pos];
                int replace = 0;
                if (!r->negated) {
                    // Positiva: mayor Netconf = evidencia positiva mas fuerte.
                    replace = (r->netconf > unique[idx].netconf);
                } else {
                    // Negativa: menor Netconf = dependencia negativa mas fuerte.
                    replace = (r->netconf < unique[idx].netconf);
                }
                if (replace) {
                    unique[idx] = *r;
                }
                goto next_rule;
            }
            pos = (pos + 1u) & mask;
        }

        used[pos] = 1;
        keys[pos] = k;
        values[pos] = (int)uniqueCount;
        unique[uniqueCount++] = *r;

    next_rule:
        ;
    }

    // Copiar de vuelta
    memcpy(CARS, unique, uniqueCount * sizeof(CAR));
    total_CARs = uniqueCount;

    free(unique);
    free(keys);
    free(values);
    free(used);
}

/*
 * Variante I
 */
typedef struct {
    Block* blocks;
    unsigned int count;
} BlockList;
/*
 * Variante I
 */

/*
 * Variante II
 */
/*
// Cambiamos Bloom por un índice invertido
typedef struct {
    Itemset** list;     // Lista de punteros a conjuntos maximales
    unsigned int count; // Número de maximales en la lista
    unsigned int size;  // Tamaño asignado
} InvertedIndexItem;

InvertedIndexItem* inverted_index; // Índice invertido global

// Pool de bloques para gestión eficiente de memoria
#define MAX_BLOCKS 10000000 // 10 millones de bloques
Block* block_pool;          // Pool de bloques preasignado
unsigned int block_pool_index = 0; // Índice actual en el pool
*/
/*
 * Variante II
 */

/* Activa los bits de la matriz compactada "b". Recorre las transacciones y para cada ítem,
 * si es frecuente (map1_N[item] > 0), activa el bit correspondiente en "b".
 */
void btiItem(const char* fileName, Bti* b, unsigned int* map1_N)
{
   for(unsigned int i = 0; i < transactions_count; i++)
   {
      for(unsigned int j = 0; j < transactions[i].count; j++)
      {
    	  if(map1_N[transactions[i].items[j]] > 0)
    	      b->bti[map1_N[transactions[i].items[j]] - 1][i / 32] |= 1U << (i % 32);
      }
   }
}

void read_transactions(const char* fileName, float min_support, Bti* b) {


	FILE* file = fopen(fileName, "r");
    if (!file) {
        perror("Error abriendo el fichero");
        exit(EXIT_FAILURE);
    }

    unsigned int unique_item_count = 0;
    unsigned int* unique_items = (unsigned int*) malloc(MAX_UNIQUE_ITEMS * sizeof(unsigned int));
    unsigned int* unique_supports = (unsigned int*) calloc(MAX_UNIQUE_ITEMS, sizeof(unsigned int));
    clases = (Clase*) malloc (40 * sizeof(Clase));
    clases_count = 0;
    char* buffer = (char*) malloc(15000 * sizeof(char));
    char* ptr;
    char* eof = fgets(buffer,15000, file);
    unsigned int item, max_item = 0, max_support = 0;
    unsigned int* unique_check = (unsigned int*) calloc(MAX_UNIQUE_ITEMS, sizeof(unsigned int));     // bandera para saber si un ítem existe en la BD
    unsigned int** support_inv;													// matrix de soporte invertida para ordenar por soporte en o(n), n la cantidad de items
    unsigned int* support_inv_count;
    unsigned int* support_inv_size;

    /* se lee la BD
     * se almacenan los ítems únicos en unique_items y los soportes en unique_supports
     * se ignora el ítem con valor mayor a MAX_UNIQUE_ITEMS (retorno silencioso)
     * se almacena el valor máximo de los items en max_item y el valor máximo del soporte en max_support
     * se almacena la cantidad de transacciones en transaction_count.
     * se trunca la lectura de la BD si se intenta leer más de MAX_TRANSACTIONS transacciones (se emite un mensaje)
      * */
    transactions_count = 0;

    while (eof != NULL) {
    	transactions[transactions_count].items = (unsigned int*) malloc (MAX_UNIQUE_ITEMS * sizeof(unsigned int));
    	transactions[transactions_count].count = 0;
    	ptr = buffer;
    	while((*ptr) & 32)
        {
    	    item = 0;
    	    while((*ptr) & 16)
    	        item = item * 10 + (*ptr++) - 48;

    	    if((*ptr) & 32) ptr++;

    	    if(item < MAX_UNIQUE_ITEMS)
    	    {
    	        if(item > max_item)
    	        	max_item = item;

    	        if (unique_check[item] == 0)
    	        {
    	            unique_items[unique_item_count++] = item;
    	            unique_check[item] = 1;
    	        }

    	        unique_supports[item]++;
    	        if(unique_supports[item] > max_support)
    	           max_support = unique_supports[item];

    	        transactions[transactions_count].items[transactions[transactions_count].count++] = item;
    	    }
    	}

    	// Después de finalizada cada transacción tomo el último ítem (la clase)
    	// y lo busco en las clases. Si lo encuentro, incremento el soporte y si no, lo añado.
    	// Además, lo quito de los 1-frecuentes, dándole un tratamiento diferente al de los ítems
    	unique_check[item] = 0;
    	unique_item_count--;

    	unsigned int class_exist = 0;
    	for(unsigned int i=0; i < clases_count; i++)
    	{
    		if(clases[i].item == item)
    		{
    			clases[i].sup++;
    			class_exist = 1;
    			break;
    		}
    	}

    	if (!class_exist)
    	{
            clases[clases_count].item = item;
            clases[clases_count++].sup = 1;
    	}
    	// clase procesada, incrementado su soporte o añadida

    	transactions[transactions_count].items = (unsigned int*) realloc (transactions[transactions_count].items, transactions[transactions_count].count * sizeof(unsigned int));
    	transactions_count++;
        if(transactions_count == MAX_TRANSACTIONS)
        {
            printf("No se procesaron todas las transacciones. Modifique la constante MAX_TRANSACTIONS.\n");
        	eof = NULL;
        }
        else eof = fgets(buffer,15000,file);
    }

    free(buffer);

    uno_frecuentes = (UnoFrecuentes*) malloc(sizeof(UnoFrecuentes));
    uno_frecuentes->items = (unsigned int*) malloc((unique_item_count + clases_count) * sizeof(unsigned int));
    uno_frecuentes->items_support = (unsigned int*) malloc((unique_item_count + clases_count) * sizeof(unsigned int));
    support_inv = (unsigned int**) malloc((max_support + 1) * sizeof(unsigned int*));
    support_inv_count = (unsigned int*) calloc(max_support + 1, sizeof(unsigned int));
    support_inv_size = (unsigned int*) calloc(max_support + 1, sizeof(unsigned int));
    for(int i = ceil(min_support * transactions_count); i <= max_support; i++)
    {
    	support_inv[i] = (unsigned int*) malloc(100 * sizeof(unsigned int));
    	support_inv_size[i] = 100;
    }
    uno_frecuentes->items_count = 0;

    // se almacena para cada valor de soporte, mayor o igual al umbral de soporte, los ítems que comparten ese soporte
    unsigned int sup_temp;
    for(int i = 0; i < unique_item_count; i++)
    {
    	sup_temp = unique_supports[unique_items[i]];
    	if(((float) sup_temp / transactions_count) >= min_support)
    	{
    		if(support_inv_count[sup_temp] == support_inv_size[sup_temp])
    		{
    			support_inv[sup_temp] = (unsigned int*) realloc(support_inv[sup_temp], (support_inv_size[sup_temp] + 100) * sizeof(unsigned int));
		        support_inv_size[sup_temp] += 100;

    		}
    		support_inv[sup_temp][support_inv_count[sup_temp]++] = unique_items[i];
    	}
    }

    // se almacenan los N uno frecuentes, ordenados por su soporte y se mapean, de 1 a N
    // se ponen las clases delante y se mapean de primeras, esto hace que se vean como ítems de forma transparente
    uno_frecuentes->map1_N = (unsigned int*) calloc((max_item + clases_count + 1), sizeof(unsigned int));

    for(unsigned int i=0; i < clases_count; i++)
    {
    	uno_frecuentes->items[uno_frecuentes->items_count] = clases[i].item;
    	uno_frecuentes->map1_N[uno_frecuentes->items[uno_frecuentes->items_count]] = uno_frecuentes->items_count + 1;
    	uno_frecuentes->items_support[uno_frecuentes->items_count++] = clases[i].sup;
    }

    for(int i = ceil(min_support * transactions_count); i <= max_support; i++)
    {
    	if(support_inv_count[i] > 0)
    		for(int j = 0; j < support_inv_count[i]; j++)
    		{
    			uno_frecuentes->items[uno_frecuentes->items_count] = support_inv[i][j];
    			uno_frecuentes->map1_N[uno_frecuentes->items[uno_frecuentes->items_count]] = uno_frecuentes->items_count + 1;
    			uno_frecuentes->items_support[uno_frecuentes->items_count++] = i;
    		}
    }

    uno_frecuentes->items = (unsigned int*) realloc(uno_frecuentes->items, uno_frecuentes->items_count * sizeof(unsigned int));
    uno_frecuentes->items_support = (unsigned int*) realloc(uno_frecuentes->items_support, uno_frecuentes->items_count * sizeof(unsigned int));

    // se inicializa la matriz que almacena las transacciones a nivel de bits
    b->btiItemsCountLastRow = transactions_count % 32;
    b->btiCountRow = (b->btiItemsCountLastRow == 0) ? transactions_count/32 : transactions_count/32 + 1;

    b->bti_data = (unsigned int*) calloc(uno_frecuentes->items_count * b->btiCountRow, sizeof(unsigned int));
    b->bti = (unsigned int**) malloc(uno_frecuentes->items_count * sizeof(unsigned int*));
    for(int i = 0; i < uno_frecuentes->items_count; i++)
        b->bti[i] = &b->bti_data[i * b->btiCountRow];

    b->btiCountColumns = uno_frecuentes->items_count;

    btiItem(fileName,b, uno_frecuentes->map1_N);

    // se libera espacio
    free(unique_check);
    free(unique_items);
    free(unique_supports);
    for(int i = ceil(min_support * transactions_count); i <= max_support; i++)
        free(support_inv[i]);
    free(support_inv);
    free(support_inv_count);
    free(support_inv_size);
    fclose(file);
}

/*
 * Variante II
 */
/*
// Función para insertar ítem manteniendo orden
void insert_sorted(unsigned int* itemset, int count, unsigned int new_item) {
    int pos = count - 1;
    unsigned int new_map = uno_frecuentes->map1_N[new_item];

    // Encontrar posición de inserción
    while (pos >= 0 && uno_frecuentes->map1_N[itemset[pos]] > new_map) {
        itemset[pos + 1] = itemset[pos];
        pos--;
    }
    itemset[pos + 1] = new_item;
}

void remove_item(unsigned int* itemset, int count, int pos) {
    for (int j = pos; j < count - 1; j++) {
        itemset[j] = itemset[j + 1];
    }
}

int is_subset(const unsigned int* subset, int subset_count,
              const unsigned int* superset, int superset_count) {
    int i = 0, j = 0;
    while (i < subset_count && j < superset_count) {
        if (subset[i] == superset[j]) {
            i++;
            j++;
        } else if (uno_frecuentes->map1_N[subset[i]] < uno_frecuentes->map1_N[superset[j]]) {
            return 0;
        } else {
            j++;
        }
    }
    return (i == subset_count);
}

int is_maximal(unsigned int* itemset, int itemset_count) {
    //if (itemset_count == 0) return 1;

    // Encontrar ítem con menor frecuencia en maximales
    unsigned int min_item = itemset[0];
    unsigned int min_count = inverted_index[min_item].count;

    for (int i = 1; i < itemset_count; i++) {
        if (inverted_index[itemset[i]].count < min_count) {
            min_count = inverted_index[itemset[i]].count;
            min_item = itemset[i];
        }
    }

    // Buscar solo en maximales que contienen el ítem mínimo
    for (int i = 0; i < inverted_index[min_item].count; i++) {
        Itemset* maximal = inverted_index[min_item].list[i];
        if (maximal->size > itemset_count &&
            is_subset(itemset, itemset_count, maximal->items, maximal->size)) {
            return 0;
        }
    }
    return 1;
}

void find_maximal_frequent(Bti* b, unsigned int* current_itemset, int current_count,
                           BlockList* current_block, int start, float min_support,
                           unsigned int current_support, unsigned int transaction_count,
                           unsigned int* maximal_total, unsigned int* block_pool_index_ptr) {

    unsigned int saved_pool_index = *block_pool_index_ptr;
    BlockList new_block = {NULL, 0};
    unsigned int temp;
    unsigned int new_support = 0;
    unsigned int tail_pruning = 0;

    for (int i = start; !tail_pruning && (i < uno_frecuentes->items_count) &&
         (*maximal_total < MAX_MAXIMAL) && (current_count < MAX_ITEMS); i++) {

        // Insertar ítem manteniendo orden
        insert_sorted(current_itemset, current_count, uno_frecuentes->items[i]);
        unsigned int new_item = uno_frecuentes->items[i];
        new_support = 0;
        // Cálculo de soporte y bloques
        if (current_count + 1 == 1) {
            new_support = uno_frecuentes->items_support[i];
        } else if (current_count + 1 == 2) {
            // Usar pool de memoria
            new_block.blocks = &block_pool[*block_pool_index_ptr];
            *block_pool_index_ptr += b->btiCountRow;
            unsigned int total_blocks = 0;

            for (unsigned int z = 0; z < b->btiCountRow; z++) {
                temp = b->bti[uno_frecuentes->map1_N[current_itemset[0]] - 1][z] &
                       b->bti[uno_frecuentes->map1_N[current_itemset[1]] - 1][z];
                if (temp != 0) {
                    new_block.blocks[total_blocks] = (Block){temp, z};
                    total_blocks++;
                    new_support += _mm_popcnt_u32(temp);
                }
            }
            new_block.count = total_blocks;
        } else {
            // Usar pool de memoria
            new_block.blocks = &block_pool[*block_pool_index_ptr];
            *block_pool_index_ptr += current_block->count;
            unsigned int valid_count = 0;

            for (int z = 0; z < current_block->count; z++) {
                temp = current_block->blocks[z].value &
                       b->bti[uno_frecuentes->map1_N[new_item] - 1][current_block->blocks[z].number];
                if (temp != 0) {
                    new_block.blocks[valid_count] = (Block){temp, current_block->blocks[z].number};
                    valid_count++;
                    new_support += _mm_popcnt_u32(temp);
                }
            }
            new_block.count = valid_count;
        }

        if (new_support >= min_support) {
            if (new_support == current_support) tail_pruning = 1;

            find_maximal_frequent(b, current_itemset, current_count + 1, &new_block, i + 1,
                                 min_support, new_support, transaction_count, maximal_total,
                                 block_pool_index_ptr);
        }

        // Liberar solo el espacio usado en el pool (restaurando índice)
        *block_pool_index_ptr = saved_pool_index;
        remove_item(current_itemset, current_count + 1, current_count);
    }

    if (!tail_pruning && is_maximal(current_itemset, current_count)) {
        if (*maximal_total < MAX_MAXIMAL) {
            // Copiar conjunto al almacenamiento maximal
            Itemset* new_maximal = &maximal_list[current_count]->sets[maximal_list[current_count]->count];
            memcpy(new_maximal->items, current_itemset, current_count * sizeof(unsigned int));
            new_maximal->size = current_count;
            new_maximal->sup = current_support;

            // Actualizar índice invertido
            for (int j = 0; j < current_count; j++) {
                unsigned int item = current_itemset[j];
                if (inverted_index[item].count == inverted_index[item].size) {
                    inverted_index[item].size = inverted_index[item].size ? inverted_index[item].size * 2 : 100;
                    inverted_index[item].list = realloc(inverted_index[item].list,
                                                      inverted_index[item].size * sizeof(Itemset*));
                }
                inverted_index[item].list[inverted_index[item].count++] = new_maximal;
            }

            maximal_list[current_count]->count++;
            (*maximal_total)++;
        }
    }
}
*/
/*
 * Variante II
 */

/*
 * Variante I
 */
int is_subset(unsigned int* subset, int subset_count, unsigned int* superset, int superset_count) {
	unsigned int low = 0;
	for (unsigned int i = 0; i < subset_count; i++)
	{
	    unsigned int high = superset_count - 1;
	    unsigned int found = 0;
	    while (low <= high)
	    {
	       unsigned int mid = (low + high) / 2;
	       if(uno_frecuentes->map1_N[superset[mid]] == uno_frecuentes->map1_N[subset[i]])
	       {
	          found = 1;
	          low = mid + 1;
	          break;
	       }
	       if(uno_frecuentes->map1_N[superset[mid]] < uno_frecuentes->map1_N[subset[i]])
	           low = mid + 1;
	       else
	           high = mid - 1;
	    }

	    if (!found) return 0;
	}
	return 1;
}

int is_maximal(unsigned int* itemset, int itemset_count) {
	unsigned long int h = 1;
	unsigned int seed = 19;
	unsigned int key;
	for (unsigned int i = 0; i < itemset_count; i++)
	    h = ((h * seed) + itemset[i]) % BLOOM_FILTER_SIZE;

	key = h;
	if(bloom[key].flat == 0)
		return 1;

	for(unsigned int j = 0; j < bloom[key].sets_count; j++)
		if (bloom[key].sets[j]->size > itemset_count && is_subset(itemset, itemset_count, bloom[key].sets[j]->items, bloom[key].sets[j]->size))
		    return 0;
	return 1;
}
/*
 * Variante I
 */

/*
 * DFS
 */
void find_maximal_frequent(Bti* b, unsigned int* current_itemset, int current_count, BlockList* current_block, BlockList* current_block_antecedent, int start,
		float min_support, unsigned int current_support, unsigned int ant_sup, unsigned int transaction_count, unsigned int* maximal_total) {

	if(current_count == 1 && uno_frecuentes->map1_N[current_itemset[0]] - 1 >= clases_count)
	    return;

	BlockList new_block = {NULL, 0};
	BlockList block_antecedent = {NULL, 0};
	unsigned int temp;
	unsigned int temp1;
	unsigned int new_support = 0;
	unsigned int support_antecedent = 0;
	unsigned int tail_pruning = 0;

	int has_frequent_superset = 0;

    for (int i = start; !tail_pruning && (i < uno_frecuentes->items_count) && (*maximal_total < MAX_MAXIMAL) && (current_count < MAX_ITEMS); i++)
    {
    	current_itemset[current_count] = uno_frecuentes->items[i];

        new_support = 0;
        support_antecedent = 0;
        if((current_count + 1) == 1)
        {
        	new_support = uno_frecuentes->items_support[i];

        }
        else if((current_count + 1) == 2) //item+clase
               {
        	       new_block.blocks = malloc(b->btiCountRow * sizeof(Block));
        	       unsigned int total_blocks = 0;

       	    	   for(unsigned int z = 0; z < b->btiCountRow; z++)
       	    	   {
        	           temp = b->bti[uno_frecuentes->map1_N[current_itemset[0]] - 1][z] & b->bti[uno_frecuentes->map1_N[current_itemset[1]] - 1][z];
        	    	   if(temp > 0)
        	    	   {
        	    		   new_block.blocks[total_blocks] = (Block){temp, z};
        	    		   total_blocks++;
        	    	   }
        	    	   new_support += _mm_popcnt_u32(temp);
        	       }

       	    	   new_block.count = total_blocks;

                   //para calcular el soporte del antecedente.... modificacion unida a los 2 nuevos parametros que paso en el llamado recursivo
				   block_antecedent.blocks = malloc(b->btiCountRow * sizeof(Block));
				   unsigned int total_blocks_antec = 0;

				   for(unsigned int z = 0; z < b->btiCountRow; z++)
				   {
					   temp1 = b->bti[uno_frecuentes->map1_N[current_itemset[1]] - 1][z]; //dejando fuera la clase
					   if(temp1 > 0)
					   {
						   block_antecedent.blocks[total_blocks_antec] = (Block){temp1, z};
						   total_blocks_antec++;
					   }
					   support_antecedent += _mm_popcnt_u32(temp1);
				   }

				   block_antecedent.count = total_blocks_antec;
               }
               else
               {
            	   new_block.blocks = malloc(current_block->count * sizeof(Block));
            	   block_antecedent.blocks = malloc(current_block_antecedent->count * sizeof(Block));
            	   unsigned int valid_count = 0;
            	   unsigned int valid_antec_count = 0;

        	       for(int z = 0; z < current_block->count; z++ )
                   {
        	    	   temp = current_block->blocks[z].value & b->bti[uno_frecuentes->map1_N[current_itemset[current_count]] - 1][current_block->blocks[z].number];
        	           if(temp > 0)
        	           {
        	               new_support += _mm_popcnt_u32(temp);
        	               new_block.blocks[valid_count] = (Block){temp, current_block->blocks[z].number};
        	               valid_count++;
        	          }
                   }
        	       new_block.count = valid_count;


                   //para calcular el soporte del antecedente... modificado
				   for(int z = 0; z < current_block_antecedent->count; z++ )
					  {
					   temp1 = current_block_antecedent->blocks[z].value & b->bti[uno_frecuentes->map1_N[current_itemset[current_count]] - 1][current_block_antecedent->blocks[z].number];
					   if(temp1 > 0)
					   {
						   support_antecedent += _mm_popcnt_u32(temp1);
						   block_antecedent.blocks[valid_antec_count] = (Block){temp1, current_block_antecedent->blocks[z].number};
						   valid_antec_count++;
					  }
					  }
				   block_antecedent.count = valid_antec_count;
}

        if (new_support >= min_support)
        {
        	if(new_support == current_support && support_antecedent == ant_sup)
         		tail_pruning  = 1;


            has_frequent_superset = 1;

            //printf(" | new_support=%u, supAntec=%u\n", new_support, support_antecedent);

            find_maximal_frequent(b, current_itemset, current_count + 1, &new_block,&block_antecedent, i + 1, min_support, new_support, support_antecedent, transaction_count, maximal_total);
            if(new_block.blocks != 0)
            	free(new_block.blocks);
            if(block_antecedent.blocks != 0)
                        	free(block_antecedent.blocks);
        }
        else if(new_block.blocks != 0 && block_antecedent.blocks != 0)
        {
        	free(new_block.blocks);
        	free(block_antecedent.blocks);
        }
    }

    if (!has_frequent_superset && is_maximal(current_itemset, current_count)) {
        if(*maximal_total < MAX_MAXIMAL)
        {
        	memcpy(maximal_list[current_count]->sets[maximal_list[current_count]->count].items, current_itemset, current_count * sizeof(unsigned int));
        	maximal_list[current_count]->sets[maximal_list[current_count]->count].size = current_count;
        	maximal_list[current_count]->sets[maximal_list[current_count]->count].sup = current_support;
        	maximal_list[current_count]->sets[maximal_list[current_count]->count].supAntec = ant_sup;  //modificado
        	(*maximal_total)++;


        	/*
        	 * Variante I
        	 */
        	unsigned int combination[MAX_ITEMS - 1];
        	tempHash[0] = 1;
        	unsigned int tempHash_count = 1;
        	unsigned int it = 0;
        	unsigned long int h;
        	unsigned int key;
        	unsigned int seed = 19;
        	for(unsigned int r = 1; r < current_count; r++)
        	{
        		h = tempHash[it];
                key = (h * seed + current_itemset[r - 1]) % BLOOM_FILTER_SIZE;

                tempHash[tempHash_count++] = key;
                bloom[key].flat = 1;
                if(bloom[key].sets_count % 100 == 0)
                    bloom[key].sets = (Itemset**) realloc(bloom[key].sets, (bloom[key].sets_count + 100) * sizeof(Itemset*));
                bloom[key].sets[bloom[key].sets_count++] = &(maximal_list[current_count]->sets[maximal_list[current_count]->count]);

                for (int i = 0; i < r; i++)
        		    combination[i] = i;

        	    while (1)
        	    {
        	        int i = r - 1;
        	        while (i >= 0 && combination[i] == current_count - r + i)
        	        {
        	            i--;
        	            it++;
        	        }


        	        if (i < 0)
        	            break;

        	        combination[i]++;
        	        for (int j = i + 1; j < r; j++)
        	            combination[j] = combination[i] + j - i;

        	        h = tempHash[it];
        	        key = (h * seed + current_itemset[combination[r - 1]]) % BLOOM_FILTER_SIZE;

        	        tempHash[tempHash_count++] = key;
        	        bloom[key].flat = 1;
        	        if(bloom[key].sets_count % 100 == 0)
        	            bloom[key].sets = (Itemset**) realloc(bloom[key].sets, (bloom[key].sets_count + 100) * sizeof(Itemset*));
        	        bloom[key].sets[bloom[key].sets_count++] = &(maximal_list[current_count]->sets[maximal_list[current_count]->count]);
        	    }
        	}
        	/*
        	 * Variante I
        	 */

        	maximal_list[current_count]->count++;
        }
    }
}

/*
 * Variante I
 */
void find_maximal_frequent_patterns(float min_support, Bti* b, unsigned int transaction_count) {
	unsigned int current_itemset[MAX_ITEMS];
    bloom = (Bloom*) calloc(BLOOM_FILTER_SIZE, sizeof(Bloom));

    for(int i = 1; i <= MAX_ITEMS; i++)
    {
    	maximal_list[i] = (MaximalList*) malloc(sizeof(MaximalList));
    	maximal_list[i]->count = 0;
    }

    maximalTotal = 0;

    find_maximal_frequent(b, &(current_itemset[0]), 0, 0, 0, 0, min_support, 0,0,transaction_count, &maximalTotal);

    printf("Total de Maximales %d\n", maximalTotal);
}
/*
 * Variante I
 */



/* ==========================================================
 * Soporte exacto de un itemset usando la matriz compactada BTI
 * (misma idea que en find_maximal_frequent pero sin recursión)
 * Devuelve el soporte absoluto (número de transacciones).
 * ========================================================== */
static unsigned int support_itemset_bti(Bti* b, const unsigned int* items, int count,
                                        Block* bufA, Block* bufB, unsigned int bufCap)
{
    if (count <= 0) return 0;

    // La BTI solo contiene ítems frecuentes: map1_N[item] > 0
    unsigned int row0 = uno_frecuentes->map1_N[items[0]];
    if (row0 == 0) return 0;
    row0 -= 1;

    unsigned int sup = 0;
    unsigned int used = 0;

    // Inicializar con la fila del primer ítem (solo bloques no-cero)
    for (unsigned int z = 0; z < b->btiCountRow; z++) {
        unsigned int v = b->bti[row0][z];
        if (v != 0) {
            if (used >= bufCap) break; // seguridad
            bufA[used].value  = v;
            bufA[used].number = z;
            used++;
            sup += _mm_popcnt_u32(v);
        }
    }

    // Intersectar con el resto de ítems
    for (int i = 1; i < count && used > 0; i++) {
        unsigned int row = uno_frecuentes->map1_N[items[i]];
        if (row == 0) return 0;
        row -= 1;

        unsigned int newUsed = 0;
        unsigned int newSup  = 0;

        for (unsigned int k = 0; k < used; k++) {
            unsigned int col = bufA[k].number;
            unsigned int v   = bufA[k].value & b->bti[row][col];
            if (v != 0) {
                if (newUsed >= bufCap) break; // seguridad
                bufB[newUsed].value  = v;
                bufB[newUsed].number = col;
                newUsed++;
                newSup += _mm_popcnt_u32(v);
            }
        }

        // swap buffers
        Block* tmp = bufA; bufA = bufB; bufB = tmp;
        used = newUsed;
        sup  = newSup;
    }

    return sup;
}

static double netconf_from_counts(unsigned int supXY, unsigned int supX,
                                  unsigned int supY, unsigned int N)
{
    if (supX == 0 || N == 0) return -1.0;

    double SopXY = (double)supXY / (double)N;
    double SopX  = (double)supX  / (double)N;
    double SopY  = (double)supY  / (double)N;

    // Netconf(X=>Y) = (Sop(XY) - Sop(X)Sop(Y)) / (Sop(X)(1-Sop(X)))
    // Evitar división por cero cuando SopX es 0 o 1.
    double denom = SopX * (1.0 - SopX);
    if (denom <= 0.0) return -1.0;

    return (SopXY - (SopX * SopY)) / denom;
}

/* Comparador para Top-K: primero netconf, luego confianza, luego soporte */
static int better_rule(const CAR* a, const CAR* b)
{
    /* Ranking estable:
       1) score (netconf suavizado por soporte)
       2) netconf (calidad pura)
       3) confianza
       4) soporte
    */
    if (a->score > b->score) return 1;
    if (a->score < b->score) return 0;

    if (a->netconf > b->netconf) return 1;
    if (a->netconf < b->netconf) return 0;

    if (a->confianza > b->confianza) return 1;
    if (a->confianza < b->confianza) return 0;

    if (a->soporte > b->soporte) return 1;
    if (a->soporte < b->soporte) return 0;

    return 0;
}


/* Heap MIN de tamaño K (la raíz es el peor de los Top-K) */
static void heap_sift_up(CAR* h, int idx)
{
    while (idx > 0) {
        int p = (idx - 1) / 2;
        // si h[p] es "mejor" que h[idx], entonces h[idx] es peor => ok para min-heap?
        // Queremos min-heap por "mejor_rule": raíz = peor.
        // Si h[idx] es peor que h[p], intercambiar.
        if (better_rule(&h[p], &h[idx])) {
            CAR tmp = h[p]; h[p] = h[idx]; h[idx] = tmp;
            idx = p;
        } else break;
    }
}

static void heap_sift_down(CAR* h, int size, int idx)
{
    for (;;) {
        int l = 2*idx + 1;
        int r = l + 1;
        int smallest = idx;

        // Seleccionar el peor (mínimo) según better_rule: si h[smallest] es mejor que h[child], child es peor.
        if (l < size && better_rule(&h[smallest], &h[l])) smallest = l;
        if (r < size && better_rule(&h[smallest], &h[r])) smallest = r;

        if (smallest != idx) {
            CAR tmp = h[idx]; h[idx] = h[smallest]; h[smallest] = tmp;
            idx = smallest;
        } else break;
    }
}

static void topk_push(CAR* heap, int* heapSize, int K, const CAR* cand)
{
    if (K <= 0) return;

    if (*heapSize < K) {
        heap[*heapSize] = *cand;
        (*heapSize)++;
        heap_sift_up(heap, (*heapSize) - 1);
        return;
    }

    // Si el candidato es mejor que la raíz (peor actual), reemplazar.
    if (better_rule(cand, &heap[0])) {
        heap[0] = *cand;
        heap_sift_down(heap, *heapSize, 0);
    }
}


/* =========================================================================
 *  OTF (ON-THE-FLY): Top-K subreglas por maximal DURANTE el DFS de maximales
 * =========================================================================
 *
 * Objetivo práctico:
 *  - Con maximales grandes, enumerar todos los subconjuntos (2^n) para buscar
 *    subreglas con alto Netconf es costoso.
 *  - En clasificación asociativa, muchas subreglas útiles son CORTAS (1-3 ítems).
 *    Esas subreglas aparecen como nodos frecuentes en el mismo árbol DFS.
 *
 * Qué hace esta versión:
 *  - Mientras el DFS extiende un patrón frecuente que inicia con una CLASE,
 *    cada nodo frecuente con tamaño>=2 produce una regla candidata:
 *        {items[1..]} -> items[0]
 *  - Cuando el DFS determina que el patrón actual es maximal, en vez de enumerar
 *    subconjuntos, se toma el Top-K (según better_rule/topk_push) sobre la pila
 *    de reglas acumuladas en el camino DFS (subreglas tipo “prefijo”).
 *
 * Nota:
 *  - Esto NO recupera todos los subconjuntos de un maximal; recupera eficientemente
 *    las subreglas visitadas en el camino DFS, que en la práctica son las que más
 *    suelen aportar (cortas y con alto Netconf).
 */

static unsigned int class_support_from_item(unsigned int class_item)
{
    for (unsigned int k = 0; k < clases_count; k++)
        if (clases[k].item == class_item) return clases[k].sup;
    return 0;
}

static void fill_metrics_from_counts(CAR* c,
                                     unsigned int supXY,
                                     unsigned int supX,
                                     unsigned int supY,
                                     unsigned int N)
{
    c->maximal.sup = supXY;
    c->maximal.supAntec = supX;
    c->consecuente.sup = supY;

    c->soporte       = (double)supXY / (double)N;
    c->soporteAntec  = (double)supX  / (double)N;
    c->soporteConsec = (double)supY  / (double)N;

    c->confianza = (supX > 0) ? ((double)supXY / (double)supX) : 0.0;
    c->netconf   = netconf_from_counts(supXY, supX, supY, N);

    if (c->soporteAntec > 0.0 && c->soporteConsec > 0.0)
        c->lift = c->soporte / (c->soporteAntec * c->soporteConsec);
    else
        c->lift = 0.0;

    if (1.0 - c->confianza != 0.0)
        c->conviction = (1.0 - c->soporteConsec) / (1.0 - c->confianza);
    else
        c->conviction = INFINITY;
}

static int make_cand_from_current(CAR* out,
                                  unsigned int* current_itemset,
                                  int current_count,
                                  unsigned int supXY,
                                  unsigned int supX,
                                  unsigned int supY,
                                  unsigned int N,
                                  unsigned int min_support_abs,
                                  double min_netconf_pos)
{
    if (current_count < 2) return 0;

    out->maximal.size = (char)current_count;
    for (int i = 0; i < current_count; i++) out->maximal.items[i] = current_itemset[i];
    out->consecuente.item = current_itemset[0];

    fill_metrics_from_counts(out, supXY, supX, supY, N);
    out->score = stable_score_from_counts(out->netconf, supXY, min_support_abs);
    out->negated = 0;
    /* Strict positive-evidence admission: equality is independence, not
       evidence in favour of the class. */
    if (out->netconf <= min_netconf_pos) return 0;
    return 1;
}

static void find_maximal_frequent_otf(Bti* b,
                                      unsigned int* current_itemset,
                                      int current_count,
                                      BlockList* current_block,
                                      BlockList* current_block_antecedent,
                                      int start,
                                      unsigned int min_support_abs,
                                      unsigned int current_support,
                                      unsigned int ant_sup,
                                      unsigned int transaction_count,
                                      unsigned int* maximal_total,
                                      int TOPK_PER_MAXIMAL,
                                      double min_netconf_pos,
                                      CAR* rule_stack,
                                      int rule_stack_size,
                                      unsigned int class_sup)
{
    if (current_count == 1 && uno_frecuentes->map1_N[current_itemset[0]] - 1 >= clases_count)
        return;

    BlockList new_block = (BlockList){NULL, 0};
    BlockList block_antecedent = (BlockList){NULL, 0};
    unsigned int temp, temp1;
    unsigned int new_support = 0;
    unsigned int support_antecedent = 0;
    unsigned int tail_pruning = 0;
    int has_frequent_superset = 0;

    for (int i = start;
         !tail_pruning && (i < (int)uno_frecuentes->items_count) && (*maximal_total < MAX_MAXIMAL) && (current_count < MAX_ITEMS);
         i++)
    {
        current_itemset[current_count] = uno_frecuentes->items[i];
        new_support = 0;
        support_antecedent = 0;

        if ((current_count + 1) == 1) {
            new_support = uno_frecuentes->items_support[i];
            class_sup = class_support_from_item(current_itemset[0]);
        }
        else if ((current_count + 1) == 2)
        {
            new_block.blocks = malloc(b->btiCountRow * sizeof(Block));
            unsigned int total_blocks = 0;

            for (unsigned int z = 0; z < b->btiCountRow; z++)
            {
                temp = b->bti[uno_frecuentes->map1_N[current_itemset[0]] - 1][z] &
                       b->bti[uno_frecuentes->map1_N[current_itemset[1]] - 1][z];
                if (temp > 0) {
                    new_block.blocks[total_blocks] = (Block){temp, z};
                    total_blocks++;
                }
                new_support += _mm_popcnt_u32(temp);
            }
            new_block.count = total_blocks;

            block_antecedent.blocks = malloc(b->btiCountRow * sizeof(Block));
            unsigned int total_blocks_antec = 0;

            for (unsigned int z = 0; z < b->btiCountRow; z++)
            {
                temp1 = b->bti[uno_frecuentes->map1_N[current_itemset[1]] - 1][z];
                if (temp1 > 0) {
                    block_antecedent.blocks[total_blocks_antec] = (Block){temp1, z};
                    total_blocks_antec++;
                }
                support_antecedent += _mm_popcnt_u32(temp1);
            }
            block_antecedent.count = total_blocks_antec;
        }
        else
        {
            new_block.blocks = malloc(current_block->count * sizeof(Block));
            block_antecedent.blocks = malloc(current_block_antecedent->count * sizeof(Block));
            unsigned int valid_count = 0;
            unsigned int valid_antec_count = 0;

            for (int z = 0; z < (int)current_block->count; z++)
            {
                temp = current_block->blocks[z].value &
                       b->bti[uno_frecuentes->map1_N[current_itemset[current_count]] - 1][current_block->blocks[z].number];
                if (temp > 0) {
                    new_support += _mm_popcnt_u32(temp);
                    new_block.blocks[valid_count] = (Block){temp, current_block->blocks[z].number};
                    valid_count++;
                }
            }
            new_block.count = valid_count;

            for (int z = 0; z < (int)current_block_antecedent->count; z++)
            {
                temp1 = current_block_antecedent->blocks[z].value &
                        b->bti[uno_frecuentes->map1_N[current_itemset[current_count]] - 1][current_block_antecedent->blocks[z].number];
                if (temp1 > 0) {
                    support_antecedent += _mm_popcnt_u32(temp1);
                    block_antecedent.blocks[valid_antec_count] = (Block){temp1, current_block_antecedent->blocks[z].number};
                    valid_antec_count++;
                }
            }
            block_antecedent.count = valid_antec_count;
        }

        if (new_support >= min_support_abs)
        {
            if (new_support == current_support && support_antecedent == ant_sup)
                tail_pruning = 1;

            has_frequent_superset = 1;

            int pushed = 0;
            CAR cand;
            if ((current_count + 1) >= 2 && class_sup > 0 && support_antecedent >= min_support_abs) {
                if (make_cand_from_current(&cand,
                                          current_itemset,
                                          current_count + 1,
                                          new_support,
                                          support_antecedent,
                                          class_sup,
                                          transaction_count,
                                          min_support_abs,
                                          min_netconf_pos))
                {
                    rule_stack[rule_stack_size++] = cand;
                    pushed = 1;
                }
            }

            find_maximal_frequent_otf(b,
                                     current_itemset,
                                     current_count + 1,
                                     &new_block,
                                     &block_antecedent,
                                     i + 1,
                                     min_support_abs,
                                     new_support,
                                     support_antecedent,
                                     transaction_count,
                                     maximal_total,
                                     TOPK_PER_MAXIMAL,
                                     min_netconf_pos,
                                     rule_stack,
                                     rule_stack_size,
                                     class_sup);

            if (pushed) rule_stack_size--;

            if (new_block.blocks) free(new_block.blocks);
            if (block_antecedent.blocks) free(block_antecedent.blocks);
        }
        else
        {
            if (new_block.blocks) free(new_block.blocks);
            if (block_antecedent.blocks) free(block_antecedent.blocks);
        }
    }

    if (!has_frequent_superset && is_maximal(current_itemset, current_count)) {
        if (*maximal_total < MAX_MAXIMAL) {

            memcpy(maximal_list[current_count]->sets[maximal_list[current_count]->count].items,
                   current_itemset, current_count * sizeof(unsigned int));
            maximal_list[current_count]->sets[maximal_list[current_count]->count].size = current_count;
            maximal_list[current_count]->sets[maximal_list[current_count]->count].sup = current_support;
            maximal_list[current_count]->sets[maximal_list[current_count]->count].supAntec = ant_sup;
            (*maximal_total)++;

            /* Variante Bloom original */
            unsigned int combination[MAX_ITEMS - 1];
            tempHash[0] = 1;
            unsigned int tempHash_count = 1;
            unsigned int it = 0;
            unsigned long int h;
            unsigned int key;
            unsigned int seed = 19;

            for (unsigned int r = 1; r < (unsigned int)current_count; r++)
            {
                h = tempHash[it];
                key = (h * seed + current_itemset[r - 1]) % BLOOM_FILTER_SIZE;

                tempHash[tempHash_count++] = key;
                bloom[key].flat = 1;
                if (bloom[key].sets_count % 100 == 0)
                    bloom[key].sets = (Itemset**) realloc(bloom[key].sets, (bloom[key].sets_count + 100) * sizeof(Itemset*));
                bloom[key].sets[bloom[key].sets_count++] = &(maximal_list[current_count]->sets[maximal_list[current_count]->count]);

                for (int i = 0; i < (int)r; i++)
                    combination[i] = i;

                while (1)
                {
                    int i = (int)r - 1;
                    while (i >= 0 && combination[i] == (unsigned int)current_count - (int)r + i)
                    {
                        i--;
                        it++;
                    }
                    if (i < 0) break;

                    combination[i]++;
                    for (int j = i + 1; j < (int)r; j++)
                        combination[j] = combination[i] + j - i;

                    h = tempHash[it];
                    key = (h * seed + current_itemset[combination[r - 1]]) % BLOOM_FILTER_SIZE;

                    tempHash[tempHash_count++] = key;
                    bloom[key].flat = 1;
                    if (bloom[key].sets_count % 100 == 0)
                        bloom[key].sets = (Itemset**) realloc(bloom[key].sets, (bloom[key].sets_count + 100) * sizeof(Itemset*));
                    bloom[key].sets[bloom[key].sets_count++] = &(maximal_list[current_count]->sets[maximal_list[current_count]->count]);
                }
            }

            maximal_list[current_count]->count++;

            /* Volcar Top-K reglas del camino DFS */
            if (TOPK_PER_MAXIMAL < 1) TOPK_PER_MAXIMAL = 1;

            CAR heap[TOPK_PER_MAXIMAL];
            int heapSize = 0;
            for (int r = 0; r < rule_stack_size; r++)
                topk_push(heap, &heapSize, TOPK_PER_MAXIMAL, &rule_stack[r]);

            for (int hidx = 0; hidx < heapSize; hidx++) {
                if (total_CARs < (int)(MAX_MAXIMAL * TOPK_SUBRULES))
                    CARS[total_CARs++] = heap[hidx];
            }
        }
    }
}

void find_maximal_frequent_patterns_otf(unsigned int min_support_abs,
                                       Bti* b,
                                       unsigned int transaction_count,
                                       int TOPK_PER_MAXIMAL,
                                       double min_netconf_pos)
{
    unsigned int current_itemset[MAX_ITEMS];
    bloom = (Bloom*) calloc(BLOOM_FILTER_SIZE, sizeof(Bloom));

    for (int i = 1; i <= MAX_ITEMS; i++) {
        maximal_list[i] = (MaximalList*) malloc(sizeof(MaximalList));
        maximal_list[i]->count = 0;
    }

    maximalTotal = 0;

    if (TOPK_PER_MAXIMAL < 1) TOPK_PER_MAXIMAL = 1;
    CARS = (CAR*) malloc((unsigned int)MAX_MAXIMAL * (unsigned int)TOPK_PER_MAXIMAL * sizeof(CAR));
    total_CARs = 0;

    CAR rule_stack[MAX_ITEMS];
    int rule_stack_size = 0;

    find_maximal_frequent_otf(b,
                             &(current_itemset[0]),
                             0,
                             0,
                             0,
                             0,
                             min_support_abs,
                             0,
                             0,
                             transaction_count,
                             &maximalTotal,
                             TOPK_PER_MAXIMAL,
                             min_netconf_pos,
                             rule_stack,
                             rule_stack_size,
                             0);

    printf("Total de Maximales %d\n", maximalTotal);
    printf("Total de CARs (OTF, antes de dedup) %d\n", total_CARs);
}

/*
 * M4: reglas positivas directas + reglas negativas formales desde el mismo
 * antecedente maximal.
 *
 * Para cada maximal etiquetado M = {c+} U X:
 *   1) se genera X -> c+ si Netconf(X -> c+) > min_netconf_pos;
 *   2) para cada clase alternativa c != c+, se genera X -> neg(c) si
 *      Netconf(X -> c) < 0.
 *
 * La negativa NO se interpreta como voto hacia otra clase, sino como evidencia
 * contra c. El clasificador M4 la usa solo dentro de la región de empate.
 */
void buildCARs_M4_direct_maximal_pos_neg(Bti* b,
                                         unsigned int min_support_abs,
                                         double min_netconf_pos)
{
    unsigned int capacity = (unsigned int)maximalTotal * (unsigned int)(clases_count > 0 ? clases_count : 1);
    if (capacity == 0) capacity = 1;

    CARS = (CAR*) malloc(capacity * sizeof(CAR));
    if (!CARS) {
        fprintf(stderr, "No hay memoria para CARS en M4.\n");
        exit(EXIT_FAILURE);
    }
    total_CARs = 0;

    unsigned int total_pos = 0;
    unsigned int total_neg = 0;

    unsigned int bufCap = b->btiCountRow;
    Block* bufA = (Block*) malloc(bufCap * sizeof(Block));
    Block* bufB = (Block*) malloc(bufCap * sizeof(Block));
    if (!bufA || !bufB) {
        fprintf(stderr, "No hay memoria para buffers de soporte en M4.\n");
        exit(EXIT_FAILURE);
    }

    for (int size = 2; size <= MAX_ITEMS; size++) {
        for (unsigned int j = 0; j < maximal_list[size]->count; j++) {
            Itemset M = maximal_list[size]->sets[j];
            if (M.size < 2) continue; /* necesita clase + antecedente */

            unsigned int class_pos = M.items[0];
            unsigned int supY_pos = class_support_from_item(class_pos);
            if (supY_pos == 0) continue; /* el maximal no esta sembrado por una clase */

            unsigned int supX = M.supAntec;
            unsigned int supXY_pos = M.sup;
            if (supX == 0 || supXY_pos < min_support_abs) continue;

            /* Construir antecedente X = M \ {class_pos}. */
            unsigned int X[MAX_ITEMS];
            int xsz = 0;
            for (int k = 1; k < M.size; k++) X[xsz++] = M.items[k];
            if (xsz <= 0) continue;

            /* 1) Regla positiva directa: X -> class_pos. */
            {
                CAR cand;
                memset(&cand, 0, sizeof(CAR));
                cand.maximal.size = (char)M.size;
                cand.maximal.items[0] = class_pos;
                for (int k = 0; k < xsz; k++) cand.maximal.items[k+1] = X[k];
                cand.consecuente.item = class_pos;
                fill_metrics_from_counts(&cand, supXY_pos, supX, supY_pos, transactions_count);
                cand.score = stable_score_from_counts(cand.netconf, supXY_pos, min_support_abs);
                cand.negated = 0;

                if (cand.netconf > min_netconf_pos) {
                    if (total_CARs >= capacity) {
                        capacity *= 2;
                        CARS = (CAR*)realloc(CARS, capacity * sizeof(CAR));
                        if (!CARS) { fprintf(stderr, "No hay memoria realloc CARS M4.\n"); exit(EXIT_FAILURE); }
                    }
                    CARS[total_CARs++] = cand;
                    total_pos++;
                }
            }

#if M4_GENERATE_NEGATIVES
            /* 2) Reglas negativas: mismo antecedente X contra clases alternativas. */
            for (unsigned int ck = 0; ck < clases_count; ck++) {
                unsigned int class_alt = clases[ck].item;
                if (class_alt == class_pos) continue;
                unsigned int supY_alt = clases[ck].sup;
                if (supY_alt == 0) continue;

                unsigned int XY[MAX_ITEMS];
                XY[0] = class_alt;
                for (int k = 0; k < xsz; k++) XY[k+1] = X[k];

                unsigned int supXY_alt = support_itemset_bti(b, XY, xsz + 1, bufA, bufB, bufCap);

                CAR neg;
                memset(&neg, 0, sizeof(CAR));
                neg.maximal.size = (char)(xsz + 1);
                neg.maximal.items[0] = class_alt; /* se imprimira como -class_alt */
                for (int k = 0; k < xsz; k++) neg.maximal.items[k+1] = X[k];
                neg.consecuente.item = class_alt;
                fill_metrics_from_counts(&neg, supXY_alt, supX, supY_alt, transactions_count);
                neg.score = stable_score_from_counts(neg.netconf, supXY_alt, min_support_abs);
                neg.negated = 1;

                /* Criterio formal: dependencia negativa respecto a la clase. */
                if (neg.netconf < 0.0) {
                    if (total_CARs >= capacity) {
                        capacity *= 2;
                        CARS = (CAR*)realloc(CARS, capacity * sizeof(CAR));
                        if (!CARS) { fprintf(stderr, "No hay memoria realloc CARS M4.\n"); exit(EXIT_FAILURE); }
                    }
                    CARS[total_CARs++] = neg;
                    total_neg++;
                }
            }
#endif
        }
    }

    free(bufA);
    free(bufB);

    printf("Total de CARs M4 positivas directas (antes de dedup) %u\n", total_pos);
    printf("Total de CARs M4 negativas formales (antes de dedup) %u\n", total_neg);
    printf("Total de CARs M4 pos+neg (antes de dedup) %u\n", total_CARs);
}


static int compare_CARS_desc(const void* pa, const void* pb)
{
    const CAR* a = (const CAR*)pa;
    const CAR* b = (const CAR*)pb;

    /* Preserve the original primary ordering: larger antecedents first. */
    if (a->maximal.size > b->maximal.size) return -1;
    if (a->maximal.size < b->maximal.size) return 1;

    /* Then larger Netconf first, as in the original insertion sort. */
    if (a->netconf > b->netconf) return -1;
    if (a->netconf < b->netconf) return 1;

    /* Deterministic tie-breakers do not alter scoring semantics. */
    if (a->negated < b->negated) return -1; /* positive before negative */
    if (a->negated > b->negated) return 1;

    if (a->maximal.items[0] < b->maximal.items[0]) return -1;
    if (a->maximal.items[0] > b->maximal.items[0]) return 1;

    for (int i = 1; i < a->maximal.size; ++i) {
        if (a->maximal.items[i] < b->maximal.items[i]) return -1;
        if (a->maximal.items[i] > b->maximal.items[i]) return 1;
    }

    return 0;
}

void sortCARS(const char* fileout)
{
    if (total_CARs > 1) {
        qsort(CARS, total_CARs, sizeof(CAR), compare_CARS_desc);
    }

    FILE* ficheroSalida = fopen(fileout, "w");
    if (!ficheroSalida) {
        perror("Error opening rules output file");
        exit(EXIT_FAILURE);
    }

    for (unsigned int i = 0; i < total_CARs; ++i) {
        fprintf(ficheroSalida, "%d ", CARS[i].maximal.size);

        for (int k = 1; k < CARS[i].maximal.size; ++k) {
            fprintf(ficheroSalida, "%u ", CARS[i].maximal.items[k]);
        }

        int consec_out = CARS[i].negated
            ? -(int)CARS[i].maximal.items[0]
            :  (int)CARS[i].maximal.items[0];

        fprintf(ficheroSalida, "%d ", consec_out);
        fprintf(ficheroSalida, "%.5f %.5f %.5f %.5f\n",
                CARS[i].soporteAntec,
                CARS[i].soporteConsec,
                CARS[i].soporte,
                CARS[i].netconf);
    }

    if (fclose(ficheroSalida) != 0) {
        perror("Error closing rules output file");
        exit(EXIT_FAILURE);
    }
}


/*
 * Imprimir en consola
 */
void printMaximales()
{
	unsigned int id = 1;
    for(int i = 1; i <= MAX_ITEMS; i++)
    {
   	   for(int j = 0; j < maximal_list[i]->count; j++)
       {
           if(maximal_list[i]->sets[j].size > 0)
    	   {
    	    	printf("%d -> ", id++);
    	    	for(int z = 0; z < maximal_list[i]->sets[j].size; z++)
    	    		printf("%d ", maximal_list[i]->sets[j].items[z]);
    	    	printf(": %d \n", maximal_list[i]->sets[j].sup);

    	   }
       }
    }
}



void printCARs()   //modificado
{
	unsigned int id = 1;
	FILE * ficheroSalida = 0;
	ficheroSalida = fopen("CARMaxOut.txt","a");
	fprintf(ficheroSalida, "The number of transactions is %d \n", transactions_count);
    for(int i = 1; i <= MAX_ITEMS; i++)
    {
   	   for(int j = 0; j < maximal_list[i]->count; j++)
       {
           if(maximal_list[i]->sets[j].size > 0)
    	   {
    	    	fprintf(ficheroSalida, "[%d] ", id++);
    	    	for(int z = 1; z < maximal_list[i]->sets[j].size; z++)
    	    		fprintf(ficheroSalida, "%d ", maximal_list[i]->sets[j].items[z]);
    	    	fprintf(ficheroSalida, "-> %d", maximal_list[i]->sets[j].items[0]);
    	    	fprintf(ficheroSalida, " with support %.2f", (double)maximal_list[i]->sets[j].sup/transactions_count);
    	    	fprintf(ficheroSalida, " and confidence %.2f \n", (double)maximal_list[i]->sets[j].sup/maximal_list[i]->sets[j].supAntec);

    	   }
       }
    }
    fprintf(ficheroSalida,"\n\n");
    fclose(ficheroSalida);
}

void printCARsNew()   //modificado
{
	unsigned int id = 1;
	FILE * ficheroSalida = 0;
	ficheroSalida = fopen("CARMaxOut.txt","a");
	fprintf(ficheroSalida, "The number of transactions is %d \n", transactions_count);

	for(int i = 1; i <= MAX_ITEMS; i++)
	    {
	   	   for(int j = 0; j < maximal_list[i]->count; j++)
	       {
	           if(maximal_list[i]->sets[j].size > 0)

                {
						fprintf(ficheroSalida, "[%d] ", id++);
						for(int k = 1; k < CARS[j].maximal.size; k++)
							fprintf(ficheroSalida, "%d ", CARS[j].maximal.items[k]);
						int consec_out = (CARS[j].negated ? -(int)CARS[j].maximal.items[0] : (int)CARS[j].maximal.items[0]);
						fprintf(ficheroSalida, "-> %d", consec_out);
						fprintf(ficheroSalida, " with support %.2f", CARS[j].soporte);
						fprintf(ficheroSalida, ", confidence %.2f", CARS[j].confianza);
						fprintf(ficheroSalida, ", lift %.2f", CARS[j].lift);
						fprintf(ficheroSalida, " and netconf %.2f \n", CARS[j].netconf);
                }
	       }
	    }

    fclose(ficheroSalida);
    //free(CARS);
}

void partition70_30(const char* filename){
	FILE *infile, *out70, *out30;
	    char line[1024];
	    int total_lines = 0;
	    int limit, i;

	    infile = fopen(filename, "r");
	    if (infile == NULL) {
	        perror("Error opening input file");
	        exit(EXIT_FAILURE);
	    }


	    while (fgets(line, sizeof(line), infile) != NULL)
	        total_lines++;


	    limit = (int)(total_lines * 0.7);


	    rewind(infile);

	    // Open output files
	    out70 = fopen("output_70.dat", "w");
	    if (out70 == NULL) {
	        perror("Error opening output_70.dat");
	        fclose(infile);
	        exit(EXIT_FAILURE);
	    }

	    out30 = fopen("output_30.dat", "w");
	    if (out30 == NULL) {
	        perror("Error opening output_30.dat");
	        fclose(infile);
	        fclose(out70);
	        exit(EXIT_FAILURE);
	    }

	    // Second pass: split lines
	    for (i = 0; i < total_lines && fgets(line, sizeof(line), infile) != NULL; i++) {
	        if (i < limit)
	            fprintf(out70, "%s", line);
	        else
	            fprintf(out30, "%s", line);
	    }

	    printf("Total lines: %d\n", total_lines);
	    printf("Written -> 70%%: %d lines, 30%%: %d lines\n", limit, total_lines - limit);

	    fclose(infile);
	    fclose(out70);
	    fclose(out30);
}

void clasificacionExacta(){

	char* buffer = (char*) malloc(15000 * sizeof(char));
	FILE* infile = fopen("output_30.dat", "r");
	char* eof = fgets(buffer,15000,infile);
	char* ptr;
	unsigned int item, max_item;
		    if (infile == NULL) {
		        perror("Error opening input file");
		        exit(EXIT_FAILURE);
		    }

		    instances_count = 0;

		       while (eof != NULL) {
		       	instances[instances_count].items = (unsigned int*) malloc (MAX_UNIQUE_ITEMS * sizeof(unsigned int));
		       	instances[instances_count].count = 0;
		       	ptr = buffer;
		       	while((*ptr) & 32)
		           {
		       	    item = 0;
		       	    while((*ptr) & 16)
		       	        item = item * 10 + (*ptr++) - 48;

		       	    if((*ptr) & 32) ptr++;

		       	    if(item < MAX_UNIQUE_ITEMS)
		       	    {
		       	        if(item > max_item)
		       	        	max_item = item;

		       	        instances[instances_count].items[instances[instances_count].count++] = item;
		       	    }
		       	}


		       	//instances[instances_count].clase = item;

		       	instances_count++;
		           if(instances_count == MAX_TRANSACTIONS)
		           {
		               printf("No se procesaron todas las instancias. Modifique la constante MAX_TRANSACTIONS.\n");
		           	eof = NULL;
		           }
		           else eof = fgets(buffer,15000,infile);
		       }
		    free(buffer);
            fclose(infile);
}


int main(int argc, char *argv[])
{
	ftime( &T1 );
	if (argc < 4 || argc > 5) {
        fprintf(stderr, "Usage: %s <filename> <min_support> <output_rules_file> [min_netconf_pos]\n", argv[0]);
        return EXIT_FAILURE;
    }

    transactions = (Transaction*) malloc (MAX_TRANSACTIONS * sizeof(Transaction));


    /*
     * Variante II
     */
    //block_pool = (Block*)malloc(MAX_BLOCKS * sizeof(Block));
    //inverted_index = (InvertedIndexItem*)calloc(MAX_UNIQUE_ITEMS, sizeof(InvertedIndexItem));
    /*
     * Variante II
     */

    Bti* b = (Bti*) malloc(sizeof(Bti));

    const char *filename = argv[1];
    float min_support = atof(argv[2]);

    if (min_support < 0 || min_support > 1) {
        fprintf(stderr, "Min support must be between 0 and 1\n");
        return EXIT_FAILURE;
    }

    const char *fileout = argv[3];

    double min_netconf_pos = MIN_NETCONF;
    if (argc >= 5) {
        min_netconf_pos = atof(argv[4]);
    }
    if (min_netconf_pos < 0.0) {
        fprintf(stderr, "Error: min_netconf_pos must be >= 0.0\n");
        return EXIT_FAILURE;
    }

#if M4_GENERATE_NEGATIVES
    fprintf(stderr, "Info DMC-PN: min_netconf_pos=%g; negative rules use Netconf(X=>c)<0\n",
            min_netconf_pos);
#else
    fprintf(stderr, "Info DMC-P: min_netconf_pos=%g; positive direct maximal rules only\n",
            min_netconf_pos);
#endif

    read_transactions(filename, min_support, b);

    unsigned int min_support_abs = (unsigned int)ceil(min_support * transactions_count);
    if (min_support_abs < 1) min_support_abs = 1;

    printf("Relative min_support = %.6f | Absolute min_support = %u\n",
           min_support, min_support_abs);

    /*
     * Variante II
     */
    /*
    // Inicializar listas maximales
        for (int i = 1; i <= MAX_ITEMS; i++) {
            maximal_list[i] = (MaximalList*)malloc(sizeof(MaximalList));
            maximal_list[i]->count = 0;
        }

        unsigned int maximalTotal = 0;
        unsigned int block_pool_idx = 0;
        unsigned int current_itemset[MAX_ITEMS] = {0};
    */
    /*
     * Variante II
     */

    /*
     * Variante I
     */
    tempHash = (unsigned int*) calloc(POW_DOS_MAX_ITEMS, sizeof(unsigned int));
    find_maximal_frequent_patterns((float)min_support_abs, b, transactions_count);
    /*
     * Variante I
     */

    /*
     * Variante II
     */
    /*
    find_maximal_frequent(b, current_itemset, 0, NULL, 0, min_support, 0, transactions_count, &maximalTotal, &block_pool_idx);
    printf("Total de Maximales %d\n", maximalTotal);

    for (int i = 0; i < MAX_UNIQUE_ITEMS; i++)
       if (inverted_index[i].list) free(inverted_index[i].list);

    free(inverted_index);
    free(block_pool);
    */
    /*
    * Variante II
    */

    //printMaximales();
    //printCARs();
/* The same direct-maximal generator is used in both builds.
     * M4_GENERATE_NEGATIVES=0 yields positive-only output; =1 adds formal
     * negative rules satisfying Netconf(X=>c)<0. */
    buildCARs_M4_direct_maximal_pos_neg(b, min_support_abs, min_netconf_pos);
    // Eliminar duplicados (misma regla generada desde distintos maximales)
    dedup_CARS_keep_best_netconf();
    sortCARS(fileout);
    //clasificacionExacta();
    //printCARsNew();

    /*
     * Liberando Memoria
     */
    for(int i = 1; i <= MAX_ITEMS; i++)
       free(maximal_list[i]);

    for(int i = 0; i < BLOOM_FILTER_SIZE; i++)
    	if(bloom[i].sets_count > 0)
    		free(bloom[i].sets);

    free(tempHash);
    free(bloom);

    free(uno_frecuentes->items);
    free(uno_frecuentes->items_support);
    free(uno_frecuentes->map1_N);
    free(uno_frecuentes);

    free(b->bti_data);
    free(b->bti);
    free(b);

    for(int i = 0; i < transactions_count; i++)
       free(transactions[i].items);

    free(transactions);
    free(clases);
   /*
    * Liberando Memoria
    */

    ftime( &T2 );
    int milisecs = ( ( T2.time - T1.time ) * 1000 ) + T2.millitm - T1.millitm;
    printf( "La búsqueda tardó ,\t %d sec, %d millisec\n", milisecs / 1000, milisecs % 1000 );

    /*
     * Código para chequear fuga de memoria
     */
    /*
    #ifdef DEBUG_MEMORY
    printf("Memoria no liberada al final: %u bytes\n", memory_allocated);
    #endif
    */
    /*
    * Código para chequear fuga de memoria
    */

    return EXIT_SUCCESS;
}
