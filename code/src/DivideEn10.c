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
#include <stdbool.h>
// directivas incluidas

typedef struct
{
  int* elementos;
  int longitud;
  int tamanno;
} CConjunto;

typedef struct
{
	CConjunto* items;
	int clase;
} CTransaccion;

// definicion de variables
int* clasesCOLECCION, cCLASES = 0 , tCLASES = 100, tBuffer = 2000;
int* clasesCOLECCION_SOPORTES;
CTransaccion*** listaTRANSACCION_CLASE;
int* cantidadTRANSACCIONES, * totalTRANSACCIONES;
const int tamanno_arregloCjt = 6000;
int* seleccionador, cantidadSELECCIONADOR = 0;
CTransaccion *** lista_10_folds_VALIDATIONS;
int * cantidadTRANSACCIONES10, * totalTRANSACCIONES10;
char ** nombres_files;
char ** nombres_DATASETS;
// definicion de variables

// prototipos de los metodos
int process_arguments(int argc, char *argv[]);
void Leer_de_Fichero(char * nombre_de_fichero);
void Formar10_Cross_validations(int porciento);
void Generar_combinaciones();
void Escribir_files();
void Conforma_dataset(int index, char * nombre_dataset);
void Crea_Datos_NACCAR(int argc, char *argv[]);
void Escribir_CLASSES();
// prototipos de los metodos

// validar argumentos
int process_arguments(int argc, char *argv[])
{
   int i;
   if (argc != 3)
   {
      printf("\nError! There are 3 mandatory argument!\n");
      return 1;
   }

   FILE* f = fopen(argv[1],"r");
   if(f == NULL)
   {
      printf("\nError! The file %s can not be read!\n",argv[1]);
      return 1;
   }
   else fclose(f);

   int l = strlen(argv[2]);
   for(i=0; i < l; i++)
      if (argv[2][i] < '0' || argv[2][i] > '9')
      {
         printf("\nError! The second parameters must be an integer!\n");
         return 1;
      }

   return 0;
}

//leer fichero
void Leer_de_Fichero(char* nombre_de_fichero)
{
   //reservar memoria
   cantidadTRANSACCIONES = 0;
   cantidadTRANSACCIONES = (int*) malloc(tCLASES * sizeof(int));
   totalTRANSACCIONES = 0;
   totalTRANSACCIONES = (int*) malloc(tCLASES * sizeof(int));
   clasesCOLECCION = 0;
   clasesCOLECCION = (int*) malloc(tCLASES * sizeof(int));
   clasesCOLECCION_SOPORTES = 0;
   clasesCOLECCION_SOPORTES = (int*) calloc(tCLASES, sizeof(int));
   listaTRANSACCION_CLASE = (CTransaccion***) malloc(tCLASES * sizeof(CTransaccion**));
   CConjunto* iConjunto = 0;
   iConjunto = (CConjunto*) malloc(sizeof(CConjunto));
   iConjunto->elementos = 0;
   iConjunto->elementos = (int*) malloc(tamanno_arregloCjt * sizeof(int));
   iConjunto->tamanno = tamanno_arregloCjt;

   FILE* fichero = fopen(nombre_de_fichero,"r");

   char* buffer = 0;
   buffer = (char*) malloc(sizeof(char) * tBuffer);

   char* temp = 0;
   temp = (char*) malloc(sizeof(char) * 30);

   int inicio, j, elem = 0, iPOS = -1;
   char* eof = fgets(buffer,tBuffer,fichero);
   while(eof!= 0)
   {
  	  inicio = 0;
  	  iConjunto->longitud = 0;
      while(buffer[inicio] != '\n')
  	  {
  	     j = 0;
  		 while(buffer[inicio] != ' ' && buffer[inicio] != '\n')
  		    temp[j++] = buffer[inicio++];

  		 temp[j] = '\0';
  		 elem = atoi(temp);
  		 if(iConjunto->longitud == iConjunto->tamanno)
  		 {
  		    iConjunto->tamanno *= 2;
  		    iConjunto->elementos = (int*) realloc(iConjunto->elementos, sizeof(int) * iConjunto->tamanno);
  		 }

  		 iConjunto->elementos[iConjunto->longitud++] = elem;

  		 if(buffer[inicio] != '\n')
            inicio++;
  	  }

  	  iPOS = cCLASES;
  	  int i;
	  for(i=0; i < iPOS; i++)
	     iPOS = (clasesCOLECCION[i] == iConjunto->elementos[iConjunto->longitud - 1]) ? i : cCLASES;

	  if(iPOS == cCLASES)
	  {
	     // no se habia leido anteriormente ninguna transaccion de esta clase
	     if(cCLASES == tCLASES)
	     {
	        tCLASES *= 2;
		    listaTRANSACCION_CLASE = (CTransaccion***) realloc(listaTRANSACCION_CLASE,tCLASES * sizeof(CTransaccion**));
		    cantidadTRANSACCIONES = (int*) realloc(cantidadTRANSACCIONES, tCLASES * sizeof(int));
		    totalTRANSACCIONES = (int*) realloc(totalTRANSACCIONES, tCLASES * sizeof(int));
		    clasesCOLECCION = (int*) realloc(clasesCOLECCION, tCLASES * sizeof(int));
		    clasesCOLECCION_SOPORTES = (int*) realloc(clasesCOLECCION_SOPORTES, tCLASES * sizeof(int));
	     }

	     cCLASES++;
	     clasesCOLECCION[iPOS] = iConjunto->elementos[iConjunto->longitud - 1];
	     totalTRANSACCIONES[iPOS] = tamanno_arregloCjt;
	     cantidadTRANSACCIONES[iPOS] = 0;
	     listaTRANSACCION_CLASE[iPOS] = 0;
	     listaTRANSACCION_CLASE[iPOS] = (CTransaccion**) malloc(sizeof(CTransaccion*) * totalTRANSACCIONES[iPOS]);
	     // no se habia leido anteriormente ninguna transaccion de esta clase
	  }
	  else if(cantidadTRANSACCIONES[iPOS] == totalTRANSACCIONES[iPOS])
	       {
	          totalTRANSACCIONES[iPOS] *=2;
		      listaTRANSACCION_CLASE[iPOS] = (CTransaccion**) realloc(listaTRANSACCION_CLASE[iPOS], sizeof(CTransaccion*) * totalTRANSACCIONES[iPOS]);
	       }

	  clasesCOLECCION_SOPORTES[iPOS]++;

	  listaTRANSACCION_CLASE[iPOS][cantidadTRANSACCIONES[iPOS]] = 0;
	  listaTRANSACCION_CLASE[iPOS][cantidadTRANSACCIONES[iPOS]] = (CTransaccion*) malloc(sizeof(CTransaccion));
	  listaTRANSACCION_CLASE[iPOS][cantidadTRANSACCIONES[iPOS]]->items = 0;
	  listaTRANSACCION_CLASE[iPOS][cantidadTRANSACCIONES[iPOS]]->items = (CConjunto*) malloc(sizeof(CConjunto));
	  listaTRANSACCION_CLASE[iPOS][cantidadTRANSACCIONES[iPOS]]->items->longitud = 0;
	  listaTRANSACCION_CLASE[iPOS][cantidadTRANSACCIONES[iPOS]]->items->tamanno = tamanno_arregloCjt;
	  listaTRANSACCION_CLASE[iPOS][cantidadTRANSACCIONES[iPOS]]->items->elementos = 0;
	  listaTRANSACCION_CLASE[iPOS][cantidadTRANSACCIONES[iPOS]]->items->elementos = (int*) malloc(sizeof(int) * listaTRANSACCION_CLASE[iPOS][cantidadTRANSACCIONES[iPOS]]->items->tamanno);
	  listaTRANSACCION_CLASE[iPOS][cantidadTRANSACCIONES[iPOS]]->clase = iConjunto->elementos[iConjunto->longitud - 1];

	  for(i=0; i < iConjunto->longitud - 1; i++)
	  {
	     if (listaTRANSACCION_CLASE[iPOS][cantidadTRANSACCIONES[iPOS]]->items->longitud == listaTRANSACCION_CLASE[iPOS][cantidadTRANSACCIONES[iPOS]]->items->tamanno)
	     {
	        listaTRANSACCION_CLASE[iPOS][cantidadTRANSACCIONES[iPOS]]->items->tamanno *= 2;
		    listaTRANSACCION_CLASE[iPOS][cantidadTRANSACCIONES[iPOS]]->items->elementos = (int*) realloc(listaTRANSACCION_CLASE[iPOS][cantidadTRANSACCIONES[iPOS]]->items->elementos,sizeof(int) * listaTRANSACCION_CLASE[iPOS][cantidadTRANSACCIONES[iPOS]]->items->tamanno);
	     }

	     listaTRANSACCION_CLASE[iPOS][cantidadTRANSACCIONES[iPOS]]->items->elementos[listaTRANSACCION_CLASE[iPOS][cantidadTRANSACCIONES[iPOS]]->items->longitud++] = iConjunto->elementos[i];
	  }

	  cantidadTRANSACCIONES[iPOS]++;
  	  // comprobar la clase de la transaccion

  	  eof = fgets(buffer,tBuffer,fichero);
   }

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

   listaTRANSACCION_CLASE = (CTransaccion***) realloc(listaTRANSACCION_CLASE, cCLASES * sizeof(CTransaccion**));
   cantidadTRANSACCIONES = (int*) realloc(cantidadTRANSACCIONES, cCLASES * sizeof(int));
   totalTRANSACCIONES = (int*) realloc(totalTRANSACCIONES, cCLASES * sizeof(int));
   clasesCOLECCION = (int*) realloc(clasesCOLECCION, cCLASES * sizeof(int));
   clasesCOLECCION_SOPORTES = (int*) realloc(clasesCOLECCION_SOPORTES, cCLASES * sizeof(int));

   int i;
   for(i=0; i < cCLASES; i++)
   {
      listaTRANSACCION_CLASE[i] = (CTransaccion**) realloc(listaTRANSACCION_CLASE[i], sizeof(CTransaccion*) * cantidadTRANSACCIONES[i]);
	  printf("Clase %i con %i elementos\n", i + 1,cantidadTRANSACCIONES[i]);
   }
}

void Escribir_CLASSES()
{
   FILE* ficheroWRITE = 0;
   ficheroWRITE = fopen("Classes.dat","w");

   int i;
   for(i=0; i < cCLASES; i++)
      fprintf(ficheroWRITE,"%i %i\n",clasesCOLECCION[i], clasesCOLECCION_SOPORTES[i]);

   fclose(ficheroWRITE);
   ficheroWRITE = 0;
}

void Formar10_Cross_validations(int porciento)
{
   lista_10_folds_VALIDATIONS = (CTransaccion***) malloc(sizeof(CTransaccion**) * 10);
   cantidadTRANSACCIONES10 = 0;
   cantidadTRANSACCIONES10 = (int*) malloc(sizeof(int) * 10);
   totalTRANSACCIONES10 = 0;
   totalTRANSACCIONES10 = (int*) malloc(sizeof(int) * 10);
   int i;
   for(i=0; i <  10; i ++)
   {
	  cantidadTRANSACCIONES10[i] = 0;
	  totalTRANSACCIONES10[i] = tamanno_arregloCjt;
	  lista_10_folds_VALIDATIONS[i] = (CTransaccion**) malloc(sizeof(CTransaccion*) * totalTRANSACCIONES10[i]);
   }

   // repartir las transacciones de las clases por las 10 carpetas
   int index = 0;
   for(i=0; i < cCLASES; i++)
   {
      cantidadSELECCIONADOR = cantidadTRANSACCIONES[i];
	  seleccionador = 0;
	  seleccionador = (int*) malloc(sizeof(int) * cantidadSELECCIONADOR);
	  int j;
	  for(j=0; j < cantidadSELECCIONADOR; j++)
	     seleccionador[j] = j;

	  Generar_combinaciones();
	  int k = 0;
	  while(k < cantidadSELECCIONADOR)
	  {
	     if(cantidadTRANSACCIONES10[index] == totalTRANSACCIONES10[index])
		 {
		    totalTRANSACCIONES10[index] *=2;
		    lista_10_folds_VALIDATIONS[index] = (CTransaccion**) realloc(lista_10_folds_VALIDATIONS[index], sizeof(CTransaccion*) * totalTRANSACCIONES10[index]);
		 }
		 lista_10_folds_VALIDATIONS[index][cantidadTRANSACCIONES10[index]] = 0;
		 lista_10_folds_VALIDATIONS[index][cantidadTRANSACCIONES10[index]] = (CTransaccion*) malloc(sizeof(CTransaccion));
		 lista_10_folds_VALIDATIONS[index][cantidadTRANSACCIONES10[index]]->items = 0;
		 lista_10_folds_VALIDATIONS[index][cantidadTRANSACCIONES10[index]]->items = (CConjunto*) malloc(sizeof(CConjunto));
		 lista_10_folds_VALIDATIONS[index][cantidadTRANSACCIONES10[index]]->items->longitud = 0;
		 lista_10_folds_VALIDATIONS[index][cantidadTRANSACCIONES10[index]]->items->tamanno = listaTRANSACCION_CLASE[i][seleccionador[k]]->items->longitud;
		 lista_10_folds_VALIDATIONS[index][cantidadTRANSACCIONES10[index]]->items->elementos = 0;
		 lista_10_folds_VALIDATIONS[index][cantidadTRANSACCIONES10[index]]->items->elementos = (int*) malloc(sizeof(int) * lista_10_folds_VALIDATIONS[index][cantidadTRANSACCIONES10[index]]->items->tamanno);
		 lista_10_folds_VALIDATIONS[index][cantidadTRANSACCIONES10[index]]->clase = listaTRANSACCION_CLASE[i][seleccionador[k]]->clase;

		 int l;
		 for(l = 0; l < listaTRANSACCION_CLASE[i][seleccionador[k]]->items->longitud; l++)
		 {
		    lista_10_folds_VALIDATIONS[index][cantidadTRANSACCIONES10[index]]->items->elementos[lista_10_folds_VALIDATIONS[index][cantidadTRANSACCIONES10[index]]->items->longitud] = listaTRANSACCION_CLASE[i][seleccionador[k]]->items->elementos[l];
		    lista_10_folds_VALIDATIONS[index][cantidadTRANSACCIONES10[index]]->items->longitud++;
		 }

		 cantidadTRANSACCIONES10[index]++;
		 k++;

   	     index = (index + 1) % 10;
	  }

	  free(seleccionador);
	  seleccionador = 0;
   }

   // repartir las transacciones de las clases por las 10 carpetas
   for(i=0; i < cCLASES; i++)
   {
	  int j;
	  for(j=0; j < cantidadTRANSACCIONES[i]; j++)
	  {
	     free(listaTRANSACCION_CLASE[i][j]->items->elementos);
	     listaTRANSACCION_CLASE[i][j]->items->elementos = 0;
	     free(listaTRANSACCION_CLASE[i][j]->items);
	     listaTRANSACCION_CLASE[i][j]->items = 0;

	     free(listaTRANSACCION_CLASE[i][j]);
	     listaTRANSACCION_CLASE[i][j] = 0;
	  }

  	  free(listaTRANSACCION_CLASE[i]);
  	  listaTRANSACCION_CLASE[i] = 0;
   }

   free(listaTRANSACCION_CLASE);
   listaTRANSACCION_CLASE = 0;

   free(cantidadTRANSACCIONES);
   cantidadTRANSACCIONES = 0;

   free(totalTRANSACCIONES);
   totalTRANSACCIONES = 0;
}

void Generar_combinaciones()
{
   srand(time(NULL));
   int i, der, auxiliar;
   for(i=0; i < cantidadSELECCIONADOR; i++)
   {
      der = rand() % cantidadSELECCIONADOR;
      if(i != der)
      {
         auxiliar = seleccionador[i];
         seleccionador[i] = seleccionador[der];
         seleccionador[der] = auxiliar;
      }
   }
}

void Escribir_files()
{
   FILE* fichero = 0;
   nombres_files = (char**) malloc(sizeof(char*) * 10);
   int i, j;
   for(i=0; i < 10; i++)
   {
      nombres_files[i] = (char*) malloc(sizeof(char) * 10);
  	  sprintf(nombres_files[i],"%i.dat",i+1);
	  fichero = fopen(nombres_files[i],"w");

	  for(j=0; j < cantidadTRANSACCIONES10[i]; j++)
	  {
	     int k;
	     for(k=0; k < lista_10_folds_VALIDATIONS[i][j]->items->longitud; k++)
	     {
	  	    int elem = lista_10_folds_VALIDATIONS[i][j]->items->elementos[k];
	  	    fprintf(fichero, "%i ", elem);
	     }
	     fprintf(fichero, "%i\n", lista_10_folds_VALIDATIONS[i][j]->clase);
	  }

	  fclose(fichero);
	  fichero = 0;
   }

   for(i=0; i < 10; i++)
   {
	  for(j=0; j < cantidadTRANSACCIONES10[i]; j++)
	  {
	     free(lista_10_folds_VALIDATIONS[i][j]->items->elementos);
	     lista_10_folds_VALIDATIONS[i][j]->items->elementos = 0;
	     free(lista_10_folds_VALIDATIONS[i][j]->items);
	     lista_10_folds_VALIDATIONS[i][j]->items = 0;

	     free(lista_10_folds_VALIDATIONS[i][j]);
	     lista_10_folds_VALIDATIONS[i][j] = 0;
	  }

  	  free(lista_10_folds_VALIDATIONS[i]);
  	  lista_10_folds_VALIDATIONS[i] = 0;
   }

   free(lista_10_folds_VALIDATIONS);
   lista_10_folds_VALIDATIONS = 0;

   free(cantidadTRANSACCIONES10);
   cantidadTRANSACCIONES10 = 0;

   free(totalTRANSACCIONES10);
   totalTRANSACCIONES10 = 0;
}

void Conforma_dataset(int ind, char* nombre_dataset)
{
   FILE* ficheroREAD = 0;
   FILE* ficheroWRITE = 0;
   ficheroWRITE = fopen(nombre_dataset,"w");

   char* buffer = 0;
   buffer = (char*) malloc(sizeof(char) * tBuffer);
   char* eof = 0;
   int i;
   for(i=0; i < 10; i++)
   {
      if(i != ind)
	  {
	     ficheroREAD = fopen(nombres_files[i],"r");
	     eof = fgets(buffer,tBuffer,ficheroREAD);
	     while(eof != 0)
	     {
	        fprintf(ficheroWRITE,"%s",buffer);
		    eof = fgets(buffer,tBuffer,ficheroREAD);
	     }

 	     fclose(ficheroREAD);
	     ficheroREAD = 0;
	  }
   }

   fclose(ficheroWRITE);
   ficheroWRITE = 0;

   free(buffer);
   buffer = 0;
}

void Crea_Datos_NACCAR(int argc, char *argv[])
{
   printf("Iniciando...\n");
   Leer_de_Fichero(argv[1]);
   Escribir_CLASSES();
   printf("readed...\n");
   Formar10_Cross_validations(atoi(argv[2]));
   printf("10 fold-cross validations prepared...\n");
   Escribir_files();
   printf("10 fold-cross validations printed...\n");
   nombres_DATASETS = 0;
   nombres_DATASETS = (char**) malloc(sizeof(char*) * 10);
   int i;
   for(i=0; i < 10; i++)
   {
      nombres_DATASETS[i] = (char*) malloc(sizeof(char) * 20);
  	  sprintf(nombres_DATASETS[i],"Dataset%i.dat",i+1);

	  Conforma_dataset(i, nombres_DATASETS[i]);
   }
   printf("Creado cjto de datos!!\n");

   for(i=0; i < 10; i++)
   {
      free(nombres_DATASETS[i]);
	  nombres_DATASETS[i] = 0;
	  free(nombres_files[i]);
	  nombres_files[i] = 0;
   }

   free(nombres_DATASETS);
   nombres_DATASETS = 0;
   free(nombres_files);
   nombres_files = 0;

   free(clasesCOLECCION);
   clasesCOLECCION = 0;

   free(clasesCOLECCION_SOPORTES);
   clasesCOLECCION_SOPORTES = 0;
}

// implementacion de los metodos
int main(int argc, char *argv[])
{
   if(process_arguments(argc,argv))
      return 1;
   Crea_Datos_NACCAR(argc,argv);
   return EXIT_SUCCESS;
}

