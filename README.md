# DDMCS_project
Repository for the final project of the **D**ata-**D**riven **M**odeling of **C**omplex **S**ystems (DDMCS) course held by prof. Walter Quattrociocchi.


# ROADMAP
* mi serve una funzione che mi pulisce tutti i dati e me li mette in data/cleaned. Con questi dati devo essere in grado di fare tutte le varie analisi che mi si presentano.
* clean_data.py, sentiment.py [finetune] e sentiment.py [generate] devono essere runnate prima e indipendentemnte da main.py. In particolare sentiment.py [generate] deve generare nuovi dati pronti per essere analizzati. [generate] deve creare csv files sia per l'analisi che comprende tutti i cleaned data sia per l'analisi suddivisa per LLMs topic.
* utils.py comprene funzioni per fare i plots.
* devo avere un main che può runnare a seconda degli argomenti le varie analisi quantitative (analysis/quantitative.py) e quelle inerenti alla sentiment analysis (l'analisi dei dati generati avverrà nello stesso main.py).
 
