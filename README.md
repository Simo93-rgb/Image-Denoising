# Image Denoising

Repository per il progetto Image Denoising del corso di Apprendimento Automatico e Apprendimento Profondo presso l'Università degli Studi del Piemonte Orientale.

## Progetto 2: Rimozione del Rumore (Image Denoising) con Autoencoder

Il secondo progetto consiste nel realizzare un sistema di pulizia delle immagini (*image denoising*). Verrà implementata un'architettura basata su **Autoencoder** che, prendendo in input un'immagine soggetta a "rumore", restituirà una sua versione "pulita".

### Dataset e Metodologia

Il modello sarà addestrato sul dataset **Fashion-MNIST**. Per ottenere l'input per la rete, verrà applicato un opportuno filtro di rumore (*noise*) alle immagini originali. L'autoencoder imparerà a ricostruire l'immagine di partenza, rimuovendo di fatto il disturbo aggiunto.