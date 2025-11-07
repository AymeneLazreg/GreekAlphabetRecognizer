# SymbolRecognizer – Reconnaissance de gestes (Alphabet grec)

Application JavaFX permettant de reconnaître des gestes dessinés à la main à partir de modèles (templates), en utilisant un algorithme de reconnaissance basé sur les points caractéristiques et les vecteurs directionnels, avec auto-tuning et agrégation des résultats pour une décision plus stable.

## Prérequis

- Java 11+
- JavaFX 21
- Système compatible avec JavaFX (Windows, Linux, macOS)

## Compilation et exécution

### Compilation
Assurez-vous que JavaFX SDK est installé et accessible sur votre machine.  


> Compilez le programme avec la commande : 
> ```
> javac --module-path "C:\javafx-sdk-21\lib" --add-modules javafx.controls,javafx.fxml,javafx.swing SymbolRecognizer.java
> ```


### Génération du fichier JAR
>Créez le fichier `SymbolRecognizer.jar` avec :
> ```
> jar cfm SymbolRecognizer.jar manifest.txt SymbolRecognizer.class
> ```


### Exécution
>Exécutez l’application avec :
> ```
> java --module-path "C:\javafx-sdk-21\lib" --add-modules javafx.controls,javafx.fxml,javafx.swing -jar SymbolRecognizer.jar
> ```


## Utilisation

### Chargement des templates
1. Cliquez sur "Charger dossier templates".  
2. Sélectionnez le dossier contenant vos modèles (lettres de l’alphabet grec).

### Format des images
- Images en noir sur fond blanc.  
- Nommage obligatoire : `lettre_1.png, lettre_2.png, ...`  
  Exemple : `alpha_1.png`, `beta_2.png`.

### Paramétrage du nombre de points
- Le nombre de points N permet de discrétiser les gestes pour calculer les vecteurs directionnels.  
- Plus N est grand ne veut pas forcement dire plus de précision, mais temps de calcul plus long.  
- Vous pouvez modifier N avec le spinner dans l’interface.

### Reconnaissance et Auto-tuning
1. Cliquez sur "Charger image à reconnaître" pour sélectionner votre dessin.  
2. Cliquez sur "Reconnaître" pour identifier le symbole à partir des templates chargés.  
3. Cliquez sur "Auto-tune N (agg.)" pour tester plusieurs valeurs de N et obtenir :
   - La décision agrégée du label reconnu
   - La confiance de la reconnaissance
   - La valeur recommandée de N pour ce symbole
   - Les distances moyennes et écart-types  

> L’auto-tune teste plusieurs valeurs de N et agrège les résultats pour stabiliser la reconnaissance.



## Principe de fonctionnement

1. Charge les templates depuis un dossier.  
2. Binarise l’image (noir sur blanc) et détecte le plus grand composant connecté.  
3. Extrait la frontière du symbole et ordonne les points autour du centre.  
4. Rééchantillonne les points en N positions uniformes.  
5. Normalise et génère un vecteur de caractéristiques.  
6. Compare le vecteur avec ceux des templates pour identifier le symbole le plus proche.  
7. Optionnel : Auto-tune sur plusieurs N et agrégation des résultats pour stabiliser la reconnaissance.

## Technologies

- Java 11+
- JavaFX 21 (controls, fxml, swing)
- Algorithme de reconnaissance gestuelle basé sur vecteurs de points normalisés
- Auto-tuning et agrégation pour robustesse

## Auteur

Aymene LAZREG
Projet académique Module HAI702 – Reconnaissance de gestes
Université de Montpellier – Faculté des Sciences