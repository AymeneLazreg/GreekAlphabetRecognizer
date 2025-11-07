compielr avec 
javac --module-path "C:\javafx-sdk-21\lib" --add-modules javafx.controls,javafx.fxml,javafx.swing SymbolRecognizer.jav

Génerer le jar avec 
jar cfm SymbolRecognizer.jar manifest.txt SymbolRecognizer.class

Executer avec : 
java --module-path "C:\javafx-sdk-21\lib" --add-modules javafx.controls,javafx.fxml,javafx.swing -jar SymbolRecognizer.jar


Chosiissez votre template de gestes a reconnaitre (alphabet grec dans notre cas) avec "charger dossier template"
(Les images doivent entre idealement en NOIR sur BLANC avec comme nom (lettre_n)
On choisis le nombre de points (divise les geste en plusieurs point puis clacul les vecteurs entre les points)
(Plus grand nombre de points =/= plus de precision)

