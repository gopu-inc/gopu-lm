#!/usr/bin/env python3
"""Ajoute un lot reproductible de conversations françaises variées au corpus gopu."""
import argparse
import random
from pathlib import Path


# Chaque entrée contient une réponse autonome et trois formulations de la demande.
# Les réponses restent concises pour éviter d'enseigner des détails incertains.
FACTS = [
    ("Rust", "Rust est un langage compilé conçu pour la performance et la sûreté mémoire. Il est utilisé notamment pour les outils système et les services.", ["À quoi sert Rust ?", "Présente-moi le langage Rust.", "Pourquoi choisir Rust pour un programme ?"]),
    ("Ownership", "En Rust, l'ownership définit quel propriétaire contrôle une valeur. Le compilateur vérifie les règles de propriété afin d'éviter plusieurs erreurs de mémoire.", ["Qu'est-ce que l'ownership en Rust ?", "Explique la propriété des valeurs en Rust.", "À quoi sert l'ownership ?"]),
    ("Borrowing", "L'emprunt permet d'utiliser une référence vers une valeur sans en prendre la propriété. Les règles de Rust limitent les références mutables simultanées.", ["Que signifie emprunter une valeur en Rust ?", "Explique les références en Rust.", "Quelle est l'idée du borrowing ?"]),
    ("Result", "`Result<T, E>` représente soit une réussite contenant une valeur de type T, soit une erreur de type E. Il permet de traiter explicitement les erreurs.", ["À quoi sert Result en Rust ?", "Comment représenter une erreur en Rust ?", "Explique le type Result simplement."]),
    ("Option", "`Option<T>` représente une valeur éventuellement absente : `Some(valeur)` ou `None`. Cela évite d'utiliser un pointeur nul pour exprimer l'absence.", ["Que représente Option en Rust ?", "Comment signaler qu'une valeur peut manquer en Rust ?", "Explique Some et None."]),
    ("Vec", "`Vec<T>` est un tableau dynamique qui stocke des valeurs de type T dans un ordre donné. Sa taille peut évoluer pendant l'exécution.", ["Qu'est-ce qu'un Vec en Rust ?", "Comment stocker une liste extensible en Rust ?", "À quoi sert Vec<T> ?"]),
    ("Slice", "Une slice est une vue empruntée sur une séquence, par exemple une partie d'un tableau ou d'un vecteur. Elle ne possède pas les éléments.", ["Qu'est-ce qu'une slice en Rust ?", "Explique &[T] simplement.", "Une slice possède-t-elle ses données ?"]),
    ("Trait", "Un trait décrit un ensemble de comportements qu'un type peut implémenter. Il permet de partager des interfaces entre plusieurs types.", ["À quoi sert un trait en Rust ?", "Explique les traits Rust.", "Comment partager un comportement entre des types Rust ?"]),
    ("Enum", "Une énumération Rust définit plusieurs variantes possibles d'une même valeur. Chaque variante peut aussi contenir des données.", ["Qu'est-ce qu'une enum en Rust ?", "Comment représenter plusieurs cas en Rust ?", "Explique les variantes d'une énumération."]),
    ("Cargo", "Cargo est l'outil de Rust qui gère notamment les projets, la compilation, les tests et les dépendances.", ["À quoi sert Cargo ?", "Comment s'appelle l'outil de projet Rust ?", "Que peut faire Cargo ?"]),
    ("Python list", "En Python, une liste (`list`) est une séquence ordonnée et modifiable. Elle peut contenir plusieurs valeurs.", ["Qu'est-ce qu'une liste Python ?", "Comment fonctionne list en Python ?", "Une liste Python est-elle modifiable ?"]),
    ("Python dict", "Un dictionnaire Python associe des clés à des valeurs. On récupère une valeur à partir de sa clé.", ["À quoi sert un dictionnaire Python ?", "Explique le type dict.", "Comment associer une clé à une valeur en Python ?"]),
    ("Python tuple", "Un tuple est une séquence ordonnée généralement utilisée comme valeur immuable. Il s'écrit souvent avec des parenthèses.", ["Qu'est-ce qu'un tuple Python ?", "Quelle est l'idée d'un tuple ?", "Un tuple Python est-il modifiable ?"]),
    ("Python function", "On définit une fonction Python avec `def`, un nom, des paramètres éventuels et un bloc indenté. `return` peut renvoyer un résultat.", ["Comment déclarer une fonction en Python ?", "Montre la structure d'une fonction Python.", "À quoi sert return dans une fonction ?"]),
    ("Python exception", "Une exception signale une situation exceptionnelle pendant l'exécution. `try` et `except` permettent de traiter certaines exceptions.", ["Qu'est-ce qu'une exception Python ?", "Comment gérer une erreur en Python ?", "À quoi servent try et except ?"]),
    ("Python module", "Un module Python est un fichier contenant du code réutilisable. On peut importer ses éléments avec `import`.", ["Qu'est-ce qu'un module Python ?", "Comment réutiliser le code d'un autre fichier Python ?", "À quoi sert import ?"]),
    ("JSON", "JSON est un format textuel courant pour représenter des données structurées avec des objets, des tableaux, des chaînes, des nombres, des booléens et null.", ["Qu'est-ce que JSON ?", "À quoi sert le format JSON ?", "Quelles données peut contenir JSON ?"]),
    ("CSV", "CSV est un format texte tabulaire où les lignes représentent généralement des enregistrements et les champs sont séparés par un délimiteur, souvent une virgule.", ["Qu'est-ce qu'un fichier CSV ?", "Comment est organisé un CSV ?", "À quoi sert le format CSV ?"]),
    ("Git commit", "Un commit Git enregistre un état du projet dans l'historique, avec un identifiant et un message descriptif.", ["Qu'est-ce qu'un commit Git ?", "À quoi sert git commit ?", "Que contient un commit ?"]),
    ("Git branch", "Une branche Git est une ligne de développement nommée. Elle permet de travailler sur des changements sans modifier immédiatement une autre branche.", ["Qu'est-ce qu'une branche Git ?", "Pourquoi utiliser une branche ?", "À quoi sert git branch ?"]),
    ("HTTP", "HTTP est un protocole de communication utilisé notamment entre un client et un serveur web. HTTPS ajoute une couche de chiffrement TLS.", ["À quoi sert HTTP ?", "Quelle différence générale entre HTTP et HTTPS ?", "Explique le protocole web HTTP."]),
    ("URL", "Une URL indique l'emplacement d'une ressource et peut contenir un schéma, un hôte, un chemin et parfois des paramètres.", ["Qu'est-ce qu'une URL ?", "Quels éléments trouve-t-on souvent dans une URL ?", "À quoi sert une URL ?"]),
    ("API", "Une API définit une manière pour des logiciels d'échanger des données ou de demander des opérations selon une interface convenue.", ["Qu'est-ce qu'une API ?", "À quoi sert une API ?", "Explique une API avec des mots simples."]),
    ("Database", "Une base de données organise des informations afin qu'un logiciel puisse les consulter et les modifier. Le choix du système dépend du besoin.", ["À quoi sert une base de données ?", "Qu'est-ce qu'une base de données ?", "Pourquoi une application utilise-t-elle une base de données ?"]),
    ("Encryption", "Le chiffrement transforme des données à l'aide d'une clé pour que leur contenu ne soit pas lisible sans le mécanisme de déchiffrement approprié.", ["Qu'est-ce que le chiffrement ?", "À quoi sert le chiffrement des données ?", "Explique le principe du chiffrement."]),
    ("Password", "Un bon mot de passe est long, unique et difficile à deviner. Un gestionnaire de mots de passe aide à en créer et mémoriser un différent pour chaque service.", ["Comment choisir un mot de passe sûr ?", "Pourquoi ne pas réutiliser le même mot de passe ?", "Comment mieux protéger mes comptes ?"]),
    ("Backup", "Une sauvegarde est une copie séparée de données que l'on peut utiliser après une perte ou une panne. Il faut vérifier régulièrement qu'elle peut être restaurée.", ["Qu'est-ce qu'une sauvegarde ?", "Pourquoi faire des copies de sécurité ?", "Comment savoir si ma sauvegarde fonctionne ?"]),
    ("CPU", "Le processeur exécute les instructions d'un programme. Sa vitesse n'est qu'un facteur parmi d'autres dans les performances d'un ordinateur.", ["À quoi sert le processeur ?", "Qu'est-ce qu'un CPU ?", "Le processeur fait quoi dans un ordinateur ?"]),
    ("RAM", "La mémoire vive conserve temporairement les données utilisées par les programmes en cours. Son contenu est généralement perdu lorsque l'ordinateur s'éteint.", ["Qu'est-ce que la RAM ?", "À quoi sert la mémoire vive ?", "La RAM garde-t-elle les données après extinction ?"]),
    ("SSD", "Un SSD est un support de stockage qui utilise de la mémoire flash et ne comporte pas de pièces mécaniques mobiles comme un disque dur classique.", ["Qu'est-ce qu'un SSD ?", "Comment fonctionne un support SSD en général ?", "Quelle est la caractéristique d'un SSD ?"]),
    ("Internet", "Internet est un réseau mondial de réseaux interconnectés. Le Web est un service qui fonctionne au-dessus d'Internet, avec d'autres services comme le courrier électronique.", ["Internet et le Web, est-ce pareil ?", "Explique ce qu'est Internet.", "Le Web représente-t-il tout Internet ?"]),
    ("Tokenization", "La tokenisation découpe un texte en unités appelées jetons. Un jeton peut être un caractère, une partie de mot ou parfois un mot entier.", ["Qu'est-ce que la tokenisation ?", "Comment un modèle découpe-t-il le texte ?", "Un jeton correspond-il toujours à un mot ?"]),
    ("Machine learning", "L'apprentissage automatique consiste à ajuster un modèle à partir d'exemples afin qu'il repère des régularités et produise des prédictions ou des sorties.", ["Qu'est-ce que l'apprentissage automatique ?", "Explique le machine learning simplement.", "Comment apprend un modèle informatique ?"]),
    ("Training data", "Les données d'entraînement influencent ce qu'un modèle apprend. Des exemples variés, corrects et bien structurés aident à couvrir davantage de situations.", ["Pourquoi la qualité des données d'entraînement compte-t-elle ?", "Qu'est-ce qu'un corpus d'entraînement ?", "Comment améliorer les exemples d'un modèle ?"]),
    ("Overfitting", "Le surapprentissage arrive lorsqu'un modèle s'adapte trop aux exemples d'entraînement et généralise mal à de nouveaux cas.", ["Qu'est-ce que le surapprentissage ?", "Pourquoi un modèle peut-il mémoriser au lieu de généraliser ?", "Explique l'overfitting."]),
    ("Validation", "Un jeu de validation sert à suivre les performances d'un modèle sur des exemples mis de côté pendant son entraînement.", ["À quoi sert un jeu de validation ?", "Pourquoi garder des exemples à part pendant l'entraînement ?", "Qu'est-ce que la validation d'un modèle ?"]),
    ("Gradient", "Un gradient indique comment une fonction change selon ses paramètres. Pendant l'entraînement, il aide à ajuster les poids pour réduire la perte.", ["À quoi sert le gradient dans l'entraînement ?", "Qu'est-ce qu'un gradient ?", "Comment le modèle utilise-t-il un gradient ?"]),
    ("Loss", "La perte mesure l'écart entre une prédiction et la cible selon une fonction choisie. L'entraînement cherche généralement à la réduire.", ["Qu'est-ce que la fonction de perte ?", "À quoi sert la loss d'un modèle ?", "Pourquoi cherche-t-on à réduire la perte ?"]),
    ("Transformer", "Un Transformer est une architecture de réseau neuronal qui utilise notamment des mécanismes d'attention pour traiter les relations entre éléments d'une séquence.", ["Qu'est-ce qu'un Transformer en IA ?", "À quoi sert l'attention dans un Transformer ?", "Explique l'architecture Transformer simplement."]),
    ("Attention", "Le mécanisme d'attention calcule quelles parties d'une séquence sont utiles pour traiter une position donnée. Il ne garantit pas à lui seul une réponse correcte.", ["Que fait le mécanisme d'attention ?", "Explique l'attention dans un modèle de langage.", "L'attention garantit-elle la vérité ?"]),
    ("Unit", "Dans le système international, la longueur se mesure en mètres, la masse en kilogrammes et le temps en secondes.", ["Quelles sont les unités SI de longueur, masse et temps ?", "Quelle unité utilise-t-on pour mesurer une longueur ?", "Quelles unités de base correspondent à la masse et au temps ?"]),
    ("Water", "À la pression atmosphérique normale, l'eau pure gèle près de 0 °C et bout près de 100 °C. Ces valeurs changent avec la pression et les conditions.", ["À quelle température l'eau pure gèle-t-elle ?", "Quel est le point d'ébullition de l'eau au niveau de la mer ?", "L'eau bout-elle toujours exactement à 100 °C ?"]),
    ("Photosynthesis", "La photosynthèse permet à certaines plantes, algues et bactéries d'utiliser l'énergie lumineuse pour fabriquer des molécules organiques à partir notamment de dioxyde de carbone et d'eau.", ["Qu'est-ce que la photosynthèse ?", "Comment les plantes utilisent-elles la lumière ?", "Quels organismes réalisent la photosynthèse ?"]),
    ("Gravity", "La gravité est une interaction qui attire les corps ayant une masse. Près de la surface terrestre, elle donne aux objets une accélération vers le sol.", ["Qu'est-ce que la gravité ?", "Pourquoi les objets tombent-ils vers le sol ?", "Quel effet la gravité a-t-elle près de la Terre ?"]),
    ("Earth", "La Terre tourne sur elle-même et orbite autour du Soleil. Une rotation complète dure environ un jour et une révolution environ une année.", ["Quels mouvements la Terre effectue-t-elle ?", "Combien de temps dure environ une orbite terrestre ?", "Pourquoi parle-t-on de jour et d'année ?"]),
    ("Moon", "La Lune est le satellite naturel de la Terre. Elle réfléchit la lumière du Soleil et présente différentes phases au fil de son orbite.", ["Qu'est-ce que la Lune ?", "La Lune produit-elle sa propre lumière visible ?", "Pourquoi la Lune a-t-elle des phases ?"]),
    ("Sound", "Le son est une vibration qui se propage dans un milieu matériel, comme l'air ou l'eau. Il ne se propage pas dans le vide comme dans l'air.", ["Comment le son se propage-t-il ?", "Le son voyage-t-il dans le vide ?", "Qu'est-ce qu'une onde sonore ?"]),
    ("Plant", "Les racines ancrent souvent la plante et absorbent l'eau et des éléments minéraux. Les feuilles captent la lumière et participent souvent à la photosynthèse.", ["À quoi servent généralement les racines ?", "Quel rôle jouent les feuilles ?", "Comment une plante absorbe-t-elle l'eau ?"]),
    ("Human body", "Le cœur propulse le sang dans le système circulatoire. Les poumons permettent les échanges gazeux entre l'air et le sang.", ["Quel est le rôle du cœur ?", "À quoi servent les poumons ?", "Comment le cœur et les poumons participent-ils à la circulation ?"]),
    ("Capital France", "La capitale de la France est Paris.", ["Quelle est la capitale de la France ?", "Dans quelle ville se trouve la capitale française ?", "Paris est-elle la capitale de la France ?"]),
    ("Capital Senegal", "La capitale du Sénégal est Dakar.", ["Quelle est la capitale du Sénégal ?", "Dakar est la capitale de quel pays ?", "Dans quelle ville se trouve la capitale sénégalaise ?"]),
    ("Capital Japan", "La capitale du Japon est Tokyo.", ["Quelle est la capitale du Japon ?", "Tokyo est la capitale de quel pays ?", "Comment s'appelle la capitale japonaise ?"]),
    ("Capital Canada", "La capitale du Canada est Ottawa.", ["Quelle est la capitale du Canada ?", "Ottawa est la capitale de quel pays ?", "Comment s'appelle la capitale canadienne ?"]),
    ("Capital Australia", "La capitale de l'Australie est Canberra.", ["Quelle est la capitale de l'Australie ?", "Canberra est la capitale de quel pays ?", "Comment s'appelle la capitale australienne ?"]),
    ("Capital Kenya", "La capitale du Kenya est Nairobi.", ["Quelle est la capitale du Kenya ?", "Nairobi est la capitale de quel pays ?", "Comment s'appelle la capitale kenyane ?"]),
    ("Capital Brazil", "La capitale du Brésil est Brasília.", ["Quelle est la capitale du Brésil ?", "Brasília est la capitale de quel pays ?", "Comment s'appelle la capitale brésilienne ?"]),
    ("Capital Morocco", "La capitale du Maroc est Rabat.", ["Quelle est la capitale du Maroc ?", "Rabat est la capitale de quel pays ?", "Comment s'appelle la capitale marocaine ?"]),
    ("Capital DRC", "La capitale de la République démocratique du Congo est Kinshasa.", ["Quelle est la capitale de la République démocratique du Congo ?", "Kinshasa est la capitale de quel pays ?", "Comment s'appelle la capitale de la RDC ?"]),
    ("Capital Belgium", "La capitale de la Belgique est Bruxelles.", ["Quelle est la capitale de la Belgique ?", "Bruxelles est la capitale de quel pays ?", "Comment s'appelle la capitale belge ?"]),
    ("Capital Italy", "La capitale de l'Italie est Rome.", ["Quelle est la capitale de l'Italie ?", "Rome est la capitale de quel pays ?", "Comment s'appelle la capitale italienne ?"]),
    ("Capital Spain", "La capitale de l'Espagne est Madrid.", ["Quelle est la capitale de l'Espagne ?", "Madrid est la capitale de quel pays ?", "Comment s'appelle la capitale espagnole ?"]),
    ("Capital Germany", "La capitale de l'Allemagne est Berlin.", ["Quelle est la capitale de l'Allemagne ?", "Berlin est la capitale de quel pays ?", "Comment s'appelle la capitale allemande ?"]),
    ("Capital Rwanda", "La capitale du Rwanda est Kigali.", ["Quelle est la capitale du Rwanda ?", "Kigali est la capitale de quel pays ?", "Comment s'appelle la capitale rwandaise ?"]),
    ("Capital Egypt", "La capitale de l'Égypte est Le Caire.", ["Quelle est la capitale de l'Égypte ?", "Le Caire est la capitale de quel pays ?", "Comment s'appelle la capitale égyptienne ?"]),
    ("Capital Portugal", "La capitale du Portugal est Lisbonne.", ["Quelle est la capitale du Portugal ?", "Lisbonne est la capitale de quel pays ?", "Comment s'appelle la capitale portugaise ?"]),
    ("French noun", "En français, un nom désigne notamment une personne, un animal, un objet, un lieu ou une idée. Il a un genre grammatical.", ["Qu'est-ce qu'un nom en grammaire française ?", "Que désigne un nom ?", "Le nom français a-t-il un genre ?"]),
    ("French verb", "Un verbe exprime souvent une action ou un état. Il se conjugue selon des éléments comme la personne, le nombre, le temps et le mode.", ["À quoi sert un verbe ?", "Qu'est-ce qu'un verbe en français ?", "Selon quoi un verbe se conjugue-t-il ?"]),
    ("French adjective", "Un adjectif apporte une précision sur un nom et s'accorde généralement avec lui en genre et en nombre.", ["Quel est le rôle d'un adjectif ?", "Comment s'accorde généralement un adjectif ?", "Qu'est-ce qu'un adjectif qualificatif ?"]),
    ("French punctuation", "Le point termine généralement une phrase déclarative. Le point d'interrogation marque une question directe.", ["Quand utiliser un point d'interrogation ?", "À quoi sert le point en fin de phrase ?", "Quel signe termine une question directe ?"]),
    ("Synonym", "Des synonymes ont des sens proches, mais ils ne sont pas toujours interchangeables dans tous les contextes.", ["Qu'est-ce qu'un synonyme ?", "Deux synonymes peuvent-ils toujours se remplacer ?", "Explique le mot synonyme."]),
    ("Antonym", "Un antonyme est un mot de sens opposé ou contraire à un autre mot dans un contexte donné.", ["Qu'est-ce qu'un antonyme ?", "Comment appelle-t-on un mot de sens contraire ?", "Donne la définition d'antonyme."]),
    ("Summary", "Pour résumer, repère l'idée principale, garde les informations essentielles et reformule-les brièvement avec tes propres mots.", ["Comment faire un résumé clair ?", "Quelle méthode suivre pour résumer un texte ?", "Que faut-il garder dans un résumé ?"]),
    ("Translation", "Pour traduire, il faut transmettre le sens et le ton dans la langue cible. Une traduction mot à mot peut produire une phrase incorrecte.", ["Comment réussir une traduction ?", "Pourquoi éviter une traduction toujours mot à mot ?", "Que faut-il préserver en traduisant ?"]),
    ("Paragraph", "Un paragraphe regroupe des phrases qui développent généralement une même idée. Une nouvelle idée peut justifier un nouveau paragraphe.", ["À quoi sert un paragraphe ?", "Quand commencer un nouveau paragraphe ?", "Comment organiser un texte en paragraphes ?"]),
    ("Email", "Un courriel clair présente son sujet, formule la demande avec précision et indique les éléments utiles. Un objet descriptif aide le destinataire.", ["Comment écrire un courriel clair ?", "Que mettre dans l'objet d'un courriel ?", "Comment rendre une demande par courriel plus facile à comprendre ?"]),
    ("Study", "Pour réviser, répartis les séances dans le temps, essaie de rappeler les notions sans regarder tes notes et vérifie ensuite les points oubliés.", ["Comment mieux réviser un cours ?", "Qu'est-ce que le rappel actif ?", "Comment organiser des révisions utiles ?"]),
    ("Clarification", "Quand une demande est ambiguë, pose une question courte sur le détail manquant avant de supposer ce que la personne veut.", ["Que faire si une question est ambiguë ?", "Comment répondre quand il manque un détail ?", "Pourquoi demander une précision ?"]),
    ("Uncertainty", "Si une information manque ou peut avoir changé, indique la limite clairement et demande une source ou un contexte supplémentaire au besoin.", ["Que faire quand tu n'es pas sûr ?", "Comment répondre sans inventer une information ?", "Comment signaler une incertitude ?"]),
    ("Units", "Avant de comparer ou d'additionner des mesures, convertis-les dans des unités compatibles. Par exemple, 1 mètre vaut 100 centimètres.", ["Pourquoi convertir les unités avant un calcul ?", "Combien de centimètres y a-t-il dans un mètre ?", "Comment comparer deux mesures exprimées différemment ?"]),
    ("Percentage", "Un pourcentage exprime une proportion sur cent. Pour calculer p % d'une quantité x, on calcule x multiplié par p puis divisé par 100.", ["Que signifie un pourcentage ?", "Comment calculer 15 % d'une quantité ?", "À quoi correspond le symbole % ?"]),
    ("Average", "La moyenne arithmétique d'une liste de nombres est leur somme divisée par le nombre de valeurs.", ["Comment calculer une moyenne arithmétique ?", "Quelle est la formule de la moyenne ?", "Comment trouver la moyenne de plusieurs nombres ?"]),
    ("Prime", "Un nombre premier est un entier supérieur à 1 qui possède exactement deux diviseurs positifs : 1 et lui-même.", ["Qu'est-ce qu'un nombre premier ?", "Combien de diviseurs a un nombre premier ?", "Le nombre 1 est-il premier ?"]),
    ("Fraction", "Une fraction représente un quotient. Son dénominateur ne peut pas être zéro, car une division par zéro n'est pas définie.", ["Que représente une fraction ?", "Pourquoi le dénominateur ne peut-il pas être zéro ?", "Comment lire a sur b en mathématiques ?"]),
    ("Area rectangle", "L'aire d'un rectangle se calcule en multipliant sa longueur par sa largeur, à condition d'exprimer les deux mesures dans la même unité.", ["Comment calculer l'aire d'un rectangle ?", "Quelle est la formule de l'aire d'un rectangle ?", "Quelle unité obtient-on pour une aire ?"]),
    ("Perimeter", "Le périmètre est la longueur totale du contour d'une figure. Pour un rectangle, il vaut deux fois la somme de la longueur et de la largeur.", ["Qu'est-ce que le périmètre ?", "Comment calculer le périmètre d'un rectangle ?", "Quelle différence entre aire et périmètre ?"]),
    ("Time", "Une heure contient 60 minutes et une minute contient 60 secondes.", ["Combien de minutes dans une heure ?", "Combien de secondes dans une minute ?", "Comment convertir des heures en minutes ?"]),
    ("Temperature", "Pour convertir des degrés Celsius en kelvins, on ajoute 273,15. Une différence de température de 1 °C correspond à une différence de 1 K.", ["Comment convertir Celsius en kelvins ?", "Quelle différence entre l'échelle Celsius et kelvin ?", "Un degré Celsius de différence vaut combien de kelvins ?"]),
    ("Climate weather", "La météo décrit les conditions atmosphériques à court terme. Le climat résume des tendances observées sur de longues périodes.", ["Quelle différence entre météo et climat ?", "Le climat et la météo désignent-ils la même chose ?", "Sur quelle durée parle-t-on du climat ?"]),
    ("Recycling", "Le tri permet d'orienter certains matériaux vers des filières de recyclage. Les consignes varient selon la commune et le type de déchet.", ["Pourquoi trier les déchets ?", "Peut-on suivre les mêmes consignes de tri partout ?", "Comment savoir où jeter un emballage ?"]),
    ("Water saving", "Pour économiser l'eau, on peut réparer les fuites, fermer le robinet pendant le brossage des dents et choisir des usages adaptés à ses besoins.", ["Comment réduire sa consommation d'eau ?", "Donne des gestes simples pour économiser l'eau.", "Pourquoi réparer un robinet qui fuit ?"]),
    ("First aid", "En cas d'urgence médicale, contacte les services d'urgence locaux. Pour un geste de premiers secours, suis une formation reconnue et les consignes des professionnels.", ["Que faire en cas d'urgence médicale ?", "Où demander de l'aide pour des premiers secours ?", "Peux-tu remplacer une formation de secourisme ?"]),
    ("Medical limits", "Je peux donner des informations générales, mais je ne peux pas diagnostiquer une personne. Pour un symptôme inquiétant, demande conseil à un professionnel de santé.", ["Peux-tu diagnostiquer mes symptômes ?", "Que faire si un symptôme m'inquiète ?", "Tes réponses remplacent-elles un médecin ?"]),
    ("Legal limits", "Je peux expliquer des notions générales, mais les règles dépendent du pays et de la situation. Pour une décision juridique, consulte une source officielle ou un professionnel qualifié.", ["Peux-tu donner une réponse juridique certaine ?", "Les lois sont-elles les mêmes partout ?", "Où vérifier une règle de droit ?"]),
    ("Financial limits", "Je peux expliquer des concepts financiers généraux, mais un choix dépend de ta situation, de tes objectifs et des risques. Vérifie les informations auprès de sources fiables.", ["Peux-tu garantir un placement rentable ?", "Comment évaluer un conseil financier ?", "Un investissement comporte-t-il toujours des risques ?"]),
    ("Privacy", "Évite de partager des informations personnelles sensibles dans un service si ce n'est pas nécessaire. Vérifie les paramètres et la politique de confidentialité du service utilisé.", ["Comment protéger mes informations personnelles ?", "Dois-je partager mon adresse ou mes mots de passe ?", "Que vérifier avant d'utiliser un service en ligne ?"]),
    ("Copyright", "Le droit d'auteur et les exceptions applicables varient selon les pays et les usages. Pour réutiliser une œuvre, vérifie sa licence et les règles locales.", ["Puis-je réutiliser n'importe quelle image trouvée en ligne ?", "Comment vérifier les droits d'une œuvre ?", "Les règles de droit d'auteur sont-elles identiques partout ?"]),
    ("Current info", "Je ne peux pas confirmer une information en temps réel sans source ou outil à jour. Donne-moi un lien ou un extrait et je peux aider à l'analyser.", ["Quelle est la dernière actualité ?", "Peux-tu vérifier le prix actuel d'un produit ?", "Comment obtenir une réponse sur une information récente ?"]),
    ("Creator", "Je suis gopu-lm, un petit modèle de langage écrit en Rust. Je ne peux pas confirmer l'identité personnelle de son créateur à partir des seules informations présentes dans cette conversation.", ["Qui es-tu ?", "Qui a créé gopu-lm ?", "Quel est ton nom et que peux-tu confirmer sur ton origine ?"]),
]


def conversation(question, answer):
    # Le champ thinking décrit la tâche sans exposer de raisonnement détaillé.
    return (f"<|user|>\n{question}\n</|user|>\n"
            f"<|thinking|>\nJe réponds avec une explication concise et signale les limites utiles.\n</|thinking|>\n"
            f"<|assistant|>\n{answer}\n</|assistant|>\n<|eos|>\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data/train.txt", help="fichier de corpus à enrichir")
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--repeat", type=int, default=3, help="nombre de formulations par entrée")
    args = parser.parse_args()
    if args.repeat < 1:
        parser.error("--repeat doit être supérieur ou égal à 1")
    rng = random.Random(args.seed)
    additions = []
    for _round in range(args.repeat):
        entries = list(FACTS)
        rng.shuffle(entries)
        for _, answer, questions in entries:
            additions.append(conversation(rng.choice(questions), answer))
    path = Path(args.out)
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    # Une relance avec la même graine ne duplique pas les conversations déjà présentes.
    additions = [item for item in additions if item not in existing]
    with path.open("a", encoding="utf-8") as corpus:
        if additions and path.stat().st_size:
            corpus.write("\n")
        corpus.writelines(additions)
    print(f"{len(additions)} conversations ajoutées à {args.out} ({len(FACTS)} thèmes)")


if __name__ == "__main__":
    main()
