#!/usr/bin/env python3
"""Stage 2 : Generation des reponses teacher (tau=0.9) - 500 questions Pokemon"""

import json
import time
import random
import datetime
from pathlib import Path
from openai import OpenAI

# === CONFIG ===
API_KEY = "nKuJabWS1epvq3x-m8by6NOU4xP4_znNL9OhmgXBPz9OeWOHlyGJIENnG8oXLT-4oOXNmESqExEMZv6o"
BASE_URL = "https://api.infomaniak.com/2/ai/48/openai/v1"
TEACHER_MODEL = "openai/gpt-oss-120b"
LOCAL_DATA_DIR = Path("/Users/adam/Documents/GitHub/deep_4_al/cours/TP/tp4/data")

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

SYSTEM_PROMPT = (
    "Tu es un expert Pokemon competitif. "
    "Pour chaque question, raisonne etape par etape a l'interieur de balises <reasoning>...</reasoning> "
    "avant de donner ta reponse finale. "
    "Sois precis, utilise les vrais noms francais des Pokemon, et structure clairement ton raisonnement "
    "avec des etapes numerotees."
)

# === BASE DE DONNEES POKEMON ===
POKEMONS = {
    "Bulbizarre": ["Plante", "Poison"], "Herbizarre": ["Plante", "Poison"], "Florizarre": ["Plante", "Poison"],
    "Salamèche": ["Feu"], "Reptincel": ["Feu"], "Dracaufeu": ["Feu", "Vol"],
    "Carapuce": ["Eau"], "Carabaffe": ["Eau"], "Tortank": ["Eau"],
    "Pikachu": ["Electrik"], "Raichu": ["Electrik"], "Raichu d'Alola": ["Electrik", "Psy"],
    "Rondoudou": ["Normal", "Fée"], "Grodoudou": ["Normal", "Fée"],
    "Nosferapti": ["Poison", "Vol"], "Nosferalto": ["Poison", "Vol"], "Nostenfer": ["Poison", "Vol"],
    "Abra": ["Psy"], "Kadabra": ["Psy"], "Alakazam": ["Psy"],
    "Machoc": ["Combat"], "Machopeur": ["Combat"], "Mackogneur": ["Combat"],
    "Fantominus": ["Spectre", "Poison"], "Spectrum": ["Spectre", "Poison"], "Ectoplasma": ["Spectre", "Poison"],
    "Onix": ["Roche", "Sol"], "Steelix": ["Acier", "Sol"],
    "Insécateur": ["Insecte", "Vol"], "Cizayox": ["Insecte", "Acier"],
    "Elektek": ["Electrik"], "Elekable": ["Electrik"],
    "Magmar": ["Feu"], "Maganon": ["Feu"],
    "Magicarpe": ["Eau"], "Léviator": ["Eau", "Vol"],
    "Métamorph": ["Normal"],
    "Evoli": ["Normal"], "Aquali": ["Eau"], "Voltali": ["Electrik"], "Pyroli": ["Feu"],
    "Mentali": ["Psy"], "Noctali": ["Ténèbres"], "Phyllali": ["Plante"], "Givrali": ["Glace"], "Nymphali": ["Fée"],
    "Lokhlass": ["Eau", "Glace"],
    "Dracolosse": ["Dragon", "Vol"], "Minidraco": ["Dragon"], "Draco": ["Dragon"],
    "Mewtwo": ["Psy"], "Mew": ["Psy"],
    "Artikodin": ["Glace", "Vol"], "Électhor": ["Electrik", "Vol"], "Sulfura": ["Feu", "Vol"],
    "Ronflex": ["Normal"], "Scarabrute": ["Insecte"], "Tauros": ["Normal"],
    "Rhinocorne": ["Sol", "Roche"], "Rhinoféros": ["Sol", "Roche"], "Rhinastoc": ["Sol", "Roche"],
    "Leveinard": ["Normal"], "Leuphorie": ["Normal"],
    "Arcanin": ["Feu"], "Caninos": ["Feu"],
    "Grolem": ["Roche", "Sol"], "Grolem d'Alola": ["Roche", "Electrik"],
    "Hypnomade": ["Psy"], "Excelangue": ["Normal"], "Coudlangue": ["Normal"],
    "Germignon": ["Plante"], "Macronium": ["Plante"], "Méganium": ["Plante"],
    "Héricendre": ["Feu"], "Feurisson": ["Feu"], "Typhlosion": ["Feu"],
    "Kaiminus": ["Eau"], "Crocrodil": ["Eau"], "Aligatueur": ["Eau"],
    "Noctowl": ["Normal", "Vol"], "Coxyclaque": ["Insecte", "Vol"],
    "Pharamp": ["Electrik"], "Wattouat": ["Electrik"],
    "Joliflor": ["Plante"], "Granivol": ["Plante", "Vol"],
    "Scorplane": ["Sol", "Vol"], "Scorvol": ["Sol", "Vol"],
    "Qulbutoké": ["Psy"], "Zarbi": ["Psy"],
    "Feuforêve": ["Spectre"], "Magirêve": ["Spectre"],
    "Scarhino": ["Insecte", "Combat"],
    "Ursaring": ["Normal"], "Teddiursa": ["Normal"],
    "Démoloss": ["Ténèbres", "Feu"], "Malosse": ["Ténèbres", "Feu"],
    "Hyporoi": ["Eau", "Dragon"], "Hypocéan": ["Eau"],
    "Porygon2": ["Normal"], "Porygon-Z": ["Normal"],
    "Cerfrousse": ["Normal"], "Insolourdo": ["Normal"],
    "Tyranocif": ["Roche", "Ténèbres"], "Ymphect": ["Roche", "Ténèbres"],
    "Lugia": ["Psy", "Vol"], "Ho-Oh": ["Feu", "Vol"],
    "Celebi": ["Psy", "Plante"],
    "Raikou": ["Electrik"], "Entei": ["Feu"], "Suicune": ["Eau"],
    "Arcko": ["Plante"], "Massko": ["Plante"], "Jungko": ["Plante"],
    "Poussifeu": ["Feu"], "Galifeu": ["Feu", "Combat"], "Braségali": ["Feu", "Combat"],
    "Gobou": ["Eau"], "Flobio": ["Eau", "Sol"], "Laggron": ["Eau", "Sol"],
    "Gardevoir": ["Psy", "Fée"], "Gallame": ["Psy", "Combat"], "Kirlia": ["Psy", "Fée"],
    "Tengalice": ["Plante", "Ténèbres"], "Pifeuil": ["Plante", "Ténèbres"],
    "Hariyama": ["Combat"], "Makuhita": ["Combat"],
    "Mysdibule": ["Acier", "Fée"], "Ténéfix": ["Ténèbres", "Spectre"],
    "Charmina": ["Combat", "Psy"], "Méditikka": ["Combat", "Psy"],
    "Sharpedo": ["Eau", "Ténèbres"], "Carvanha": ["Eau", "Ténèbres"],
    "Wailord": ["Eau"], "Wailmer": ["Eau"],
    "Chamallot": ["Feu", "Sol"], "Camérupt": ["Feu", "Sol"],
    "Chartor": ["Feu"],
    "Libégon": ["Sol", "Dragon"], "Vibraninf": ["Sol", "Dragon"], "Kraknoix": ["Sol"],
    "Altaria": ["Dragon", "Vol"], "Tylton": ["Normal", "Vol"],
    "Mangriff": ["Normal"], "Séviper": ["Poison"],
    "Barpau": ["Eau"], "Milobellus": ["Eau"],
    "Absol": ["Ténèbres"],
    "Branette": ["Spectre"], "Polichombr": ["Spectre"],
    "Mélodelfe": ["Fée"], "Mélofée": ["Fée"],
    "Métalosse": ["Acier", "Psy"], "Métang": ["Acier", "Psy"], "Terhal": ["Acier", "Psy"],
    "Drattack": ["Dragon"], "Drattak": ["Dragon", "Vol"],
    "Latias": ["Dragon", "Psy"], "Latios": ["Dragon", "Psy"],
    "Kyogre": ["Eau"], "Groudon": ["Sol"], "Rayquaza": ["Dragon", "Vol"],
    "Jirachi": ["Acier", "Psy"], "Deoxys": ["Psy"],
    "Regirock": ["Roche"], "Regice": ["Glace"], "Registeel": ["Acier"],
    "Tortipouss": ["Plante"], "Boskara": ["Plante"], "Torterra": ["Plante", "Sol"],
    "Ouisticram": ["Feu"], "Chimpenfeu": ["Feu", "Combat"], "Simiabraz": ["Feu", "Combat"],
    "Tiplouf": ["Eau"], "Prinplouf": ["Eau"], "Pingoléon": ["Eau", "Acier"],
    "Etouraptor": ["Normal", "Vol"], "Etourvol": ["Normal", "Vol"],
    "Carchacrok": ["Dragon", "Sol"], "Carmache": ["Dragon", "Sol"], "Griknot": ["Dragon", "Sol"],
    "Lucario": ["Combat", "Acier"], "Riolu": ["Combat"],
    "Hippodocus": ["Sol"], "Hippopotas": ["Sol"],
    "Cradopaud": ["Poison", "Combat"], "Coatox": ["Poison", "Combat"],
    "Corboss": ["Ténèbres", "Vol"], "Cornebre": ["Ténèbres", "Vol"],
    "Moufouette": ["Poison", "Ténèbres"], "Moufflair": ["Poison", "Ténèbres"],
    "Goinfrex": ["Normal"], "Munchlax": ["Normal"],
    "Archeodong": ["Acier", "Psy"], "Archéomire": ["Acier", "Psy"],
    "Spiritomb": ["Spectre", "Ténèbres"],
    "Roserade": ["Plante", "Poison"], "Rosélia": ["Plante", "Poison"],
    "Momartik": ["Glace", "Spectre"], "Stalgamin": ["Glace"],
    "Dimoret": ["Ténèbres", "Glace"], "Farfuret": ["Ténèbres", "Glace"],
    "Togekiss": ["Fée", "Vol"], "Togetic": ["Fée", "Vol"], "Togepi": ["Fée"],
    "Motisma": ["Electrik", "Spectre"],
    "Cresselia": ["Psy"], "Darkrai": ["Ténèbres"],
    "Dialga": ["Acier", "Dragon"], "Palkia": ["Eau", "Dragon"], "Giratina": ["Spectre", "Dragon"],
    "Heatran": ["Feu", "Acier"],
    "Regigigas": ["Normal"], "Arceus": ["Normal"],
    "Manaphy": ["Eau"], "Phione": ["Eau"], "Shaymin": ["Plante"],
    "Vipélierre": ["Plante"], "Lianaja": ["Plante"], "Majaspic": ["Plante"],
    "Gruikui": ["Feu"], "Grotichon": ["Feu", "Combat"], "Roitiflam": ["Feu", "Combat"],
    "Moustillon": ["Eau"], "Mateloutre": ["Eau"], "Clamiral": ["Eau"],
    "Gueriaigle": ["Normal", "Vol"],
    "Zoroark": ["Ténèbres"], "Zorua": ["Ténèbres"],
    "Tranchodon": ["Dragon"], "Incisache": ["Dragon"], "Coupenotte": ["Dragon"],
    "Minotaupe": ["Sol", "Acier"], "Rototaupe": ["Sol"],
    "Cryptéro": ["Psy", "Vol"], "Limonde": ["Sol", "Electrik"],
    "Mygavolt": ["Insecte", "Electrik"], "Brutalibre": ["Combat", "Vol"],
    "Tutafeh": ["Spectre"], "Tutankafer": ["Spectre"],
    "Escargaume": ["Insecte"], "Limaspeed": ["Insecte"],
    "Fermite": ["Insecte", "Acier"],
    "Cobaltium": ["Acier", "Combat"], "Terrakium": ["Roche", "Combat"], "Viridium": ["Plante", "Combat"],
    "Boréas": ["Vol"], "Fulguris": ["Electrik", "Vol"], "Démétéros": ["Sol", "Vol"],
    "Reshiram": ["Dragon", "Feu"], "Zekrom": ["Dragon", "Electrik"], "Kyurem": ["Dragon", "Glace"],
    "Keldeo": ["Eau", "Combat"], "Genesect": ["Insecte", "Acier"],
    "Méloetta": ["Normal", "Psy"], "Victini": ["Psy", "Feu"],
    "Marisson": ["Plante"], "Boguérisse": ["Plante"], "Blindépique": ["Plante", "Combat"],
    "Feunnec": ["Feu"], "Roussil": ["Feu"], "Goupelin": ["Feu", "Psy"],
    "Grenousse": ["Eau"], "Croâporal": ["Eau"], "Amphinobi": ["Eau", "Ténèbres"],
    "Sapereau": ["Normal"], "Excavarenne": ["Normal", "Sol"],
    "Flambusard": ["Feu", "Vol"], "Braisillon": ["Feu", "Vol"],
    "Prismillon": ["Insecte", "Vol"],
    "Flabébé": ["Fée"], "Floette": ["Fée"], "Florges": ["Fée"],
    "Dragmara": ["Poison", "Dragon"], "Brutalibré": ["Combat", "Vol"],
    "Sonistrelle": ["Vol", "Dragon"], "Bruyverne": ["Vol", "Dragon"],
    "Dedenne": ["Electrik", "Fée"],
    "Mucuscule": ["Dragon"], "Colimucus": ["Dragon"], "Muplodocus": ["Dragon"],
    "Couafarel": ["Normal"], "Trousselin": ["Acier", "Fée"],
    "Xerneas": ["Fée"], "Yveltal": ["Ténèbres", "Vol"], "Zygarde": ["Dragon", "Sol"],
    "Diancie": ["Roche", "Fée"], "Hoopa": ["Psy", "Spectre"], "Volcanion": ["Feu", "Eau"],
    "Brindibou": ["Plante", "Vol"], "Efflèche": ["Plante", "Vol"], "Archéduc": ["Plante", "Spectre"],
    "Flamiaou": ["Feu"], "Matoufeu": ["Feu"], "Félinferno": ["Feu", "Ténèbres"],
    "Otaquin": ["Eau"], "Otarlette": ["Eau"], "Oratoria": ["Eau", "Fée"],
    "Manglouton": ["Normal"], "Argouste": ["Normal"],
    "Bombydou": ["Insecte", "Fée"], "Rubombelle": ["Insecte", "Fée"],
    "Rocabot": ["Roche"], "Lougaroc": ["Roche"],
    "Froussardine": ["Eau"], "Mimiqui": ["Spectre", "Fée"],
    "Dodoala": ["Normal"], "Togedemaru": ["Electrik", "Acier"],
    "Draïeul": ["Normal", "Dragon"], "Silvallié": ["Normal"],
    "Cosmog": ["Psy"], "Cosmovum": ["Psy"], "Solgaleo": ["Psy", "Acier"], "Lunala": ["Psy", "Spectre"],
    "Necrozma": ["Psy"],
    "Tokopisco": ["Eau", "Fée"], "Tokorico": ["Electrik", "Fée"], "Tokopiyon": ["Psy", "Fée"], "Tokotoro": ["Plante", "Fée"],
    "Zeraora": ["Electrik"], "Marshadow": ["Combat", "Spectre"], "Magearna": ["Acier", "Fée"],
    "Ouistempo": ["Plante"], "Badabouin": ["Plante"], "Gorythmic": ["Plante"],
    "Flambino": ["Feu"], "Lapyro": ["Feu"], "Pyrobut": ["Feu"],
    "Larméléon": ["Eau"], "Arrozard": ["Eau"], "Lézargus": ["Eau"],
    "Corvaillus": ["Vol", "Acier"], "Palarticho": ["Combat"], "Sorcilence": ["Psy"],
    "Pachyradjah": ["Sol"], "Nigosier": ["Vol", "Eau"],
    "Hexadron": ["Dragon", "Spectre"], "Duralugon": ["Acier", "Dragon"],
    "Zacian": ["Fée"], "Zamazenta": ["Combat"],
    "Éthernatos": ["Poison", "Dragon"], "Sylveroy": ["Psy", "Plante"],
    "Regieleki": ["Electrik"], "Regidrago": ["Dragon"],
    "Poussacha": ["Plante"], "Matourgeon": ["Plante"], "Miascarade": ["Plante", "Ténèbres"],
    "Chochodile": ["Feu"], "Crocogril": ["Feu"], "Flâmigator": ["Feu", "Spectre"],
    "Coiffeton": ["Eau"], "Canarbello": ["Eau"], "Palmaval": ["Eau"],
    "Pohm": ["Electrik"], "Pohmotte": ["Electrik", "Combat"], "Pohmarmotte": ["Electrik", "Combat"],
    "Tapatoès": ["Normal", "Vol"],
    "Chongjian": ["Ténèbres", "Plante"], "Baojian": ["Ténèbres", "Glace"],
    "Tinglu": ["Ténèbres", "Sol"], "Yuyu": ["Ténèbres", "Feu"],
    "Koraidon": ["Combat", "Dragon"], "Miraidon": ["Electrik", "Dragon"],
    "Serpente": ["Acier", "Dragon"],
}

TYPES = ["Normal", "Feu", "Eau", "Plante", "Electrik", "Glace", "Combat", "Poison",
         "Sol", "Vol", "Psy", "Insecte", "Roche", "Spectre", "Dragon", "Ténèbres", "Acier", "Fée"]

ATTAQUES = {
    "Lance-Flammes": {"type": "Feu", "puissance": 90, "categorie": "Spécial"},
    "Déflagration": {"type": "Feu", "puissance": 110, "categorie": "Spécial"},
    "Surchauffe": {"type": "Feu", "puissance": 130, "categorie": "Spécial"},
    "Boutefeu": {"type": "Feu", "puissance": 120, "categorie": "Physique"},
    "Flamme Ultime": {"type": "Feu", "puissance": 150, "categorie": "Spécial"},
    "Poing de Feu": {"type": "Feu", "puissance": 75, "categorie": "Physique"},
    "Roue de Feu": {"type": "Feu", "puissance": 60, "categorie": "Physique"},
    "Surf": {"type": "Eau", "puissance": 90, "categorie": "Spécial"},
    "Hydrocanon": {"type": "Eau", "puissance": 110, "categorie": "Spécial"},
    "Hydroqueue": {"type": "Eau", "puissance": 90, "categorie": "Physique"},
    "Cascade": {"type": "Eau", "puissance": 80, "categorie": "Physique"},
    "Ébullition": {"type": "Eau", "puissance": 80, "categorie": "Spécial"},
    "Aqua-Jet": {"type": "Eau", "puissance": 40, "categorie": "Physique"},
    "Ocroupi": {"type": "Eau", "puissance": 65, "categorie": "Spécial"},
    "Tempête Florale": {"type": "Plante", "puissance": 90, "categorie": "Spécial"},
    "Lance-Soleil": {"type": "Plante", "puissance": 120, "categorie": "Spécial"},
    "Végé-Attak": {"type": "Plante", "puissance": 150, "categorie": "Spécial"},
    "Lame-Feuille": {"type": "Plante", "puissance": 90, "categorie": "Physique"},
    "Martobois": {"type": "Plante", "puissance": 120, "categorie": "Physique"},
    "Fouet Lianes": {"type": "Plante", "puissance": 45, "categorie": "Physique"},
    "Tonnerre": {"type": "Electrik", "puissance": 90, "categorie": "Spécial"},
    "Fatal-Foudre": {"type": "Electrik", "puissance": 110, "categorie": "Spécial"},
    "Éclair Fou": {"type": "Electrik", "puissance": 90, "categorie": "Physique"},
    "Coup d'Jus": {"type": "Electrik", "puissance": 80, "categorie": "Spécial"},
    "Électacle": {"type": "Electrik", "puissance": 120, "categorie": "Physique"},
    "Cage-Éclair": {"type": "Electrik", "puissance": 0, "categorie": "Statut"},
    "Laser Glace": {"type": "Glace", "puissance": 90, "categorie": "Spécial"},
    "Blizzard": {"type": "Glace", "puissance": 110, "categorie": "Spécial"},
    "Lyophilisation": {"type": "Glace", "puissance": 70, "categorie": "Spécial"},
    "Poinglace": {"type": "Glace", "puissance": 75, "categorie": "Physique"},
    "Éclats Glace": {"type": "Glace", "puissance": 40, "categorie": "Physique"},
    "Close Combat": {"type": "Combat", "puissance": 120, "categorie": "Physique"},
    "Aurasphère": {"type": "Combat", "puissance": 80, "categorie": "Spécial"},
    "Surpuissance": {"type": "Combat", "puissance": 120, "categorie": "Physique"},
    "Poing Boost": {"type": "Combat", "puissance": 40, "categorie": "Physique"},
    "Vampipoing": {"type": "Combat", "puissance": 75, "categorie": "Physique"},
    "Balayette": {"type": "Combat", "puissance": 60, "categorie": "Physique"},
    "Séisme": {"type": "Sol", "puissance": 100, "categorie": "Physique"},
    "Telluriforce": {"type": "Sol", "puissance": 90, "categorie": "Spécial"},
    "Tunnel": {"type": "Sol", "puissance": 80, "categorie": "Physique"},
    "Piétisol": {"type": "Sol", "puissance": 60, "categorie": "Physique"},
    "Mille Flèches": {"type": "Sol", "puissance": 90, "categorie": "Physique"},
    "Rapace": {"type": "Vol", "puissance": 120, "categorie": "Physique"},
    "Vent Violent": {"type": "Vol", "puissance": 110, "categorie": "Spécial"},
    "Aéropique": {"type": "Vol", "puissance": 60, "categorie": "Physique"},
    "Atterrissage": {"type": "Vol", "puissance": 0, "categorie": "Statut"},
    "Danse-Plume": {"type": "Vol", "puissance": 0, "categorie": "Statut"},
    "Psyko": {"type": "Psy", "puissance": 90, "categorie": "Spécial"},
    "Psycho Boost": {"type": "Psy", "puissance": 140, "categorie": "Spécial"},
    "Extrasenseur": {"type": "Psy", "puissance": 80, "categorie": "Spécial"},
    "Coupe Psycho": {"type": "Psy", "puissance": 70, "categorie": "Physique"},
    "Psyfrape": {"type": "Psy", "puissance": 0, "categorie": "Statut"},
    "Vibrobscur": {"type": "Ténèbres", "puissance": 80, "categorie": "Spécial"},
    "Tranche-Nuit": {"type": "Ténèbres", "puissance": 70, "categorie": "Physique"},
    "Représailles": {"type": "Ténèbres", "puissance": 50, "categorie": "Physique"},
    "Coup Bas": {"type": "Ténèbres", "puissance": 70, "categorie": "Physique"},
    "Machination": {"type": "Ténèbres", "puissance": 0, "categorie": "Statut"},
    "Trou Noir": {"type": "Ténèbres", "puissance": 80, "categorie": "Spécial"},
    "Draco-Météore": {"type": "Dragon", "puissance": 130, "categorie": "Spécial"},
    "Colère": {"type": "Dragon", "puissance": 120, "categorie": "Physique"},
    "Draco-Griffe": {"type": "Dragon", "puissance": 80, "categorie": "Physique"},
    "Dracosouffle": {"type": "Dragon", "puissance": 60, "categorie": "Spécial"},
    "Hurlement": {"type": "Dragon", "puissance": 0, "categorie": "Statut"},
    "Danse Draco": {"type": "Dragon", "puissance": 0, "categorie": "Statut"},
    "Ball'Ombre": {"type": "Spectre", "puissance": 80, "categorie": "Spécial"},
    "Griffe Ombre": {"type": "Spectre", "puissance": 70, "categorie": "Physique"},
    "Châtiment": {"type": "Spectre", "puissance": 50, "categorie": "Physique"},
    "Revenant": {"type": "Spectre", "puissance": 120, "categorie": "Physique"},
    "Onde Folie": {"type": "Spectre", "puissance": 0, "categorie": "Statut"},
    "Luminocanon": {"type": "Acier", "puissance": 80, "categorie": "Spécial"},
    "Poing Météore": {"type": "Acier", "puissance": 90, "categorie": "Physique"},
    "Tête de Fer": {"type": "Acier", "puissance": 80, "categorie": "Physique"},
    "Gyroballe": {"type": "Acier", "puissance": 0, "categorie": "Physique"},
    "Queue de Fer": {"type": "Acier", "puissance": 100, "categorie": "Physique"},
    "Pouvoir Lunaire": {"type": "Fée", "puissance": 95, "categorie": "Spécial"},
    "Éclat Magique": {"type": "Fée", "puissance": 80, "categorie": "Spécial"},
    "Câlinerie": {"type": "Fée", "puissance": 90, "categorie": "Physique"},
    "Garde Mystik": {"type": "Fée", "puissance": 0, "categorie": "Statut"},
    "Lame de Roc": {"type": "Roche", "puissance": 100, "categorie": "Physique"},
    "Éboulement": {"type": "Roche", "puissance": 75, "categorie": "Physique"},
    "Rochers Furtifs": {"type": "Roche", "puissance": 0, "categorie": "Statut"},
    "Tomberoche": {"type": "Roche", "puissance": 60, "categorie": "Physique"},
    "Gemme de Roc": {"type": "Roche", "puissance": 80, "categorie": "Spécial"},
    "Bombe Beurk": {"type": "Poison", "puissance": 90, "categorie": "Spécial"},
    "Cradovague": {"type": "Poison", "puissance": 95, "categorie": "Spécial"},
    "Poison-Croix": {"type": "Poison", "puissance": 70, "categorie": "Physique"},
    "Pics Toxik": {"type": "Poison", "puissance": 0, "categorie": "Statut"},
    "Toxik": {"type": "Poison", "puissance": 0, "categorie": "Statut"},
    "Bourdon": {"type": "Insecte", "puissance": 90, "categorie": "Spécial"},
    "Mégacorne": {"type": "Insecte", "puissance": 120, "categorie": "Physique"},
    "Demi-Tour": {"type": "Insecte", "puissance": 70, "categorie": "Physique"},
    "Plaie-Croix": {"type": "Insecte", "puissance": 80, "categorie": "Physique"},
    "Ultralaser": {"type": "Normal", "puissance": 150, "categorie": "Spécial"},
    "Giga Impact": {"type": "Normal", "puissance": 150, "categorie": "Physique"},
    "Retour": {"type": "Normal", "puissance": 102, "categorie": "Physique"},
    "Plaquage": {"type": "Normal", "puissance": 85, "categorie": "Physique"},
    "Vive-Attaque": {"type": "Normal", "puissance": 40, "categorie": "Physique"},
    "Tranche": {"type": "Normal", "puissance": 70, "categorie": "Physique"},
    "Façade": {"type": "Normal", "puissance": 70, "categorie": "Physique"},
}

TALENTS = [
    "Intimidation", "Lévitation", "Adaptabilité", "Torrent", "Brasier", "Engrais",
    "Essaim", "Statik", "Écaille Spéciale", "Multiécaille", "Peau Dure", "Absorb Eau",
    "Pare-Feu", "Paratonnerre", "Impudence", "Inconscient", "Fermeté", "Attention",
    "Pression", "Synchro", "Régé-Force", "Régénération", "Turbo", "Chlorophylle",
    "Glissade", "Pied Véloce", "Sable Cachot", "Garde Magik", "Armurouillée",
    "Télépathe", "Matinal", "Sans Limite", "Technicien", "Optimiste",
    "Ailes Bourrasque", "Peau Miracle", "Protéen", "Libéro", "Garde-Corps",
]

OBJETS = [
    "Orbe Vie", "Bandeau Choix", "Lunettes Choix", "Écharpe Choix", "Restes",
    "Ceinture Pro", "Baie Sitrus", "Gilet d'Assaut", "Ballon", "Veste de Combat",
    "Focus Sash", "Pierre Évolitive", "Vive Griffe", "Balle Fer", "Mouchoir Soie",
    "Lentilscope", "Herbe Blanche", "Gemme Normal", "Orbe Toxique", "Orbe Flamme",
    "Boue Noire", "Casque Brut", "Câble Garde", "Baguette Rose", "Plaque Terre",
    "Plaque Draco", "Plaque Fer", "Méga-Canne", "Écaille de Dragon", "Écharpe Soie",
]

NATURES = [
    {"nom": "Jovial", "boost": "Vitesse", "nerf": "Attaque Spéciale"},
    {"nom": "Rigide", "boost": "Attaque", "nerf": "Attaque Spéciale"},
    {"nom": "Modeste", "boost": "Attaque Spéciale", "nerf": "Attaque"},
    {"nom": "Timide", "boost": "Vitesse", "nerf": "Attaque"},
    {"nom": "Prudent", "boost": "Défense Spéciale", "nerf": "Attaque Spéciale"},
    {"nom": "Calme", "boost": "Défense Spéciale", "nerf": "Attaque"},
    {"nom": "Assuré", "boost": "Défense", "nerf": "Attaque Spéciale"},
    {"nom": "Malin", "boost": "Défense", "nerf": "Vitesse"},
    {"nom": "Hardi", "boost": None, "nerf": None},
]

FORMATS = ["OU (OverUsed)", "UU (UnderUsed)", "Ubers", "NU (NeverUsed)", "RU (RarelyUsed)",
           "LC (Little Cup)", "VGC (Video Game Championships)", "Doubles OU", "Monotype", "Anything Goes"]

ROLES = ["sweeper physique", "sweeper spécial", "wall physique", "wall spécial",
         "mixed attacker", "pivot", "revenge killer", "support", "lead", "tank",
         "stallbreaker", "hazard setter", "hazard remover (spinner/defogger)", "cleric"]

STRATEGIES = ["Trick Room", "Danse Pluie", "Zénith", "Grêle", "Tempête de Sable",
              "Baton Pass", "Stall", "Hyper Offense", "Balance", "Bulky Offense",
              "Sun Team", "Rain Team", "Sand Team", "Hail Team"]

TEMPLATES = {
    "efficacite_types": [
        "Quelle est l'efficacité d'une attaque de type {type1} contre un Pokémon de type {type2} ? Explique le multiplicateur.",
        "Un {pokemon1} ({types1}) utilise {attaque} contre un {pokemon2} ({types2}). Analyse l'efficacité de type.",
        "Pourquoi le type {type1} est-il super efficace / peu efficace / inefficace contre le type {type2} ?",
        "Quelles sont toutes les faiblesses et résistances d'un Pokémon de type {type1}/{type2} ?",
        "Un {pokemon1} affronte un {pokemon2}. Analyse les interactions de types offensives et défensives pour chaque côté.",
        "Explique pourquoi {pokemon1} ({types1}) a une double faiblesse au type {type1}.",
        "Quels types sont immunisés aux attaques de type {type1} et pourquoi ?",
        "Compare les faiblesses de {pokemon1} ({types1}) et {pokemon2} ({types2}). Lequel est le plus vulnérable ?",
    ],
    "stab_et_degats": [
        "Explique le concept de STAB et calcule les dégâts relatifs de {attaque} utilisé par {pokemon1} vs par {pokemon2}.",
        "Si {pokemon1} ({types1}) utilise {attaque} (type {type_attaque}, puissance {puissance}), quel est le multiplicateur STAB ?",
        "Compare les dégâts de {attaque1} et {attaque2} utilisés par {pokemon1}. Lequel est plus efficace avec STAB ?",
        "Un {pokemon1} tient un {objet}. Calcule le boost de dégâts sur {attaque}.",
        "Quelle est la formule de calcul des dégâts en Pokémon ? Applique-la à {pokemon1} utilisant {attaque} contre {pokemon2}.",
    ],
    "stats_et_comparaisons": [
        "Compare les stats de base de {pokemon1} et {pokemon2}. Lequel est meilleur offensivement ? Défensivement ?",
        "Quel est le BST (Base Stat Total) de {pokemon1} et comment sont réparties ses stats ?",
        "Pourquoi {pokemon1} est-il considéré comme un bon {role} malgré son BST de {bst} ?",
        "Analyse les stats de {pokemon1} : est-il plutôt offensif, défensif ou équilibré ?",
        "Compare les speed tiers de {pokemon1}, {pokemon2} et {pokemon3}. Qui attaque en premier ?",
        "Avec la nature {nature} et 252 EVs en {stat}, quelle sera la stat finale de {pokemon1} niveau 100 ?",
        "{pokemon1} a-t-il de meilleures stats défensives ou offensives ? Justifie avec les valeurs.",
    ],
    "evolution": [
        "Décris la chaîne d'évolution complète de {pokemon1} et les conditions pour chaque évolution.",
        "Quelles sont toutes les évolutions possibles de {pokemon1} et quelle est la meilleure en compétitif ?",
        "Compare {pokemon1} avant et après évolution. Quels gains de stats et quels nouveaux types ?",
        "Comment faire évoluer {pokemon1} en {pokemon2} ? Quelles conditions sont nécessaires ?",
        "Pourquoi certains joueurs gardent {pokemon1} non évolué (avec Évoluroc) plutôt que de le faire évoluer ?",
        "Explique le mécanisme de la Méga-Évolution avec l'exemple de {pokemon1}. Quels changements de stats et de type ?",
        "Quelles sont les différentes formes régionales de {pokemon1} et leurs différences de type ?",
    ],
    "team_building": [
        "Construis une équipe équilibrée autour de {pokemon1} comme {role} principal. Explique les synergies.",
        "Quels Pokémon forment un bon core défensif avec {pokemon1} ? Analyse la couverture de types.",
        "Comment contrer efficacement {pokemon1} en format {format} ? Propose 3 contre-mesures.",
        "{pokemon1} et {pokemon2} forment-ils un bon duo offensif ? Analyse leur complémentarité.",
        "Construis un core Feu/Eau/Plante autour de {pokemon1}. Quels Pokémon choisir ?",
        "En format {format}, quels sont les meilleurs partenaires pour {pokemon1} ?",
        "Propose un moveset compétitif pour {pokemon1} en tant que {role} avec l'objet {objet}.",
        "Quelle équipe {strategie} inclut idéalement {pokemon1} ? Explique son rôle.",
    ],
    "mecaniques_combat": [
        "Explique comment fonctionne le talent {talent} et quels Pokémon en bénéficient le plus.",
        "Quel est l'intérêt de l'objet {objet} sur {pokemon1} ? Dans quelles situations l'utiliser ?",
        "Comment les Rochers Furtifs affectent-ils {pokemon1} ({types1}) à l'entrée sur le terrain ?",
        "Explique la mécanique de priorité des attaques. Pourquoi {attaque} passe-t-elle avant les autres ?",
        "Quelle est la différence entre une attaque physique et spéciale ? Comment cela affecte-t-il {pokemon1} ?",
        "Comment fonctionne la météo {strategie} et quels Pokémon en profitent ?",
        "Explique le concept de 'pivot' avec l'exemple de {pokemon1} utilisant Demi-Tour ou Volt-Switch.",
        "Pourquoi certains Pokémon utilisent-ils {attaque} malgré sa faible puissance ?",
    ],
    "formats_et_tiers": [
        "Pourquoi {pokemon1} est-il classé en {format} ? Qu'est-ce qui le rend si fort/faible ?",
        "Compare l'utilisation de {pokemon1} en {format} vs en VGC. Quelles différences de sets ?",
        "Quels sont les Pokémon les plus utilisés en {format} et pourquoi dominent-ils le métagame ?",
        "Pourquoi {pokemon1} a-t-il été banni en Ubers ? Qu'est-ce qui le rendait trop puissant en OU ?",
        "En VGC, pourquoi {pokemon1} est-il plus populaire qu'en Singles OU ?",
        "Quelles sont les principales menaces contre {pokemon1} en {format} ?",
    ],
    "calculs_avances": [
        "Calcule les dégâts exacts de {attaque} (puissance {puissance}) utilisé par {pokemon1} (Attaque {stat}) contre {pokemon2} (Défense 100).",
        "Avec un boost de Danse Draco (+1 Attaque, +1 Vitesse), {pokemon1} peut-il OHKO {pokemon2} avec {attaque} ?",
        "Quelle est la probabilité qu'un {pokemon1} survive à {attaque} de {pokemon2} avec les EVs optimaux ?",
        "Compare les dégâts de {attaque1} (puissance {puissance1}) vs {attaque2} (puissance {puissance2}) sur {pokemon2}.",
        "Si {pokemon1} tient {objet} et utilise {attaque} après un boost de +2, quels dégâts fera-t-il ?",
    ],
    "strategies_specifiques": [
        "Comment monter une équipe {strategie} efficace ? Quels sont les Pokémon clés ?",
        "Comment contrer une équipe basée sur {strategie} ? Propose des solutions.",
        "Explique le rôle de {pokemon1} dans une équipe {strategie}. Quel set utiliser ?",
        "Quels sont les avantages et inconvénients d'une stratégie {strategie} en {format} ?",
        "Comment {pokemon1} peut-il être utilisé comme wallbreaker contre les équipes stall ?",
    ],
}


def get_pokemon_types(pokemon):
    types = POKEMONS.get(pokemon, ["Normal"])
    return "/".join(types)

def get_random_attaque_info():
    nom = random.choice(list(ATTAQUES.keys()))
    info = ATTAQUES[nom]
    return nom, info

def generate_question(template_category):
    template = random.choice(TEMPLATES[template_category])
    pokemon1 = random.choice(list(POKEMONS.keys()))
    pokemon2 = random.choice([p for p in POKEMONS.keys() if p != pokemon1])
    pokemon3 = random.choice([p for p in POKEMONS.keys() if p not in [pokemon1, pokemon2]])
    attaque1_nom, attaque1_info = get_random_attaque_info()
    attaque2_nom, attaque2_info = get_random_attaque_info()
    nature = random.choice(NATURES)
    question = template.format(
        type1=random.choice(TYPES), type2=random.choice(TYPES),
        pokemon1=pokemon1, pokemon2=pokemon2, pokemon3=pokemon3,
        types1=get_pokemon_types(pokemon1), types2=get_pokemon_types(pokemon2),
        attaque=attaque1_nom, attaque1=attaque1_nom, attaque2=attaque2_nom,
        type_attaque=attaque1_info["type"], puissance=attaque1_info["puissance"],
        puissance1=attaque1_info["puissance"], puissance2=attaque2_info["puissance"],
        talent=random.choice(TALENTS), objet=random.choice(OBJETS),
        nature=nature["nom"],
        stat=random.choice(["Attaque", "Défense", "Attaque Spéciale", "Défense Spéciale", "Vitesse", "PV"]),
        role=random.choice(ROLES), format=random.choice(FORMATS),
        strategie=random.choice(STRATEGIES),
        bst=random.choice([480, 500, 520, 540, 570, 580, 600, 670, 680]),
    )
    return question


def generate_teacher_response(question, temperature=0.3, max_retries=3):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question}
    ]
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=TEACHER_MODEL,
                messages=messages,
                temperature=temperature,
                max_tokens=4096,
                logprobs=True,
                top_logprobs=1
            )
            content = response.choices[0].message.content
            logprobs_data = response.choices[0].logprobs
            if len(content) < 100:
                print(f"  [Retry {attempt+1}] Reponse trop courte ({len(content)} chars)")
                time.sleep(2 ** attempt)
                continue
            serialized_logprobs = []
            if logprobs_data and logprobs_data.content:
                for token_data in logprobs_data.content:
                    serialized_logprobs.append({
                        "token": token_data.token,
                        "logprob": token_data.logprob
                    })
            return {
                "content": content,
                "logprobs": serialized_logprobs,
                "has_reasoning": "<reasoning>" in content and "</reasoning>" in content,
                "temperature": temperature
            }
        except Exception as e:
            wait = 2 ** (attempt + 1)
            print(f"  [Retry {attempt+1}] Erreur: {e} - attente {wait}s")
            time.sleep(wait)
    print(f"  ECHEC apres {max_retries} tentatives")
    return None


# === GENERATION DES 1000 QUESTIONS ===
random.seed(42)
all_questions = []
categories = list(TEMPLATES.keys())
generated_questions = set()
attempts = 0
while len(all_questions) < 1000 and attempts < 5000:
    cat = categories[len(all_questions) % len(categories)]
    q = generate_question(cat)
    if q not in generated_questions:
        generated_questions.add(q)
        all_questions.append({"category": cat, "question": q})
    attempts += 1

random.shuffle(all_questions)
all_questions_stage2 = all_questions[500:]

print(f"Questions Stage 2 pretes : {len(all_questions_stage2)}")

# === GENERATION STAGE 2 (tau=0.9) ===
stage2_raw = []
save_path = Path("/Users/adam/Documents/GitHub/deep_4_al/cours/TP/tp4/data/stage2_raw_1000.json")

print("=" * 60)
print("STAGE 2 : Generation a haute temperature (tau=0.9)")
print(f"Debut : {datetime.datetime.now().strftime('%H:%M:%S')}")
print("=" * 60)

for i, item in enumerate(all_questions_stage2):
    print(f"[{i+1}/500] {item['category'][:15]:15s} | {item['question'][:55]}...", end=" ", flush=True)
    result = generate_teacher_response(item["question"], temperature=0.9)
    if result:
        stage2_raw.append({
            "category": item["category"],
            "question": item["question"],
            "response": result["content"],
            "logprobs": result["logprobs"],
            "has_reasoning": result["has_reasoning"],
            "temperature": result["temperature"]
        })
        print(f"OK {len(result['content'])}c", flush=True)
    else:
        print("ECHEC", flush=True)
    if (i + 1) % 25 == 0:
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(stage2_raw, f, ensure_ascii=False, indent=2)
        print(f"  >>> Sauvegarde : {len(stage2_raw)} exemples ({datetime.datetime.now().strftime('%H:%M:%S')})")
    time.sleep(0.5)

with open(save_path, "w", encoding="utf-8") as f:
    json.dump(stage2_raw, f, ensure_ascii=False, indent=2)

print(f"\nStage 2 TERMINE : {len(stage2_raw)}/500")
print(f"Avec raisonnement : {sum(1 for r in stage2_raw if r['has_reasoning'])}")
print(f"Sauvegarde : {save_path}")
print(f"Fin : {datetime.datetime.now().strftime('%H:%M:%S')}")
